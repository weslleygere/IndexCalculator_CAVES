import os
import re

import numpy as np
from maad import sound

from .params_loader import ConfigParams
from .utils import AudioMetadata


class PreProcessor:
    """
    Pre-process one loaded audio file into segment-level outputs.

    The processor splits each audio file into fixed-duration segments, rejects
    clipped segments, optionally downsamples and optionally normalizes valid
    segments.

    Parameters
    ----------
    config : ConfigParams
        Configuration object containing preprocessing parameters.
    """

    _HHMMSS_RE = re.compile(r"_(\d{6})(?=\.[wW][aA][vV]$)")

    def __init__(self, config: ConfigParams) -> None:
        self.clipping_threshold = config.clipping_threshold
        self.segment_duration = config.segment_duration
        self.target_sample_rate = config.target_sample_rate
        self.normalize_audio = config.normalize_audio
        self.segment_tolerance_percent = config.segment_tolerance_percent

    def process_file(self, audio: AudioMetadata) -> list[AudioMetadata]:
        """
        Run preprocessing for one loaded audio file.

        Parameters
        ----------
        audio : AudioMetadata
            Loaded audio container.

        Returns
        -------
        list[AudioMetadata]
            Segment-level preprocessing results.
        """
        segments = self._segment(audio)

        for segment in segments:
            if segment.ok:
                self._process_segment(segment)

        audio.wave = None

        return segments

    def _segment(self, audio: AudioMetadata) -> list[AudioMetadata]:
        """
        Split one loaded audio file into fixed-duration segments.

        Parameters
        ----------
        audio : AudioMetadata
            Loaded audio container.

        Returns
        -------
        list[AudioMetadata]
            Segment-level containers.
        """
        try:
            wave = audio.require_wave()
            sample_rate = audio.require_sample_rate()

            segment_samples = self.segment_duration * sample_rate
            if segment_samples <= 0:
                raise ValueError("Segment size must be positive")

            total_samples = wave.size
            if total_samples == 0:
                raise ValueError("Empty audio file")

            full_segments = total_samples // segment_samples
            tail_samples = total_samples % segment_samples

            segments = [
                self._build_segment(
                    source=audio,
                    wave=wave,
                    sample_rate=sample_rate,
                    segment_index=index,
                    segment_samples=segment_samples,
                    pad=False,
                )
                for index in range(full_segments)
            ]

            tail_segment = self._build_tail_segment(
                source=audio,
                wave=wave,
                sample_rate=sample_rate,
                full_segments=full_segments,
                tail_samples=tail_samples,
                segment_samples=segment_samples,
            )

            if tail_segment is not None:
                segments.append(tail_segment)

            if not segments:
                raise ValueError("No segments were produced")

            return segments

        except Exception as exc:
            return [
                AudioMetadata.fail(
                    stage="preprocess",
                    file_name=audio.file_name,
                    directory_name=audio.directory_name,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            ]

    def _process_segment(self, segment: AudioMetadata) -> None:
        """
        Apply clipping check, downsampling, and normalization to one segment.

        Parameters
        ----------
        segment : AudioMetadata
            Segment metadata object modified in place.
        """
        try:
            wave = segment.require_wave()
            sample_rate = segment.require_sample_rate()

            if self._is_clipped(wave):
                raise ValueError("Segment is clipped")

            wave, sample_rate = self._downsample(wave, sample_rate)
            wave = self._normalize(wave)

            segment.stage = "preprocess"
            segment.status = "success"
            segment.wave = wave
            segment.sample_rate = sample_rate
            segment.error = None
            segment.error_type = None

        except Exception as exc:
            self._fail_segment(
                segment=segment,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    def _build_tail_segment(
        self,
        source: AudioMetadata,
        wave: np.ndarray,
        sample_rate: int,
        full_segments: int,
        tail_samples: int,
        segment_samples: int,
    ) -> AudioMetadata | None:
        """
        Build the tail segment if it exists and satisfies tolerance settings.

        Parameters
        ----------
        source : AudioMetadata
            Source audio container.
        wave : np.ndarray
            Source waveform.
        sample_rate : int
            Source sample rate.
        full_segments : int
            Number of full segments.
        tail_samples : int
            Number of remaining samples after full segmentation.
        segment_samples : int
            Samples per target segment.

        Returns
        -------
        AudioMetadata | None
            Tail segment, failed tail segment, or None if there is no tail.
        """
        if tail_samples == 0:
            return None

        tail_duration = tail_samples / float(sample_rate)
        missing_duration = self.segment_duration - tail_duration
        missing_percent = (missing_duration / self.segment_duration) * 100.0

        if missing_percent > self.segment_tolerance_percent:
            return self._segment_failure(
                source=source,
                segment_index=full_segments,
                error=(
                    f"Discarded segment: {tail_duration:.1f}s "
                    f"(expected {self.segment_duration}s, "
                    f"tol {self.segment_tolerance_percent}%)"
                ),
                error_type="SegmentToleranceError",
            )

        return self._build_segment(
            source=source,
            wave=wave,
            sample_rate=sample_rate,
            segment_index=full_segments,
            segment_samples=segment_samples,
            pad=True,
        )

    def _build_segment(
        self,
        source: AudioMetadata,
        wave: np.ndarray,
        sample_rate: int,
        segment_index: int,
        segment_samples: int,
        pad: bool,
    ) -> AudioMetadata:
        """
        Build one segment metadata object.

        Parameters
        ----------
        source : AudioMetadata
            Source audio container.
        wave : np.ndarray
            Source waveform.
        sample_rate : int
            Source sample rate.
        segment_index : int
            Zero-based segment index.
        segment_samples : int
            Number of samples per segment.
        pad : bool
            Whether to zero-pad incomplete segments.

        Returns
        -------
        AudioMetadata
            Segment metadata.
        """
        start_sample = segment_index * segment_samples
        end_sample = start_sample + segment_samples

        segment_wave = wave[start_sample:end_sample]

        if pad and segment_wave.size < segment_samples:
            segment_wave = np.pad(
                segment_wave,
                (0, segment_samples - segment_wave.size),
                mode="constant",
                constant_values=0,
            )

        start_seconds = segment_index * self.segment_duration
        segment_file_name = self._build_segment_file_name(
            file_name=source.file_name,
            segment_index=segment_index,
            start_seconds=start_seconds,
        )

        return AudioMetadata.success(
            stage="preprocess",
            file_name=segment_file_name,
            segment_id=segment_index,
            directory_name=source.directory_name,
            wave=segment_wave.astype(np.float32, copy=False),
            sample_rate=sample_rate,
        )

    def _segment_failure(
        self,
        source: AudioMetadata,
        segment_index: int,
        error: str,
        error_type: str,
    ) -> AudioMetadata:
        """
        Build a failed segment metadata object.

        Parameters
        ----------
        source : AudioMetadata
            Source audio container.
        segment_index : int
            Zero-based segment index.
        error : str
            Error message.
        error_type : str
            Error type.

        Returns
        -------
        AudioMetadata
            Failed segment metadata.
        """
        start_seconds = segment_index * self.segment_duration
        segment_file_name = self._build_segment_file_name(
            file_name=source.file_name,
            segment_index=segment_index,
            start_seconds=start_seconds,
        )

        return AudioMetadata.fail(
            stage="preprocess",
            file_name=segment_file_name,
            segment_id=segment_index,
            directory_name=source.directory_name,
            error=error,
            error_type=error_type,
        )

    def _build_segment_file_name(
        self,
        file_name: str,
        segment_index: int,
        start_seconds: int,
    ) -> str:
        """
        Build the segment file name.

        Parameters
        ----------
        file_name : str
            Original file name.
        segment_index : int
            Zero-based segment index.
        start_seconds : int
            Segment start time in seconds.

        Returns
        -------
        str
            Segment file name.
        """
        match = self._HHMMSS_RE.search(file_name)

        if not match:
            prefix, ext = os.path.splitext(file_name)
            return f"{prefix}_seg{segment_index + 1:03d}{ext or '.wav'}"

        hhmmss = match.group(1)
        base_seconds = (
            int(hhmmss[0:2]) * 3600
            + int(hhmmss[2:4]) * 60
            + int(hhmmss[4:6])
        )

        segment_seconds = base_seconds + start_seconds
        hh = segment_seconds // 3600
        mm = (segment_seconds % 3600) // 60
        ss = segment_seconds % 60

        return self._HHMMSS_RE.sub(
            f"_{hh:02d}{mm:02d}{ss:02d}",
            file_name,
            count=1,
        )

    def _is_clipped(self, wave: np.ndarray) -> bool:
        """
        Check whether a segment exceeds the clipping threshold.

        Parameters
        ----------
        wave : np.ndarray
            Segment waveform.

        Returns
        -------
        bool
            True if clipped.
        """
        return bool(np.max(np.abs(wave)) >= self.clipping_threshold)

    def _downsample(
        self,
        wave: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, int]:
        """
        Downsample a segment if target_sample_rate is configured and lower.

        Parameters
        ----------
        wave : np.ndarray
            Segment waveform.
        sample_rate : int
            Current sample rate.

        Returns
        -------
        tuple[np.ndarray, int]
            Waveform and sample rate after optional downsampling.
        """
        if self.target_sample_rate is None or self.target_sample_rate >= sample_rate:
            return wave, sample_rate

        resampled = sound.resample(
            wave,
            fs=sample_rate,
            target_fs=self.target_sample_rate,
        )

        return resampled.astype(np.float32, copy=False), self.target_sample_rate

    def _normalize(self, wave: np.ndarray) -> np.ndarray:
        """
        Normalize waveform if configured.

        Parameters
        ----------
        wave : np.ndarray
            Segment waveform.

        Returns
        -------
        np.ndarray
            Original or normalized waveform.
        """
        if not self.normalize_audio:
            return wave

        return sound.normalize(wave, max_amp=1.0).astype(np.float32, copy=False)

    @staticmethod
    def _fail_segment(
        segment: AudioMetadata,
        error: str,
        error_type: str,
    ) -> None:
        """
        Mark a segment as failed in place.

        Parameters
        ----------
        segment : AudioMetadata
            Segment to update.
        error : str
            Error message.
        error_type : str
            Error type.
        """
        segment.stage = "preprocess"
        segment.status = "failed"
        segment.wave = None
        segment.acoustic_idx = None
        segment.error = error
        segment.error_type = error_type
