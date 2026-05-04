import os
import re
from typing import cast

import numpy as np
from maad import sound

from .params_loader import ConfigParams
from .utils import AudioMetadata


class PreProcessor:
    """
    Pre-process one loaded audio container into segment-level outputs.

    The processor performs segmentation first and then applies clipping checks,
    optional resampling, and optional normalization to each valid segment.
    Segment-level failures are returned as failed containers instead of raising.

    Parameters
    ----------
    config : ConfigParams
        Configuration object containing parameters for pre-processing steps.
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
        Run full preprocessing for one loaded audio container.
        
        Parameters
        ----------
        audio : AudioMetadata
            The loaded audio file to process.
        
        Returns
        -------
        list[AudioMetadata]
            Segment-level outputs. Items may be successful processed segments or
            failures produced during segmentation or subsequent processing.
        """

        # Step 1: Segment the audio into fixed-duration segments with padding as needed.
        segmented = self._segment(audio)

        processed_segments: list[AudioMetadata] = []
        for segmented_item in segmented:
            if not segmented_item.ok:
                processed_segments.append(segmented_item)
                continue

            seg_wave = cast(np.ndarray, segmented_item.wave)
            seg_sample_rate = cast(int, segmented_item.sample_rate)
            seg_file_name = segmented_item.file_name
            seg_segment_id = segmented_item.segment_id
            seg_directory_name = segmented_item.directory_name

            # Step 2: Check for saturation/clipping
            try:
                if self._check_saturation(seg_wave):
                    processed_segments.append(
                        AudioMetadata(
                            stage="preprocess",
                            status="failed",
                            file_name=seg_file_name,
                            segment_id=seg_segment_id,
                            directory_name=seg_directory_name,
                            error="Segment is clipped",
                            error_type="ClippingError",
                        )
                    )
                    continue
                
                # Step 3: Downsample if configured and target sample rate is lower than original
                out_wave, out_sample_rate = self._downsample(seg_wave, seg_sample_rate)

                # Step 4: Normalize if configured
                out_wave = self._normalize(out_wave)

                processed_segments.append(
                    AudioMetadata(
                        stage="preprocess",
                        status="success",
                        file_name=seg_file_name,
                        segment_id=seg_segment_id,
                        directory_name=seg_directory_name,
                        wave=out_wave,
                        sample_rate=out_sample_rate,
                    )
                )
            except Exception as exc:
                processed_segments.append(
                    AudioMetadata(
                        stage="preprocess",
                        status="failed",
                        file_name=seg_file_name,
                        segment_id=seg_segment_id,
                        directory_name=seg_directory_name,
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                )

        return processed_segments

    def _segment(self, audio: AudioMetadata) -> list[AudioMetadata]:
        """
        Split one loaded audio into fixed-duration segment containers.

        Parameters
        ----------
        audio : AudioMetadata
            Input audio container from the load stage.

        Returns
        -------
        list[AudioMetadata]
            One container per segment candidate.
            Full segments are returned as success.
            Tail segments are either padded and accepted (success) or rejected
            (failed) according to segment_tolerance_percent.
        """

        file_name = audio.file_name
        directory_name = audio.directory_name
        base_segment_id = audio.segment_id or file_name
        try:
            wave = cast(np.ndarray, audio.wave)
            sample_rate = cast(int, audio.sample_rate)

            total_samples = wave.size
            segment_samples = int(self.segment_duration * sample_rate)

            full_segments = total_samples // segment_samples
            tail_samples = total_samples % segment_samples

            prefix, ext = os.path.splitext(file_name)
            match = self._HHMMSS_RE.search(file_name)
            base_seconds = 0
            if match:
                hhmmss = match.group(1)
                base_seconds = int(hhmmss[0:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:6])

            def build_segment_file_name(segment_index: int, start_seconds: int) -> str:
                """Build the output filename for a segment start time."""
                if match:
                    segment_seconds = base_seconds + start_seconds
                    hh = segment_seconds // 3600
                    mm = (segment_seconds % 3600) // 60
                    ss = segment_seconds % 60
                    return self._HHMMSS_RE.sub(
                        f"_{hh:02d}{mm:02d}{ss:02d}",
                        file_name,
                        count=1,
                    )
                return f"{prefix}_seg{segment_index + 1:03d}{ext or '.wav'}"

            segments: list[AudioMetadata] = []
            for index in range(full_segments):
                start_t = index * self.segment_duration
                end_t = start_t + self.segment_duration
                segment_signal = sound.trim(
                    wave,
                    fs=sample_rate,
                    min_t=float(start_t),
                    max_t=float(end_t),
                    pad=True,
                    pad_constant=0,
                )
                segment_file_name = build_segment_file_name(index, start_t)
                segment_id = f"{base_segment_id}#start={start_t}s"

                segments.append(
                    AudioMetadata(
                        stage="preprocess",
                        status="success",
                        file_name=segment_file_name,
                        segment_id=segment_id,
                        directory_name=directory_name,
                        wave=segment_signal,
                        sample_rate=sample_rate,
                    )
                )

            if tail_samples > 0:
                tail_duration = tail_samples / float(sample_rate)
                missing_duration = self.segment_duration - tail_duration
                missing_percent = (missing_duration / self.segment_duration) * 100.0

                tail_index = full_segments
                start_t = tail_index * self.segment_duration
                tail_file_name = build_segment_file_name(tail_index, start_t)
                tail_segment_id = f"{base_segment_id}#start={start_t}s"

                if missing_percent <= self.segment_tolerance_percent:
                    tail_signal = sound.trim(
                        wave,
                        fs=sample_rate,
                        min_t=float(start_t),
                        max_t=float(start_t + self.segment_duration),
                        pad=True,
                        pad_constant=0,
                    )
                    segments.append(
                        AudioMetadata(
                            stage="preprocess",
                            status="success",
                            file_name=tail_file_name,
                            segment_id=tail_segment_id,
                            directory_name=directory_name,
                            wave=tail_signal,
                            sample_rate=sample_rate,
                        )
                    )
                else:
                    segments.append(
                        AudioMetadata(
                            stage="preprocess",
                            status="failed",
                            file_name=tail_file_name,
                            segment_id=tail_segment_id,
                            directory_name=directory_name,
                            error=f"Discarded segment: {tail_duration:.1f}s (expected {self.segment_duration}s, tol {self.segment_tolerance_percent}%)",
                            error_type="SegmentToleranceError",
                        )
                    )

            return segments
        except Exception as exc:
            return [
                AudioMetadata(
                    stage="preprocess",
                    status="failed",
                    file_name=file_name,
                    segment_id=base_segment_id,
                    directory_name=directory_name,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            ]

    def _check_saturation(
        self,
        wave: np.ndarray,
    ) -> bool:
        """
        Check whether a segment exceeds the configured clipping threshold.

        Parameters
        ----------
        wave : np.ndarray
            Segment waveform data.

        Returns
        -------
        bool
            True if the segment is clipped, otherwise False.
        """
        return bool(np.max(np.abs(wave)) >= self.clipping_threshold)

    def _downsample(
        self,
        wave: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, int]:
        """
        Resample a segment to target_sample_rate when target is lower.

        Parameters
        ----------
        wave : np.ndarray
            Segment waveform data.
        sample_rate : int
            Segment sampling rate.

        Returns
        -------
        tuple[np.ndarray, int]
            Tuple containing (wave, sample_rate) after optional resampling.
            If target_sample_rate is None or not lower than sample_rate,
            the original inputs are returned unchanged.
        """
        if self.target_sample_rate is None:
            return wave, sample_rate

        target_sample_rate = cast(int, self.target_sample_rate)

        if target_sample_rate >= sample_rate:
            return wave, sample_rate

        resampled = sound.resample(
            wave,
            fs=sample_rate,
            target_fs=target_sample_rate
        )
        return resampled, target_sample_rate

    def _normalize(
        self,
        wave: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize a segment waveform to max amplitude 1.0.

        Parameters
        ----------
        wave : np.ndarray
            Segment waveform data.

        Returns
        -------
        np.ndarray
            Normalized waveform.
        """
        if not self.normalize_audio:
            return wave
            
        return sound.normalize(wave, max_amp=1.0)
