import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AudioMetadata:
    """
    Standardized result container for audio processing stages.

    Parameters
    ----------
    stage : str
        Processing stage name.
    status : str
        Processing status: "success" or "failed".
    file_name : str
        File or segment file name.
    segment_id : int | None
        Segment identifier within the source file, or None for file-level results.
    directory_name : str | None
        Parent directory name.
    wave : np.ndarray | None
        Audio waveform data.
    acoustic_idx : dict[str, Any] | None
        Acoustic index values.
    sample_rate : int | None
        Audio sample rate.
    processing_time : float | None
        File-level processing time in seconds.
    error : str | None
        Error message when status is "failed".
    error_type : str | None
        Error type when status is "failed".
    """

    stage: str
    status: str
    file_name: str
    segment_id: int | None = None
    directory_name: str | None = None
    wave: np.ndarray | None = None
    acoustic_idx: dict[str, Any] | None = None
    sample_rate: int | None = None
    processing_time: float | None = None
    error: str | None = None
    error_type: str | None = None

    @property
    def ok(self) -> bool:
        """
        Whether the processing result is successful.

        Returns
        -------
        bool
            True if status is "success", otherwise False.
        """
        return self.status == "success"

    @classmethod
    def fail(
        cls,
        stage: str,
        file_name: str,
        *,
        segment_id: int | None = None,
        directory_name: str | None = None,
        error: str | None = None,
        error_type: str | None = None,
        sample_rate: int | None = None,
        processing_time: float | None = None,
    ) -> "AudioMetadata":
        """
        Build a failed metadata record.

        Parameters
        ----------
        stage : str
            Processing stage name.
        file_name : str
            File or segment file name.
        segment_id : int | None
            Segment identifier.
        directory_name : str | None
            Parent directory name.
        error : str | None
            Error message.
        error_type : str | None
            Error type.
        sample_rate : int | None
            Audio sample rate.
        processing_time : float | None
            File-level processing time in seconds.

        Returns
        -------
        AudioMetadata
            Failed metadata record.
        """
        return cls(
            stage=stage,
            status="failed",
            file_name=file_name,
            segment_id=segment_id,
            directory_name=directory_name,
            sample_rate=sample_rate,
            processing_time=processing_time,
            error=error,
            error_type=error_type,
        )

    @classmethod
    def success(
        cls,
        stage: str,
        file_name: str,
        *,
        segment_id: int | None = None,
        directory_name: str | None = None,
        wave: np.ndarray | None = None,
        acoustic_idx: dict[str, Any] | None = None,
        sample_rate: int | None = None,
        processing_time: float | None = None,
    ) -> "AudioMetadata":
        """
        Build a successful metadata record.

        Parameters
        ----------
        stage : str
            Processing stage name.
        file_name : str
            File or segment file name.
        segment_id : int | None
            Segment identifier.
        directory_name : str | None
            Parent directory name.
        wave : np.ndarray | None
            Audio waveform data.
        acoustic_idx : dict[str, Any] | None
            Acoustic index values.
        sample_rate : int | None
            Audio sample rate.
        processing_time : float | None
            File-level processing time in seconds.

        Returns
        -------
        AudioMetadata
            Successful metadata record.
        """
        return cls(
            stage=stage,
            status="success",
            file_name=file_name,
            segment_id=segment_id,
            directory_name=directory_name,
            wave=wave,
            acoustic_idx=acoustic_idx,
            sample_rate=sample_rate,
            processing_time=processing_time,
            error=None,
            error_type=None,
        )

    def require_wave(self, message: str = "Missing waveform") -> np.ndarray:
        """
        Return waveform or raise if missing.

        Parameters
        ----------
        message : str
            Error message for missing waveform.

        Returns
        -------
        np.ndarray
            Waveform array.
        """
        if self.wave is None:
            raise ValueError(message)

        return cast(np.ndarray, self.wave)

    def require_sample_rate(self, message: str = "Missing sample rate") -> int:
        """
        Return sample rate or raise if missing.

        Parameters
        ----------
        message : str
            Error message for missing sample rate.

        Returns
        -------
        int
            Sample rate.
        """
        if self.sample_rate is None:
            raise ValueError(message)

        return cast(int, self.sample_rate)


class FileScanner:
    """
    Recursively scan a directory for WAV files grouped by parent directory.

    Parameters
    ----------
    data_dir : Path | str
        Base directory containing WAV files.
    """

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir).resolve()

    def iter_audio_paths_by_directory(self) -> Iterator[tuple[str, list[Path]]]:
        """
        Yield WAV file paths grouped by directory in deterministic order.

        Yields
        ------
        tuple[str, list[Path]]
            Directory key and list of WAV paths.
        """
        for root, dirs, files in os.walk(self.data_dir):
            dirs.sort()

            wav_files = sorted(
                file for file in files
                if file.lower().endswith(".wav")
            )

            if not wav_files:
                continue

            root_path = Path(root)

            try:
                directory_key = root_path.relative_to(self.data_dir).as_posix()
            except ValueError:
                directory_key = root_path.name

            if directory_key == ".":
                directory_key = self.data_dir.name

            yield directory_key, [root_path / file for file in wav_files]


class CheckpointManager:
    """
    Manage processed-file checkpoint records.

    Parameters
    ----------
    output_dir : Path | str
        Directory where the checkpoint file is stored.
    data_dir : Path | str
        Base input directory used to compute relative checkpoint keys.
    checkpoint_file_name : str
        Checkpoint file name.
    """

    def __init__(
        self,
        output_dir: Path | str,
        data_dir: Path | str,
        checkpoint_file_name: str = "processed_files_checkpoint.txt",
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.output_dir / checkpoint_file_name
        self.data_dir_abs = Path(data_dir).resolve()

        self.records: dict[str, tuple[int, int, int]] = {}

    def check_resume_mode(self) -> bool:
        """
        Load checkpoint records if available.

        Returns
        -------
        bool
            True if checkpoint records were loaded.
        """
        if not self.checkpoint_path.exists():
            return False

        with self.checkpoint_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                parsed = self._parse_checkpoint_line(line)
                if parsed is None:
                    continue

                key, counts = parsed
                self.records[key] = counts

        return bool(self.records)

    def check_pending(self, paths: list[Path]) -> tuple[list[Path], int, int, int]:
        """
        Split paths into pending files and previously counted records.

        Parameters
        ----------
        paths : list[Path]
            Input file paths.

        Returns
        -------
        tuple[list[Path], int, int, int]
            Pending paths, prior successful segments, prior failed segments,
            and prior file-level failures.
        """
        pending: list[Path] = []
        seg_success = 0
        seg_failed = 0
        file_failed = 0

        for path in paths:
            key = self._get_key(path)

            if key not in self.records:
                pending.append(path)
                continue

            success_count, failed_count, file_failed_count = self.records[key]
            seg_success += success_count
            seg_failed += failed_count
            file_failed += file_failed_count

        return pending, seg_success, seg_failed, file_failed

    def mark_completed(self, file_path: Path, results: list[AudioMetadata]) -> tuple[int, int, int]:
        """
        Append the final file state to the checkpoint.

        Parameters
        ----------
        file_path : Path
            Original source audio file path.
        results : list[AudioMetadata]
            Final processing results for the file.

        Returns
        -------
        tuple[int, int, int]
            Successful segment count, failed segment count, and file-level
            failure count.
        """
        seg_success, seg_failed, file_failed = self._count_results(results)
        key = self._get_key(file_path)

        with self.checkpoint_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{key}|{seg_success}|{seg_failed}|{file_failed}\n")
            stream.flush()

        self.records[key] = (seg_success, seg_failed, file_failed)

        return seg_success, seg_failed, file_failed

    @staticmethod
    def _count_results(results: list[AudioMetadata]) -> tuple[int, int, int]:
        """
        Count successful segments, failed segments, and file-level failures.

        Parameters
        ----------
        results : list[AudioMetadata]
            Final processing results.

        Returns
        -------
        tuple[int, int, int]
            Successful segment count, failed segment count, and file-level
            failure count.
        """
        if not results:
            return 0, 0, 1

        if len(results) == 1 and not results[0].ok and results[0].stage == "load":
            return 0, 0, 1

        seg_success = sum(1 for result in results if result.ok)
        seg_failed = sum(1 for result in results if not result.ok)

        return seg_success, seg_failed, 0

    @staticmethod
    def _parse_checkpoint_line(
        line: str,
    ) -> tuple[str, tuple[int, int, int]] | None:
        """
        Parse one checkpoint line.

        Parameters
        ----------
        line : str
            Raw checkpoint line.

        Returns
        -------
        tuple[str, tuple[int, int, int]] | None
            Parsed key and counts, or None for invalid empty lines.
        """
        line = line.strip()

        if not line:
            return None

        parts = line.split("|")

        try:
            if len(parts) == 4:
                return parts[0], (int(parts[1]), int(parts[2]), int(parts[3]))

            if len(parts) == 3:
                return parts[0], (int(parts[1]), int(parts[2]), 0)

            if len(parts) == 2:
                is_success = parts[1] == "success"
                return parts[0], (
                    1 if is_success else 0,
                    0,
                    0 if is_success else 1,
                )

            return line, (1, 0, 0)

        except ValueError:
            logger.warning("Ignoring invalid checkpoint line: %s", line)
            return None

    def _get_key(self, file_path: Path | str) -> str:
        """
        Compute a stable checkpoint key for a file path.

        Parameters
        ----------
        file_path : Path | str
            File path.

        Returns
        -------
        str
            Relative path when possible, otherwise file name.
        """
        abs_file_path = Path(file_path).resolve()

        try:
            return abs_file_path.relative_to(self.data_dir_abs).as_posix()
        except ValueError:
            return abs_file_path.name


class CsvWriter:
    """
    Stream acoustic index results and errors to CSV files.

    Parameters
    ----------
    output_file : str | Path
        Acoustic indices CSV path.
    error_file : str | Path
        Error CSV path.
    resume : bool
        If True, append to an existing data CSV when available.
    """

    BASE_DATA_FIELDS = [
        "directory_name",
        "file_name",
        "segment_id",
        "sample_rate",
        "processing_time",
    ]

    ERROR_FIELDS = [
        "stage",
        "directory_name",
        "file_name",
        "segment_id",
        "error",
        "error_type",
    ]

    def __init__(
        self,
        output_file: str | Path,
        error_file: str | Path,
        resume: bool = False,
    ) -> None:
        self.output_file = Path(output_file)
        self.error_file = Path(error_file)
        self.resume = resume

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.error_file.parent.mkdir(parents=True, exist_ok=True)

        self._data_stream = None
        self._data_writer: csv.DictWriter[str] | None = None
        self._data_fieldnames: list[str] = []
        self._warned_extra_keys = False

        self._errors_stream = None
        self._errors_writer: csv.DictWriter[str] | None = None

    def __enter__(self) -> "CsvWriter":
        """
        Enter context manager.

        Returns
        -------
        CsvWriter
            Current writer instance.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Exit context manager and close file handles.
        """
        self.close()

    def consume(self, file_results: list[AudioMetadata]) -> None:
        """
        Write all final results from one source file.

        Parameters
        ----------
        file_results : list[AudioMetadata]
            Final processing results.
        """
        if not file_results:
            logger.warning("Received empty result list; no CSV row was written")
            return

        for result in file_results:
            if result.ok:
                self._write_success(result)
            else:
                self._write_error(result)

    def flush(self) -> None:
        """
        Flush open CSV streams.
        """
        if self._data_stream is not None:
            self._data_stream.flush()

        if self._errors_stream is not None:
            self._errors_stream.flush()

    def close(self) -> None:
        """
        Flush and close CSV streams.
        """
        if self._data_writer is None:
            self._ensure_data_writer(fieldnames=self.BASE_DATA_FIELDS)

        self.flush()

        if self._data_stream is not None:
            self._data_stream.close()
            self._data_stream = None

        if self._errors_stream is not None:
            self._errors_stream.close()
            self._errors_stream = None

    def _ensure_data_writer(self, fieldnames: list[str]) -> None:
        """
        Ensure the data CSV writer is ready.

        Parameters
        ----------
        fieldnames : list[str]
            Data CSV field names.
        """
        if self._data_writer is not None:
            return

        header = None
        data_exists = (
            self.resume
            and self.output_file.exists()
            and self.output_file.stat().st_size > 0
        )

        if data_exists:
            header = self._read_data_header()

        if header:
            self._data_fieldnames = header
            self._data_stream = self.output_file.open("a", newline="", encoding="utf-8")
        else:
            self._data_fieldnames = fieldnames
            self._data_stream = self.output_file.open("w", newline="", encoding="utf-8")

        self._data_writer = csv.DictWriter(
            self._data_stream,
            fieldnames=self._data_fieldnames,
        )

        if not header:
            self._data_writer.writeheader()

    def _read_data_header(self) -> list[str] | None:
        """
        Read the header from an existing data CSV file.

        Returns
        -------
        list[str] | None
            Header list if present.
        """
        with self.output_file.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)

        return header if header else None

    def _ensure_error_writer(self) -> None:
        """
        Ensure the error CSV writer is ready.
        """
        if self._errors_writer is not None:
            return

        error_exists = self.error_file.exists() and self.error_file.stat().st_size > 0
        mode = "a" if error_exists else "w"

        self._errors_stream = self.error_file.open(mode, newline="", encoding="utf-8")
        self._errors_writer = csv.DictWriter(
            self._errors_stream,
            fieldnames=self.ERROR_FIELDS,
        )

        if not error_exists:
            self._errors_writer.writeheader()

    def _write_success(self, result: AudioMetadata) -> None:
        """
        Write one successful acoustic index row.

        Parameters
        ----------
        result : AudioMetadata
            Successful acoustic index result.
        """
        if not result.acoustic_idx:
            self._write_error(
                AudioMetadata.fail(
                    stage=result.stage,
                    file_name=result.file_name,
                    segment_id=result.segment_id,
                    directory_name=result.directory_name,
                    sample_rate=result.sample_rate,
                    processing_time=result.processing_time,
                    error="Successful result without acoustic_idx",
                    error_type="MissingAcousticIndicesError",
                )
            )
            return

        if self._data_writer is None:
            fieldnames = self.BASE_DATA_FIELDS + sorted(result.acoustic_idx.keys())
            self._ensure_data_writer(fieldnames)

        if self._data_writer is None:
            raise RuntimeError("Data writer was not initialized")

        row: dict[str, Any] = self._base_data_row(result)

        extra_keys = []
        for key, value in result.acoustic_idx.items():
            if key in self._data_fieldnames:
                row[key] = value
            else:
                extra_keys.append(key)

        if extra_keys and not self._warned_extra_keys:
            logger.warning(
                "Ignoring acoustic index keys not present in CSV header: %s",
                sorted(extra_keys),
            )
            self._warned_extra_keys = True

        self._data_writer.writerow(row)

    def _write_error(self, result: AudioMetadata) -> None:
        """
        Write one error row.

        Parameters
        ----------
        result : AudioMetadata
            Failed processing result.
        """
        if self._errors_writer is None:
            self._ensure_error_writer()

        if self._errors_writer is None:
            raise RuntimeError("Error writer was not initialized")

        self._errors_writer.writerow(
            {
                "stage": result.stage,
                "directory_name": result.directory_name,
                "file_name": result.file_name,
                "segment_id": result.segment_id,
                "error": result.error,
                "error_type": result.error_type,
            }
        )

    @classmethod
    def _base_data_row(cls, result: AudioMetadata) -> dict[str, Any]:
        """
        Build base metadata row for the acoustic indices CSV.

        Parameters
        ----------
        result : AudioMetadata
            Successful acoustic index result.

        Returns
        -------
        dict[str, Any]
            Base row.
        """
        return {
            "directory_name": result.directory_name,
            "file_name": result.file_name,
            "segment_id": result.segment_id,
            "sample_rate": "" if result.sample_rate is None else result.sample_rate,
            "processing_time": (
                ""
                if result.processing_time is None
                else round(result.processing_time, 2)
            ),
        }
