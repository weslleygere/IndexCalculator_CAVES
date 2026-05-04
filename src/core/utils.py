import os
import csv
import logging
from dataclasses import dataclass
from typing import Any, Iterator
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class AudioMetadata:
    """
    Standardized result for audio processing stages (loading, preprocessing).
    
    Parameters
    ----------
    stage : str
        The processing stage (e.g., 'load', 'preprocess').
    status : str
        The status of the processing (e.g., 'success', 'failed').
    file_name : str
        The name of the audio file.
    segment_id : str | None
        Stable segment-level identifier used for cross-run joins.
    directory_name : str | None
        Parent directory name used to disambiguate identical file names.
    wave : np.ndarray | None
        The audio waveform data.
    acoustic_idx : dict[str, Any] | None
        A dictionary containing acoustic indices for the audio file.  
    embeddings : dict[str, Any] | None
        A dictionary containing pooled embedding values for the audio file.
    sample_rate : int | None
        The sample rate of the audio data.
    error : str | None
        Error message if the processing failed.
    error_type : str | None
        Type of the error if the processing failed.
    """
    stage: str
    status: str
    file_name: str
    segment_id: str | None = None
    directory_name: str | None = None
    wave: np.ndarray | None = None
    acoustic_idx: dict[str, Any] | None = None
    embeddings: dict[str, Any] | None = None
    sample_rate: int | None = None
    processing_time: float | None = None
    error: str | None = None
    error_type: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "success"


class FileScanner:
    """
    Scans directories for audio files and manages path resolution for checkpoints.
    
    Parameters
    ----------
    data_dir : Path | str
        Directory containing WAV files. Search is recursive.
    """
    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir).resolve()

    def iter_audio_paths_by_directory(self) -> Iterator[tuple[str, list[Path]]]:
        """
        Yield WAV file paths grouped by directory (deterministic order).

        Yields
        ------
        tuple[str, list[Path]]
            (directory_key, list_of_wav_paths)
        """
        for root, dirs, files in os.walk(self.data_dir):
            dirs.sort()
            wav_files = sorted(file for file in files if file.lower().endswith(".wav"))
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
    Manages the parsing, loading, and appending of processed file states.
    
    Parameters
    ----------
    output_dir : Path | str
        Directory to hold the checkpoint txt file.
    data_dir : Path | str
        Base directory containing input WAV files, used to compute relative paths.
    """
    def __init__(self, output_dir: Path | str, data_dir: Path | str, checkpoint_file_name: str = "processed_files_checkpoint.txt"):
        self.output_dir = Path(output_dir).resolve()
        self.checkpoint_path = self.output_dir / checkpoint_file_name
        self.data_dir_abs = Path(data_dir).resolve()
        self.records: dict[str, tuple[int, int, int]] = {}

    def check_resume_mode(self) -> bool:
        """
        Loads the checkpoint, populates internal records, and states if resuming.
        
        Returns
        -------
        bool
            True if previous successful records were loaded, indicating resume mode.
            False if checkpoint doesn't exist or is empty.
        """
        if not self.checkpoint_path.exists():
            return False
            
        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) == 4:
                    # New format: path|seg_success|seg_failed|file_failed
                    self.records[parts[0]] = (int(parts[1]), int(parts[2]), int(parts[3]))
                elif len(parts) == 3:
                    # Previous format: path|seg_success|seg_failed
                    self.records[parts[0]] = (int(parts[1]), int(parts[2]), 0)
                elif len(parts) == 2:
                    # Legacy format: path|success or path|failed
                    is_success = (parts[1] == "success")
                    self.records[parts[0]] = (1 if is_success else 0, 0, 0 if is_success else 1)
                else:
                    # Legacy fallback
                    self.records[line] = (1, 0, 0)
        return len(self.records) > 0

    def check_pending(self, paths: list[Path]) -> tuple[list[Path], int, int, int]:
        """
        Filters out already processed paths based on loaded records.
        
        Parameters
        ----------
        paths : list[Path]
            A list of file paths to check against loaded checkpoints.

        Returns
        -------
        tuple[list[Path], int, int, int]
            A tuple containing:
            - A list of paths left to be processed.
            - Total number of prior segments successes.
            - Total number of prior segments failures.
            - Total number of prior file-level failures.
        """
        pending = []
        seg_success = 0
        seg_failed = 0
        file_failed = 0
        for p in paths:
            key = self._get_key(p)
            if key in self.records:
                s, f, ff = self.records[key]
                seg_success += s
                seg_failed += f
                file_failed += ff
            else:
                pending.append(p)
        return pending, seg_success, seg_failed, file_failed

    def mark_completed(self, file_path: Path, results: list[AudioMetadata]) -> tuple[int, int, int]:
        """
        Evaluates processing final state from segments, appending to checkpoint.
        
        Parameters
        ----------
        file_path : Path
            The absolute or relative path to the original audio file.
        results : list[AudioMetadata]
            The list of calculated results for segments derived from this file.
            
        Returns
        -------
        tuple[int, int, int]
            The number of successful segments, failed segments, and failed files (if failure happened before segmentation).
        """
        seg_success_count = 0
        seg_failed_count = 0
        file_failed_count = 0
        
        # If the failure happened in the load step, we have exactly 1 result representing the whole file
        if len(results) == 1 and not results[0].ok and results[0].stage == "load":
            file_failed_count = 1
        else:
            seg_success_count = sum(1 for c in results if c.ok)
            seg_failed_count = len(results) - seg_success_count
            
        key = self._get_key(file_path)
        
        with open(self.checkpoint_path, "a", encoding="utf-8") as f:
            f.write(f"{key}|{seg_success_count}|{seg_failed_count}|{file_failed_count}\n")
            
        self.records[key] = (seg_success_count, seg_failed_count, file_failed_count)
        return seg_success_count, seg_failed_count, file_failed_count

    def _get_key(self, file_path: Path | str) -> str:
        """
        Computes a relative key for the checkpoint record based on the file path.
        
        Parameters
        ----------
        file_path : Path | str
            The absolute or relative path to the original audio file.
            
        Returns
        -------
        str
             A relative path key for checkpointing, or just the file name if relative path cannot be computed.
        """
        abs_file_path = Path(file_path).resolve()
        try:
            return abs_file_path.relative_to(self.data_dir_abs).as_posix()
        except ValueError:
            return abs_file_path.name


class CsvWriter:
    """
    Stream processing outputs to CSV files and maintain execution summary.

    Two CSV files are produced:
    - acoustic_[mode].csv: successful acoustic index or embedding rows
    - acoustic_errors.csv: failures from any stage
    
    Parameters
    ----------
    output_file : str | Path
        Path to the primary CSV output (acoustic indices).
    error_file : str | Path
        Path to the errors CSV output.
    resume : bool
        If True, appends to existing CSVs and infers headers from them.
    """

    def __init__(
        self,
        output_file: str | Path,
        error_file: str | Path,
        resume: bool = False,
        mode: str = "indices",
    ) -> None:
        self.output_file = Path(output_file)
        self.error_file = Path(error_file)
        self.resume = resume
        self.mode = mode

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.error_file.parent.mkdir(parents=True, exist_ok=True)

        self._data_stream = None
        self._data_writer: csv.DictWriter[str] | None = None
        self._data_fieldnames: list[str] = []
        self._warned_extra_keys = False
        
        self._initialize_data_writer_for_resume_if_needed()

        # Always append to error_file if it exists to preserve errors from prior passes (e.g. indices then embeddings)
        error_exists = self.error_file.exists() and self.error_file.stat().st_size > 0
        self._errors_stream = self.error_file.open("a" if error_exists else "w", newline="", encoding="utf-8")
        self._errors_writer = csv.DictWriter(
            self._errors_stream,
            fieldnames=[
                "stage",
                "directory_name",
                "file_name",
                "error",
                "error_type",
            ],
        )
        if not error_exists:
            self._errors_writer.writeheader()

    def __enter__(self) -> "CsvWriter":
        """Support for context manager protocol to ensure proper resource management."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Ensure that all open file streams are properly closed when exiting the context."""
        self.close()

    def consume(self, file_results: list[AudioMetadata]) -> None:
        """
        Consume and persist all stage outputs produced for one input file.
        
        Parameters
        ----------
        file_results : list[AudioMetadata]
            The chunk of calculated metadata results to inject into either the
            success log (indices CSV) or the error log (errors CSV).
        """
        
        for result in file_results:
            if result.ok:
                self._write_success_row(result)
            else:
                self._write_error_row(result)

    def close(self) -> None:
        """Flush and close open CSV files."""
        if self._data_writer is None:
            self._data_fieldnames = ["directory_name", "file_name", "sample_rate", "processing_time"]
            self._data_stream = self.output_file.open("w", newline="", encoding="utf-8")
            self._data_writer = csv.DictWriter(self._data_stream, fieldnames=self._data_fieldnames)
            self._data_writer.writeheader()

        if self._data_stream is not None:
            self._data_stream.close()

        self._errors_stream.close()

    def _initialize_data_writer_for_resume_if_needed(self) -> None:
        """Open data CSV in append mode if a resumable file already exists."""
        data_exists = self.resume and self.output_file.exists() and self.output_file.stat().st_size > 0
        if not data_exists:
            return

        with self.output_file.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)

        if not header:
            return

        self._data_fieldnames = header
        self._data_stream = self.output_file.open("a", newline="", encoding="utf-8")
        self._data_writer = csv.DictWriter(self._data_stream, fieldnames=self._data_fieldnames)

    def _write_success_row(self, result: AudioMetadata) -> None:
        """
        Write one successful data row to the CSV.
        
        Parameters
        ----------
        result : AudioMetadata
            The successful processing result containing data to log.
        """
        data = result.acoustic_idx if self.mode == "indices" else result.embeddings
        if data:
            self._write_data_row(result, data)

    def _write_data_row(self, result: AudioMetadata, data: dict[str, Any]) -> None:
        """Write one data row to the CSV."""

        if self._data_writer is None:
            self._data_fieldnames = ["directory_name", "file_name", "sample_rate", "processing_time"] + sorted(data.keys())
            self._data_stream = self.output_file.open("w", newline="", encoding="utf-8")
            self._data_writer = csv.DictWriter(self._data_stream, fieldnames=self._data_fieldnames)
            self._data_writer.writeheader()

        row: dict[str, Any] = {
            "directory_name": result.directory_name,
            "file_name": result.file_name,
            "sample_rate": "" if result.sample_rate is None else result.sample_rate,
            "processing_time": "" if result.processing_time is None else round(result.processing_time, 2),
        }

        extra_keys = []
        for key, value in data.items():
            if key in self._data_fieldnames:
                row[key] = value
            else:
                extra_keys.append(key)

        if extra_keys and not self._warned_extra_keys:
            logger.warning(
                "Additional data keys were ignored because CSV header is fixed after first row: %s",
                sorted(extra_keys),
            )
            self._warned_extra_keys = True

        self._data_writer.writerow(row)

    def _write_error_row(self, result: AudioMetadata) -> None:
        """
        Write one failed stage output to the errors CSV.
        
        Parameters
        ----------
        result : AudioMetadata
            The failed processing result containing error information to log.
        """
        self._errors_writer.writerow(
            {
                "stage": result.stage,
                "directory_name": result.directory_name,
                "file_name": result.file_name,
                "error": result.error,
                "error_type": result.error_type,
            }
        )
