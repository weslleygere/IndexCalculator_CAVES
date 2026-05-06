from __future__ import annotations

import logging
import multiprocessing
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from .config.settings import Settings
from .core.audio_loader import AudioLoader
from .core.idx_processor import AcousticIndexProcessor
from .core.params_loader import ConfigParams
from .core.pre_processor import PreProcessor
from .core.utils import AudioMetadata, CheckpointManager, CsvWriter, FileScanner


logger = logging.getLogger(__name__)


_WORKER_LOADER: AudioLoader
_WORKER_PREPROCESSOR: PreProcessor
_WORKER_INDEX_PROCESSOR: AcousticIndexProcessor


def _init_worker(config: ConfigParams) -> None:
    """
    Initialize worker-local processors.

    Parameters
    ----------
    config : ConfigParams
        Shared configuration object passed to each worker process.
    """
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=RuntimeWarning)

    global _WORKER_LOADER, _WORKER_PREPROCESSOR, _WORKER_INDEX_PROCESSOR

    _WORKER_LOADER = AudioLoader(config=config)
    _WORKER_PREPROCESSOR = PreProcessor(config=config)
    _WORKER_INDEX_PROCESSOR = AcousticIndexProcessor(config=config)


def _process_one_file(file_path: Path) -> list[AudioMetadata]:
    """
    Process one WAV file from loading to acoustic index extraction.

    Parameters
    ----------
    file_path : Path
        Source WAV path.

    Returns
    -------
    list[AudioMetadata]
        Final segment-level results, or one file-level failure.
    """
    start_time = time.monotonic()

    loaded = _WORKER_LOADER.load_audio(file_path)
    if not loaded.ok:
        loaded.processing_time = time.monotonic() - start_time
        return [loaded]

    segments = _WORKER_PREPROCESSOR.process_file(loaded)

    results: list[AudioMetadata] = []

    for segment in segments:
        if segment.ok:
            results.append(_WORKER_INDEX_PROCESSOR.process_file(segment))
        else:
            results.append(segment)

    processing_time = time.monotonic() - start_time

    for result in results:
        result.processing_time = processing_time

    return results


class Pipeline:
    """
    Orchestrate audio scanning, multiprocessing, CSV writing, and checkpointing.

    Parameters
    ----------
    settings : Settings
        Runtime settings loaded from the environment.
    """

    def __init__(self, settings: Settings) -> None:
        self.output_dir = settings.data.create_output_dir()
        self.data_path = settings.data.data_path
        self.workers = settings.processing.max_workers
        self.config_params = ConfigParams(settings.data.config_params_path)

    def execute(self) -> None:
        """
        Execute the acoustic index calculation pipeline.
        """
        scanner = FileScanner(self.data_path)
        checkpoint_mgr = CheckpointManager(
            output_dir=self.output_dir,
            data_dir=self.data_path,
            checkpoint_file_name="processed_files_checkpoint.txt",
        )

        resume_mode = checkpoint_mgr.check_resume_mode()

        logger.info(
            "Pass: acoustic_indices | Workers: %s | Resume: %s",
            self.workers,
            "Enabled" if resume_mode else "Disabled",
        )

        with CsvWriter(
            output_file=self.output_dir / "acoustic_indices.csv",
            error_file=self.output_dir / "acoustic_errors.csv",
            resume=resume_mode,
        ) as writer:
            with ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
                initargs=(self.config_params,),
            ) as executor:
                self._process_directories(
                    scanner=scanner,
                    checkpoint_mgr=checkpoint_mgr,
                    writer=writer,
                    executor=executor,
                )

    def _process_directories(
        self,
        scanner: FileScanner,
        checkpoint_mgr: CheckpointManager,
        writer: CsvWriter,
        executor: ProcessPoolExecutor,
    ) -> None:
        """
        Process all discovered directories.

        Parameters
        ----------
        scanner : FileScanner
            Audio file scanner.
        checkpoint_mgr : CheckpointManager
            Checkpoint manager.
        writer : CsvWriter
            CSV writer.
        executor : ProcessPoolExecutor
            Worker pool.
        """
        for dir_key, all_paths in scanner.iter_audio_paths_by_directory():
            (
                pending_paths,
                success_count,
                failed_count,
                file_failed_count,
            ) = checkpoint_mgr.check_pending(all_paths)

            if not pending_paths:
                logger.info(
                    "Skipping dir [%s] - all %s files already processed.",
                    dir_key,
                    len(all_paths),
                )
                continue

            logger.info(
                "Processing dir [%s] - %s pending out of %s total files.",
                dir_key,
                len(pending_paths),
                len(all_paths),
            )

            counts = self._process_pending_paths(
                pending_paths=pending_paths,
                writer=writer,
                checkpoint_mgr=checkpoint_mgr,
                executor=executor,
                initial_counts=(success_count, failed_count, file_failed_count),
            )

            logger.info(
                "Completed dir [%s] | Total files: %s | Success segments: %s | Failed segments: %s | Failed files: %s.",
                dir_key,
                len(all_paths),
                counts[0],
                counts[1],
                counts[2],
            )

    def _process_pending_paths(
        self,
        pending_paths: list[Path],
        writer: CsvWriter,
        checkpoint_mgr: CheckpointManager,
        executor: ProcessPoolExecutor,
        initial_counts: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """
        Process all pending files from one directory.

        Parameters
        ----------
        pending_paths : list[Path]
            Files not present in the checkpoint.
        writer : CsvWriter
            CSV writer.
        checkpoint_mgr : CheckpointManager
            Checkpoint manager.
        executor : ProcessPoolExecutor
            Worker pool.
        initial_counts : tuple[int, int, int]
            Previously counted successful segments, failed segments, and
            file-level failures.

        Returns
        -------
        tuple[int, int, int]
            Updated successful segment count, failed segment count, and
            file-level failure count.
        """
        success_count, failed_count, file_failed_count = initial_counts

        results_iterator = executor.map(_process_one_file, pending_paths)

        progress_bar = tqdm(
            zip(pending_paths, results_iterator),
            total=len(pending_paths),
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
            ncols=120,
        )

        for source_path, file_results in progress_bar:
            writer.consume(file_results)
            writer.flush()

            seg_success, seg_failed, file_failed = checkpoint_mgr.mark_completed(
                source_path,
                file_results,
            )

            success_count += seg_success
            failed_count += seg_failed
            file_failed_count += file_failed

        return success_count, failed_count, file_failed_count
