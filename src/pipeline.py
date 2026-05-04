from __future__ import annotations

import logging
import time
import multiprocessing
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm

from .config.settings import Settings
from .core.audio_loader import AudioLoader
from .core.params_loader import ConfigParams
from .core.utils import AudioMetadata, FileScanner, CheckpointManager, CsvWriter
from .core.idx_processor import AcousticIndexProcessor
from .core.embedding_processor import EmbeddingProcessor
from .core.pre_processor import PreProcessor

logger = logging.getLogger(__name__)


def _init_worker(config: ConfigParams, mode: str, hf_token: str | None = None) -> None:
    """
    Configures each worker in the pool to avoid reloading parameters for every file.
    
    Parameters
    ----------
    config : ConfigParams
        Global audio and indices configuration mapped uniformly to every process.
    """
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    
    global _WORKER_MODE, _WORKER_LOADER, _WORKER_PREPROCESSOR, _WORKER_INDEX_PROCESSOR, _WORKER_EMBEDDING_PROCESSOR

    _WORKER_MODE = mode
    _WORKER_LOADER = AudioLoader(config=config)
    _WORKER_PREPROCESSOR = PreProcessor(config=config)
    
    if mode == "indices":
        _WORKER_INDEX_PROCESSOR = AcousticIndexProcessor(config=config)
    elif mode == "embeddings":
        _WORKER_EMBEDDING_PROCESSOR = EmbeddingProcessor(config=config, hf_token=hf_token)


def _process_one_file(file_path: Path) -> list[AudioMetadata]:
    """
    Processes 1 entire file: loads, pre-processes, and extracts indices.
    
    Parameters
    ----------
    file_path : Path
        Absolute filepath to the source WAV.

    Returns
    -------
    list[AudioMetadata]
        A mapping series holding containers referencing segments computed inside
        or capturing standard failure exceptions per segment layout.
    """
    mode = _WORKER_MODE
    
    start_time = time.monotonic()

    # 1. Load the audio file
    loaded = _WORKER_LOADER.load_audio(file_path)
    if not loaded.ok:
        return [loaded]

    # 2. Pre-process the audio 
    preprocessed = _WORKER_PREPROCESSOR.process_file(loaded)
    
    final_results: list[AudioMetadata] = []
    for segment in preprocessed:
        if not segment.ok:
            final_results.append(segment)
            continue
        
        # 3. Process indices or embeddings based on single pass mode
        if mode == "indices":
            final_results.append(_WORKER_INDEX_PROCESSOR.process_file(segment))
        elif mode == "embeddings":
            final_results.append(_WORKER_EMBEDDING_PROCESSOR.process_file(segment))

    processing_time = time.monotonic() - start_time
    for result in final_results:
        if result.ok:
            result.processing_time = processing_time

    return final_results


class Pipeline:
    """
    Orchestrates directories, delegating paths to worker pools and writing in real-time.
    
    Parameters
    ----------
    settings : Settings
        The unified and robust typed settings combining runtime config and env context.
    """

    def __init__(self, settings: Settings) -> None:
        self.output_dir = settings.data.create_output_dir()
        self.data_path = settings.data.data_path
        self.workers = settings.processing.max_workers
        self.mode = settings.processing.mode
        self.hf_token = settings.auth.hf_token
        self.config_params = ConfigParams(settings.data.config_params_path)
    
    def run(self) -> None:
        """
        Executes the main pipeline workload.
        Creates sequential scans skipping previously logged states (checkpoints)
        delegating audio computing into pool distributions uniformly persisting real-time updates to CSV.
        """
        if self.mode == "both":
            self._run_single_mode("indices")
            self._run_single_mode("embeddings")
            return

        self._run_single_mode(self.mode)
    
    def _run_single_mode(self, mode: str) -> None:
        """Run one full pass for a specific mode."""
        scanner = FileScanner(self.data_path)
        checkpoint_mgr = CheckpointManager(
            self.output_dir,
            self.data_path,
            checkpoint_file_name=f"processed_files_checkpoint_{mode}.txt",
        )

        workers = self._resolve_workers(mode)
        resume_mode = checkpoint_mgr.check_resume_mode()

        logger.info(
            "Pass: %s | Workers: %s | Resume: %s",
            mode,
            workers,
            'Enabled' if resume_mode else 'Disabled'
        )

        with (
            CsvWriter(
                output_file=self.output_dir / f"acoustic_{mode}.csv",
                error_file=self.output_dir / "acoustic_errors.csv",
                resume=resume_mode,
                mode=mode,
            ) as writer,
            (
                ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_worker,
                    initargs=(self.config_params, mode, self.hf_token),
                )
                if workers > 1
                else nullcontext()
            ) as executor
        ):
            if workers <= 1:
                _init_worker(self.config_params, mode, self.hf_token)

            for dir_key, all_paths in scanner.iter_audio_paths_by_directory():
                pending_paths, success_count, failed_count, file_failed_count = checkpoint_mgr.check_pending(all_paths)

                if not pending_paths:
                    continue

                logger.info(
                    "Processing dir [%s] - %s pending out of %s total files.",
                    dir_key,
                    len(pending_paths),
                    len(all_paths),
                )

                executor_map = executor.map if executor else map
                results_iterator = executor_map(_process_one_file, pending_paths)

                progress_bar = tqdm(
                    zip(pending_paths, results_iterator),
                    total=len(pending_paths),
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                    ncols=120
                )

                for path_source, file_results in progress_bar:
                    writer.consume(file_results)

                    seg_success, seg_failed, file_failed = checkpoint_mgr.mark_completed(path_source, file_results)
                    success_count += seg_success
                    failed_count += seg_failed
                    file_failed_count += file_failed

                logger.info(
                    "Pass completed dir [%s] | Total Files: %s | Success (segments): %s | Failures (segments): %s | Failures (files): %s.",
                    dir_key,
                    len(all_paths),
                    success_count,
                    failed_count,
                    file_failed_count,
                )

    def _resolve_workers(self, mode: str) -> int | None:
        """Resolve workers by pass mode, forcing single worker for GPU pass."""
        if mode == "embeddings" and self.workers and self.workers > 1:
            return 1
        return self.workers
