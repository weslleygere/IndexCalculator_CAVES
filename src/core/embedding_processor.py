import logging
from contextlib import nullcontext
from typing import Any, cast

import numpy as np
import torch
from transformers import AutoProcessor
from transformers import ClapModel

from .params_loader import ConfigParams
from .utils import AudioMetadata

logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """
    Compute pooled CLAP embeddings for one preprocessed audio container.

    The input audio is chunked into short windows, each chunk is embedded with
    the CLAP audio encoder, and chunk-level embeddings are pooled into one row.

    Parameters
    ----------
    config : ConfigParams
        Configuration object containing parameters for embedding calculation.    
    """

    def __init__(self, config: ConfigParams, hf_token: str | None = None) -> None:
        self.model_name = config.model_name
        self.chunk_duration_seconds = config.chunk_duration_seconds
        self.chunk_hop_seconds = config.chunk_hop_seconds
        self.pooling = config.pooling
        self.batch_size = config.batch_size
        self.device_mode = config.device
        self.use_fp16 = config.use_fp16
        
        self.hf_token = hf_token

        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            token=self.hf_token
        )
        self._model = cast(
            Any,
            ClapModel.from_pretrained(
                self.model_name,
                use_safetensors=True,
                token=self.hf_token,
            ),
        )

        cuda_available = torch.cuda.is_available()
        wants_cuda = self.device_mode in {"auto", "cuda"}
        self.device = "cuda" if wants_cuda and cuda_available else "cpu"
        if self.device_mode == "cuda" and self.device != "cuda":
            logger.warning("CUDA requested but unavailable. Falling back to CPU for embeddings.")

        self._use_amp = self.device == "cuda" and self.use_fp16

        self._model.to(self.device)
        self._model.eval()

    def process_file(self, audio: AudioMetadata) -> AudioMetadata:
        """
        Calculate pooled CLAP embeddings for one preprocessed audio container.
        
        Parameters
        ----------
        audio : AudioMetadata
            Preprocessed input container containing waveform and sample rate.

        Returns
        -------
        AudioMetadata
            Success container with embeddings values, or failed container with
            error metadata if calculation fails.
        """
        try:
            wave = cast(np.ndarray, audio.wave)
            sample_rate = cast(int, audio.sample_rate)

            # Step 1: Chunk the audio signal into short windows
            chunks = self._chunk_signal(wave, sample_rate)

            # Step 2: Encode each chunk with the CLAP audio encoder
            chunk_embeddings = self._encode_chunks(chunks, sample_rate)

            # Step 3: Pool chunk-level embeddings into one row
            pooled = self._pool_embeddings(chunk_embeddings)

            return AudioMetadata(
                stage="embedding",
                status="success",
                file_name=audio.file_name,
                segment_id=audio.segment_id,
                directory_name=audio.directory_name,
                sample_rate=audio.sample_rate,
                embeddings=pooled,
            )

        except Exception as exc:
            return AudioMetadata(
                stage="embedding",
                status="failed",
                file_name=audio.file_name,
                segment_id=audio.segment_id,
                directory_name=audio.directory_name,
                error=str(exc),
                error_type=type(exc).__name__,
                sample_rate=audio.sample_rate,
            )

    def _chunk_signal(self, wave: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        """
        Chunk the audio signal into short windows based on configured duration and hop.
        
        Parameters
        ----------
        wave : np.ndarray
            Segment waveform.
        sample_rate : int
            Sample rate of the audio signal.
        
        Returns
        -------
        list[np.ndarray]
            List of audio chunks as numpy arrays.
        """
        chunk_len = int(round(self.chunk_duration_seconds * sample_rate))
        hop_len = int(round(self.chunk_hop_seconds * sample_rate))

        if wave.size <= chunk_len:
            padded = np.zeros(chunk_len, dtype=np.float32)
            padded[: wave.size] = wave
            return [padded]

        chunks: list[np.ndarray] = []
        start = 0
        while start + chunk_len <= wave.size:
            chunks.append(wave[start : start + chunk_len])
            start += hop_len

        if start < wave.size:
            tail = wave[start:]
            if tail.size > 0:
                padded = np.zeros(chunk_len, dtype=np.float32)
                padded[: tail.size] = tail
                chunks.append(padded)

        if not chunks:
            raise ValueError("No valid audio chunks generated for embedding.")

        return chunks

    def _encode_chunks(self, chunks: list[np.ndarray], sample_rate: int) -> np.ndarray:
        """
        Encode each audio chunk with the CLAP audio encoder and return stacked embeddings.
        
        Parameters
        ----------
        chunks : list[np.ndarray]
            List of audio chunks as numpy arrays.
        sample_rate : int
            Sample rate of the audio signal.

        Returns
        -------
        np.ndarray
            Stacked embeddings for all audio chunks.
        """
        batches: list[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i : i + self.batch_size]
                inputs = self._processor(
                    audio=batch,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                amp_context = self._autocast_context()
                with amp_context:
                    embeddings = cast(Any, self._model.get_audio_features(**inputs))

                if hasattr(embeddings, "detach"):
                    emb_tensor = embeddings
                elif isinstance(embeddings, tuple) and embeddings:
                    emb_tensor = embeddings[0]
                else:
                    pooled = getattr(embeddings, "pooler_output", None)
                    if pooled is None:
                        raise ValueError("Unexpected embedding output type returned by model.")
                    emb_tensor = pooled

                emb_np = cast(Any, emb_tensor).detach().cpu().numpy()
                batches.append(np.asarray(emb_np, dtype=np.float32))

        return np.vstack(batches)

    def _pool_embeddings(self, embeddings: np.ndarray) -> dict[str, Any]:
        """
        Pool the embeddings into a single vector.

        Parameters
        ----------
        embeddings : np.ndarray
            Stacked embeddings for all audio chunks.

        Returns
        -------
        dict[str, Any]
            Pooled embedding values.
        """
        mean_vec = embeddings.mean(axis=0)
        out: dict[str, Any] = {f"e_mean_{i:04d}": float(v) for i, v in enumerate(mean_vec)}

        if self.pooling == "mean_std":
            std_vec = embeddings.std(axis=0)
            out.update({f"e_std_{i:04d}": float(v) for i, v in enumerate(std_vec)})

        return out

    def _autocast_context(self):
        """Return a context manager for automatic mixed precision if enabled and supported."""
        if self._use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()
    