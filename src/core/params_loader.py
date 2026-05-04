from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, init=False)
class ConfigParams:
    """
    Typed parameter set loaded from JSON, including audio logic and indices.
    
    Parameters
    ----------
    params_json_path : str | Path
        The filesystem location to the JSON definition.
    """
    # Audio preprocessing parameters
    channel: str
    detrend: bool
    segment_duration: int
    clipping_threshold: float
    target_sample_rate: int | None
    normalize_audio: bool
    segment_tolerance_percent: float

    # Spectrogram parameters
    nperseg: int
    noverlap: int

    # Spectral indices parameters
    flim_low: tuple[int, int]
    flim_mid: tuple[int, int]
    flim_hi: tuple[int, int]
    dB_threshold: float
    fmin: int
    fmax: int
    bin_step: int
    flim_bio: tuple[int, int]
    flim_anthro: tuple[int, int]
    flim_bioacoustics: tuple[int, int]
    R_compatible: str

    # Temporal indices parameters
    mode: str
    Nt: int
    compatibility: str

    # Embedding extraction parameters
    model_name: str
    chunk_duration_seconds: float
    chunk_hop_seconds: float
    pooling: str
    batch_size: int
    device: str
    use_fp16: bool

    def __init__(self, params_json_path: str | Path) -> None:

        with open(params_json_path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)

        try:
            preprocess = payload["preprocess"]
            spectrogram = payload["spectrogram"]
            spectral = payload["spectral_indices"]
            temporal = payload["temporal_indices"]
            embedding = payload["embeddings"]
            
            # Audio preprocessing parameters
            self.channel = preprocess["channel"]
            self.detrend = preprocess["detrend"]
            self.segment_duration = preprocess["segment_duration"]
            self.clipping_threshold = preprocess["clipping_threshold"]
            self.target_sample_rate = preprocess["target_sample_rate"]
            self.normalize_audio = preprocess["normalize_audio"]
            self.segment_tolerance_percent = preprocess["segment_tolerance_percent"]

            # Spectrogram parameters
            self.nperseg = spectrogram["nperseg"]
            self.noverlap = spectrogram["noverlap"]

            # Spectral indices parameters
            self.flim_low = tuple(spectral["flim_low"])
            self.flim_mid = tuple(spectral["flim_mid"])
            self.flim_hi = tuple(spectral["flim_hi"])
            self.dB_threshold = spectral["dB_threshold"]
            self.fmin = spectral["fmin"]
            self.fmax = spectral["fmax"]
            self.bin_step = spectral["bin_step"]
            self.flim_bio = tuple(spectral["flim_bio"])
            self.flim_anthro = tuple(spectral["flim_anthro"])
            self.flim_bioacoustics = tuple(spectral["flim_bioacoustics"])
            self.R_compatible = spectral["R_compatible"]

            # Temporal indices parameters
            self.mode = temporal["mode"]
            self.Nt = temporal["Nt"]
            self.compatibility = temporal["compatibility"]
            
            # Embedding extraction parameters
            self.model_name = embedding["model_name"]
            self.chunk_duration_seconds = embedding["chunk_duration_seconds"]
            self.chunk_hop_seconds = embedding["chunk_hop_seconds"]
            self.pooling = embedding["pooling"]
            self.batch_size = embedding["batch_size"]
            self.device = embedding["device"]
            self.use_fp16 = embedding["use_fp16"]
            
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid parameter JSON structure: {exc}") from exc

        self._validate_preprocess_params()
        self._validate_embedding_params()

    def _validate_preprocess_params(self) -> None:
        """Validate audio preprocessing parameters."""
        if isinstance(self.segment_duration, bool) or not isinstance(self.segment_duration, int) or self.segment_duration <= 0:
            raise ValueError("preprocess.segment_duration must be an integer > 0")

        if isinstance(self.clipping_threshold, bool) or not isinstance(self.clipping_threshold, (int, float)) or self.clipping_threshold < 0.0 or self.clipping_threshold > 1.0:
            raise ValueError("preprocess.clipping_threshold must be between 0 and 1")

        if self.target_sample_rate is not None:
            if isinstance(self.target_sample_rate, bool) or not isinstance(self.target_sample_rate, int) or self.target_sample_rate <= 0:
                raise ValueError("preprocess.target_sample_rate must be null or an integer > 0. Common values are 22050, 44100, or 48000")

        if not isinstance(self.normalize_audio, bool):
            raise ValueError("preprocess.normalize_audio must be boolean")

        if isinstance(self.segment_tolerance_percent, bool) or not isinstance(self.segment_tolerance_percent, (int, float)) or self.segment_tolerance_percent < 0.0 or self.segment_tolerance_percent > 100.0:
            raise ValueError("preprocess.segment_tolerance_percent must be between 0 and 100")

    def _validate_embedding_params(self) -> None:
        """Validate embedding extraction parameters."""
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("embeddings.model_name must be a non-empty string")

        if not isinstance(self.chunk_duration_seconds, (int, float)) or self.chunk_duration_seconds <= 0:
            raise ValueError("embeddings.chunk_duration_seconds must be > 0")

        if not isinstance(self.chunk_hop_seconds, (int, float)) or self.chunk_hop_seconds <= 0:
            raise ValueError("embeddings.chunk_hop_seconds must be > 0")

        self.pooling = str(self.pooling).strip().lower()
        if self.pooling not in {"mean", "mean_std"}:
            raise ValueError("embeddings.pooling must be one of: mean, mean_std")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("embeddings.batch_size must be an integer > 0")

        self.device = str(self.device).strip().lower()
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("embeddings.device must be one of: auto, cpu, cuda")

        if not isinstance(self.use_fp16, bool):
            raise ValueError("embeddings.use_fp16 must be boolean")
