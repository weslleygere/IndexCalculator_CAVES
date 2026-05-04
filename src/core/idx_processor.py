import logging
from typing import Any, cast

import numpy as np
from maad import sound, features

from .params_loader import ConfigParams
from .utils import AudioMetadata


logger = logging.getLogger(__name__)


class AcousticIndexProcessor:
    """
    Compute acoustic indices for one preprocessed audio container.

    The processor computes temporal and spectral alpha indices and returns them
    in a standardized AudioMetadata payload under acoustic_idx.

    Parameters
    ----------
    config : ConfigParams
        Configuration object containing parameters for index calculation.
    """

    def __init__(self, config: ConfigParams) -> None:
        # Spectrogram parameters
        self.nperseg = config.nperseg
        self.noverlap = config.noverlap

        # Spectral indices parameters
        self.flim_low = config.flim_low
        self.flim_mid = config.flim_mid
        self.flim_hi = config.flim_hi
        self.db_threshold = config.dB_threshold
        self.fmin = config.fmin
        self.fmax = config.fmax
        self.bin_step = config.bin_step
        self.flim_bio = config.flim_bio
        self.flim_anthro = config.flim_anthro
        self.flim_bioacoustics = config.flim_bioacoustics
        self.r_compatible = config.R_compatible

        # Temporal indices parameters
        self.mode = config.mode
        self.nt = config.Nt
        self.compatibility = config.compatibility


    def process_file(self, audio: AudioMetadata) -> AudioMetadata:
        """
        Calculate acoustic indices for one preprocessed audio container.

        Parameters
        ----------
        audio : AudioMetadata
            Preprocessed input container containing waveform and sample rate.

        Returns
        -------
        AudioMetadata
            Success container with acoustic_idx values, or failed container with
            error metadata if calculation fails.
        """
        try:
            wave = cast(np.ndarray, audio.wave)
            sample_rate = cast(int, audio.sample_rate)

            values: dict[str, Any] = {}

            # Step 1: Compute temporal indices
            values.update(self._compute_temporal_indices(wave, sample_rate))
            
            # Step 2: Compute spectral indices
            values.update(self._compute_spectral_indices(wave, sample_rate))

            return AudioMetadata(
                stage="acoustic_idx",
                status="success",
                file_name=audio.file_name,
                segment_id=audio.segment_id,
                directory_name=audio.directory_name,
                acoustic_idx=values,
                sample_rate=sample_rate,
            )

        except Exception as exc:
            return AudioMetadata(
                stage="acoustic_idx",
                status="failed",
                file_name=audio.file_name,
                segment_id=audio.segment_id,
                directory_name=audio.directory_name,
                error=str(exc),
                error_type=type(exc).__name__,
                sample_rate=audio.sample_rate,
            )

    def _compute_temporal_indices(self, wave: np.ndarray, sample_rate: int) -> dict[str, Any]:
        """
        Compute temporal alpha indices and return prefixed scalar values.

        Parameters
        ----------
        wave : np.ndarray
            Segment waveform.
        sample_rate : int
            Waveform sampling rate.

        Returns
        -------
        dict[str, Any]
            Temporal index values prefixed with "t_".
        """
        temporal_df = features.all_temporal_alpha_indices(
            wave,
            sample_rate,
            mode=self.mode,
            Nt=self.nt,
            compatibility=self.compatibility,
            verbose=False,
            display=False,
        )
        return self._frame_to_prefixed_dict(temporal_df, prefix="t_")

    def _compute_spectral_indices(self, wave: np.ndarray, sample_rate: int) -> dict[str, Any]:
        """
        Compute spectral alpha indices and return prefixed scalar values.

        Parameters
        ----------
        wave : np.ndarray
            Segment waveform.
        sample_rate : int
            Waveform sampling rate.

        Returns
        -------
        dict[str, Any]
            Spectral index values prefixed with "s_".
        """
        sxx_power, tn, fn, _ = sound.spectrogram(
            wave,
            sample_rate,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            mode="psd",
            verbose=False,
            display=False,
        )

        spectral_df, _ = features.all_spectral_alpha_indices(
            sxx_power,
            tn,
            fn,
            flim_low=self.flim_low,
            flim_mid=self.flim_mid,
            flim_hi=self.flim_hi,
            dB_threshold=self.db_threshold,
            fmin=self.fmin,
            fmax=self.fmax,
            bin_step=self.bin_step,
            flim_bioPh=self.flim_bio,
            flim_antroPh=self.flim_anthro,
            flim=self.flim_bioacoustics,
            R_compatible=self.r_compatible,
            verbose=False,
            display=False,
        )

        return self._frame_to_prefixed_dict(spectral_df, prefix="s_")

    @staticmethod
    def _frame_to_prefixed_dict(frame: Any, prefix: str) -> dict[str, Any]:
        """
        Convert the first row of a result DataFrame into a prefixed dictionary.

        Parameters
        ----------
        frame : Any
            DataFrame-like object returned by maad index functions.
        prefix : str
            Prefix added to each output key.

        Returns
        -------
        dict[str, Any]
            Normalized scalar values with prefixed keys.
        """
        if frame is None or frame.empty:
            return {}

        first_row = frame.iloc[0].to_dict()
        return {
            f"{prefix}{key}": AcousticIndexProcessor._normalize_value(value)
            for key, value in first_row.items()
        }

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """
        Normalize a value for CSV output, converting numpy scalars and handling NaN/inf.

        Parameters
        ----------
        value : Any
            Value to normalize.

        Returns
        -------
        Any
            Normalized value.
            - NumPy scalar types are converted to native Python scalars.
            - NaN and infinite float values are converted to None.
        """
        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return None
            return value

        return value
