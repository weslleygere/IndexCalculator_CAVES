from typing import Any

import numpy as np
from maad import features, sound

from .params_loader import ConfigParams
from .utils import AudioMetadata


class AcousticIndexProcessor:
    """
    Compute acoustic indices for one preprocessed audio segment.

    The processor mutates the received AudioMetadata object in place by adding
    acoustic_idx values and clearing the waveform after computation. If any
    index is invalid, the whole segment is marked as failed.

    Parameters
    ----------
    config : ConfigParams
        Configuration object containing parameters for index calculation.
    """

    def __init__(self, config: ConfigParams) -> None:
        self.flims = config.flims
        self.compatibility = config.compatibility

    def process_file(self, audio: AudioMetadata) -> AudioMetadata:
        """
        Calculate acoustic indices for one preprocessed audio segment.

        Parameters
        ----------
        audio : AudioMetadata
            Preprocessed segment containing waveform and sample rate.

        Returns
        -------
        AudioMetadata
            The same input object, updated in place as either success or failed.
        """
        try:
            wave = audio.require_wave("Missing waveform for acoustic index calculation")
            sample_rate = audio.require_sample_rate(
                "Missing sample rate for acoustic index calculation"
            )

            acoustic_idx = self._compute_indices(wave, sample_rate)
            self._validate_indices(acoustic_idx)

            audio.stage = "acoustic_idx"
            audio.status = "success"
            audio.acoustic_idx = acoustic_idx
            audio.sample_rate = sample_rate
            audio.error = None
            audio.error_type = None

        except Exception as exc:
            audio.stage = "acoustic_idx"
            audio.status = "failed"
            audio.acoustic_idx = None
            audio.error = str(exc)
            audio.error_type = type(exc).__name__

        finally:
            audio.wave = None

        return audio

    def _compute_indices(self, wave: np.ndarray, sample_rate: int) -> dict[str, Any]:
        """
        Compute temporal and spectral acoustic indices.

        Parameters
        ----------
        wave : np.ndarray
            Segment waveform.
        sample_rate : int
            Segment sample rate.

        Returns
        -------
        dict[str, Any]
            Dictionary with temporal and spectral acoustic indices.
        """
        acoustic_idx: dict[str, Any] = {}

        acoustic_idx.update(self._compute_temporal_indices(wave, sample_rate))
        acoustic_idx.update(self._compute_spectral_indices(wave, sample_rate))

        return acoustic_idx

    def _compute_temporal_indices(
        self,
        wave: np.ndarray,
        sample_rate: int,
    ) -> dict[str, Any]:
        """
        Compute temporal alpha indices.

        Parameters
        ----------
        wave : np.ndarray
            Segment waveform.
        sample_rate : int
            Segment sample rate.

        Returns
        -------
        dict[str, Any]
            Temporal index values prefixed with "t_".
        """
        temporal_df = features.all_temporal_alpha_indices(
            wave,
            sample_rate,
            compatibility=self.compatibility,
        )

        return self._frame_to_prefixed_dict(temporal_df, prefix="t_")

    def _compute_spectral_indices(
        self,
        wave: np.ndarray,
        sample_rate: int,
    ) -> dict[str, Any]:
        """
        Compute spectral alpha indices.

        Parameters
        ----------
        wave : np.ndarray
            Segment waveform.
        sample_rate : int
            Segment sample rate.

        Returns
        -------
        dict[str, Any]
            Spectral index values prefixed with "s_".
        """
        sxx_power, tn, fn, _ = sound.spectrogram(
            wave,
            sample_rate,
            flims=self.flims,
        )

        spectral_df, _ = features.all_spectral_alpha_indices(
            sxx_power,
            tn,
            fn,
        )

        return self._frame_to_prefixed_dict(spectral_df, prefix="s_")

    @staticmethod
    def _frame_to_prefixed_dict(frame: Any, prefix: str) -> dict[str, Any]:
        """
        Convert the first row of a DataFrame-like object into a dictionary.

        Parameters
        ----------
        frame : Any
            DataFrame-like object returned by maad.
        prefix : str
            Prefix added to each output key.

        Returns
        -------
        dict[str, Any]
            Normalized scalar values with prefixed keys.
        """
        if frame is None or frame.empty:
            return {}

        row = frame.iloc[0].to_dict()

        return {
            f"{prefix}{key}": AcousticIndexProcessor._normalize_value(value)
            for key, value in row.items()
        }

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """
        Convert NumPy scalar values to Python scalars and standardize invalid values.

        Parameters
        ----------
        value : Any
            Raw acoustic index value.

        Returns
        -------
        Any
            Native Python value, or None if value is NaN or infinite.
        """
        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None

        return value

    @staticmethod
    def _validate_indices(acoustic_idx: dict[str, Any]) -> None:
        """
        Validate acoustic index output.

        Parameters
        ----------
        acoustic_idx : dict[str, Any]
            Computed acoustic index dictionary.

        Raises
        ------
        ValueError
            If no indices were computed or at least one value is invalid.
        """
        if not acoustic_idx:
            raise ValueError("No acoustic indices were computed")

        invalid_keys = [
            key
            for key, value in acoustic_idx.items()
            if value is None
        ]

        if invalid_keys:
            preview = ", ".join(invalid_keys[:10])
            suffix = "..." if len(invalid_keys) > 10 else ""
            raise ValueError(f"Invalid acoustic indices found: {preview}{suffix}")
