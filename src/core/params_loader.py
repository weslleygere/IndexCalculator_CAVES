from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True, init=False)
class ConfigParams:
    """
    Typed parameter set loaded from a JSON configuration file.

    Parameters
    ----------
    params_json_path : str | Path
        Path to the JSON configuration file.
    """

    # Audio preprocessing parameters
    channel: str
    detrend: bool
    segment_duration: int
    clipping_threshold: float
    target_sample_rate: int | None
    normalize_audio: bool
    segment_tolerance_percent: float

    # Index parameters
    compatibility: str

    # Spectrogram parameters
    flims: tuple[int, int]

    def __init__(self, params_json_path: str | Path) -> None:
        payload = self._load_json(params_json_path)

        preprocess = self._get_section(payload, "preprocess")
        spectrogram = self._get_section(payload, "spectrogram")
        indices = self._get_section(payload, "indices")

        self.channel = self._get_required(preprocess, "channel")
        self.detrend = self._get_required(preprocess, "detrend")
        self.segment_duration = self._get_required(preprocess, "segment_duration")
        self.clipping_threshold = self._get_required(preprocess, "clipping_threshold")
        self.target_sample_rate = self._get_optional(preprocess, "target_sample_rate")
        self.normalize_audio = self._get_required(preprocess, "normalize_audio")
        self.segment_tolerance_percent = self._get_required(
            preprocess,
            "segment_tolerance_percent",
        )

        self.compatibility = self._get_required(indices, "compatibility")
        self.flims = tuple(self._get_required(spectrogram, "flims"))

        self._validate()

    @staticmethod
    def _load_json(params_json_path: str | Path) -> dict[str, Any]:
        """
        Load a JSON configuration file.

        Parameters
        ----------
        params_json_path : str | Path
            Path to JSON file.

        Returns
        -------
        dict[str, Any]
            Parsed JSON payload.
        """
        path = Path(params_json_path)

        try:
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Parameter JSON not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON file: {path}") from exc

        if not isinstance(payload, dict):
            raise ValueError("Parameter JSON root must be an object")

        return payload

    @staticmethod
    def _get_section(payload: dict[str, Any], section: str) -> dict[str, Any]:
        """
        Return a required JSON section.

        Parameters
        ----------
        payload : dict[str, Any]
            Parsed JSON payload.
        section : str
            Section name.

        Returns
        -------
        dict[str, Any]
            Section payload.
        """
        value = payload.get(section)

        if not isinstance(value, dict):
            raise ValueError(f"Missing or invalid section: {section}")

        return value

    @staticmethod
    def _get_required(section: dict[str, Any], key: str) -> Any:
        """
        Return a required section value.

        Parameters
        ----------
        section : dict[str, Any]
            JSON section.
        key : str
            Parameter name.

        Returns
        -------
        Any
            Parameter value.
        """
        if key not in section:
            raise ValueError(f"Missing required parameter: {key}")

        return section[key]

    @staticmethod
    def _get_optional(section: dict[str, Any], key: str) -> Any:
        """
        Return an optional section value.

        Parameters
        ----------
        section : dict[str, Any]
            JSON section.
        key : str
            Parameter name.

        Returns
        -------
        Any
            Parameter value or None.
        """
        return section.get(key)

    def _validate(self) -> None:
        """
        Validate all loaded parameters.
        """
        self._validate_audio_params()
        self._validate_index_params()
        self._validate_spectrogram_params()

    def _validate_audio_params(self) -> None:
        """
        Validate audio preprocessing parameters.
        """
        if not isinstance(self.channel, str) or not self.channel.strip():
            raise ValueError("preprocess.channel must be a non-empty string")

        if not isinstance(self.detrend, bool):
            raise ValueError("preprocess.detrend must be boolean")

        if (
            isinstance(self.segment_duration, bool)
            or not isinstance(self.segment_duration, int)
            or self.segment_duration <= 0
        ):
            raise ValueError("preprocess.segment_duration must be an integer > 0")

        if (
            isinstance(self.clipping_threshold, bool)
            or not isinstance(self.clipping_threshold, (int, float))
            or not 0.0 <= float(self.clipping_threshold) <= 1.0
        ):
            raise ValueError("preprocess.clipping_threshold must be between 0 and 1")

        self.clipping_threshold = float(self.clipping_threshold)

        if self.target_sample_rate is not None:
            if (
                isinstance(self.target_sample_rate, bool)
                or not isinstance(self.target_sample_rate, int)
                or self.target_sample_rate <= 0
            ):
                raise ValueError(
                    "preprocess.target_sample_rate must be null or an integer > 0"
                )

        if not isinstance(self.normalize_audio, bool):
            raise ValueError("preprocess.normalize_audio must be boolean")

        if (
            isinstance(self.segment_tolerance_percent, bool)
            or not isinstance(self.segment_tolerance_percent, (int, float))
            or not 0.0 <= float(self.segment_tolerance_percent) <= 100.0
        ):
            raise ValueError(
                "preprocess.segment_tolerance_percent must be between 0 and 100"
            )

        self.segment_tolerance_percent = float(self.segment_tolerance_percent)

    def _validate_index_params(self) -> None:
        """
        Validate acoustic index parameters.
        """
        if not isinstance(self.compatibility, str) or not self.compatibility.strip():
            raise ValueError("indices.compatibility must be a non-empty string")

    def _validate_spectrogram_params(self) -> None:
        """
        Validate spectrogram parameters.
        """
        if (
            not isinstance(self.flims, tuple)
            or len(self.flims) != 2
            or any(isinstance(value, bool) for value in self.flims)
            or not all(isinstance(value, int) for value in self.flims)
        ):
            raise ValueError("spectrogram.flims must be a list with two integers")

        low, high = self.flims

        if low < 0 or high <= low:
            raise ValueError("spectrogram.flims must satisfy 0 <= low < high")
