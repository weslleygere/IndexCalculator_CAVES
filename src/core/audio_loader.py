from pathlib import Path

import numpy as np
from maad import sound

from .params_loader import ConfigParams
from .utils import AudioMetadata


class AudioLoader:
    """
    Load WAV files using channel and detrend settings from configuration.

    Parameters
    ----------
    config : ConfigParams
        Typed configuration containing audio loading parameters.
    """

    def __init__(self, config: ConfigParams) -> None:
        self.channel = config.channel
        self.detrend = config.detrend

    def load_audio(self, file_path: Path | str) -> AudioMetadata:
        """
        Load one WAV file.

        Parameters
        ----------
        file_path : Path | str
            Path to the WAV file.

        Returns
        -------
        AudioMetadata
            Loaded audio metadata or a failed metadata object.
        """
        path = Path(file_path).resolve()

        try:
            wave, sample_rate = sound.load(
                str(path),
                channel=self.channel,
                detrend=self.detrend,
            )

            if wave is None:
                raise ValueError("Loaded waveform is None")

            if sample_rate is None:
                raise ValueError("Loaded sample rate is None")

            return AudioMetadata.success(
                stage="load",
                file_name=path.name,
                directory_name=path.parent.name,
                wave=np.asarray(wave, dtype=np.float32),
                sample_rate=int(sample_rate),
            )

        except Exception as exc:
            return self._fail(path, exc)

    @staticmethod
    def _fail(path: Path, exc: Exception) -> AudioMetadata:
        """
        Build a failed load result.

        Parameters
        ----------
        path : Path
            Source audio path.
        exc : Exception
            Raised exception.

        Returns
        -------
        AudioMetadata
            Failed load metadata.
        """
        return AudioMetadata.fail(
            stage="load",
            file_name=path.name,
            directory_name=path.parent.name,
            error=str(exc),
            error_type=type(exc).__name__,
        )
