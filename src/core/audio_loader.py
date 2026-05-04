from pathlib import Path

from maad import sound
import numpy as np

from .params_loader import ConfigParams
from .utils import AudioMetadata

import logging
logger = logging.getLogger(__name__)


class AudioLoader:
    """
    Load audio files from a directory using channel/detrend settings from config.
    
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
        Load a single WAV file.

        Parameters
        ----------
        file_path : Path | str
            Path to the audio file.

        Returns
        -------
        AudioMetadata
            Standardized load result including status, payload and error metadata.
        """
        path_obj = Path(file_path).resolve()
        directory_name = path_obj.parent.name

        try:
            wave, sr = sound.load(str(path_obj), channel=self.channel, detrend=self.detrend)
            return AudioMetadata(
                stage="load",
                status="success",
                file_name=path_obj.name,
                segment_id=path_obj.name,
                directory_name=directory_name,
                wave=np.asarray(wave, dtype=np.float32),
                sample_rate=sr,
            )

        except Exception as e:
            return AudioMetadata(
                stage="load",
                status="failed",
                file_name=path_obj.name,
                segment_id=path_obj.name,
                directory_name=directory_name,
                error=str(e),
                error_type=type(e).__name__,
            )
        