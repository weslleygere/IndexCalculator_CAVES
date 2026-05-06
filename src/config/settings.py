import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    """
    Database connection settings loaded from environment variables.

    Parameters
    ----------
    drivername : str
        SQLAlchemy database driver name.
    username : str | None
        Database username.
    password : str | None
        Database password.
    host : str | None
        Database host.
    port : int
        Database port.
    database : str | None
        Database name.
    """

    drivername: str
    username: Optional[str]
    password: Optional[str]
    host: Optional[str]
    port: int
    database: Optional[str]

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls(
            drivername=os.getenv("DB_DRIVERNAME", "postgresql+psycopg2").strip(),
            username=os.getenv("DB_USERNAME"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_DATABASE"),
        )


@dataclass
class DataConfig:
    """
    Data-related configuration settings.

    Parameters
    ----------
    data_path : Path
        Path to the input audio directory.
    config_params_path : Path
        Path to the JSON file containing index calculation parameters.
    output_dir_base : Path
        Directory where output files will be saved.
    resume_output_dir : Path | None
        Optional existing output directory to resume an interrupted run.

    Attributes
    ----------
    output_dir : Path
        Timestamped directory created within `output_dir_base` to store results.
        It is created explicitly during runtime, not during import.
    """

    data_path         : Path
    config_params_path: Path
    output_dir_base   : Path
    resume_output_dir : Optional[Path] = None
    output_dir        : Path = field(init=False)

    def __post_init__(self) -> None:
        """Validate paths after initialization."""
        self._validate_data_directory()
        self._validate_config_params_file()
        self._validate_output_base()
        self._validate_resume_output_dir()

    @classmethod
    def from_env(cls) -> "DataConfig":
        resume_env = os.getenv("RESUME_OUTPUT_DIR", "")
        
        return cls(
            data_path=Path(os.getenv("DATA_PATH", "data/raw")),
            config_params_path=Path(os.getenv("CONFIG_PARAMS_PATH", "data/parameters/config_params.json")),
            output_dir_base=Path(os.getenv("OUTPUT_DIR", "output")),
            resume_output_dir=Path(resume_env) if resume_env else None,
        )

    def _validate_data_directory(self) -> None:
        """Validate that DATA_PATH exists and is a directory."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        if not self.data_path.is_dir():
            raise NotADirectoryError(f"DATA_PATH is not a directory: {self.data_path}")

    def _validate_output_base(self) -> None:
        """Validate or create the base output directory."""
        if not self.output_dir_base.exists():
            self.output_dir_base.mkdir(parents=True, exist_ok=True)
        elif not self.output_dir_base.is_dir():
            raise NotADirectoryError(f"OUTPUT_DIR is not a directory: {self.output_dir_base}")

    def _validate_config_params_file(self) -> None:
        """Validate CONFIG_PARAMS_PATH existence and format."""
        if not self.config_params_path.exists():
            raise FileNotFoundError(f"Parameter JSON file not found: {self.config_params_path}")

        if not self.config_params_path.is_file():
            raise ValueError(f"Parameter JSON path is not a file: {self.config_params_path}")

        if self.config_params_path.suffix.lower() != ".json":
            raise ValueError(f"CONFIG_PARAMS_PATH must point to a .json file: {self.config_params_path}")

    def _validate_resume_output_dir(self) -> None:
        """Validate RESUME_OUTPUT_DIR when provided."""
        if self.resume_output_dir is None:
            return

        if self.resume_output_dir.exists() and not self.resume_output_dir.is_dir():
            raise NotADirectoryError(
                f"RESUME_OUTPUT_DIR exists but is not a directory: {self.resume_output_dir}"
            )

        self.resume_output_dir.mkdir(parents=True, exist_ok=True)


    def create_output_dir(self) -> Path:
        """
        Create a timestamped output directory once, during controlled execution.

        Returns
        -------
        Path
            Full path to the created timestamped output directory.
        """
        if not hasattr(self, "output_dir"):
            if self.resume_output_dir:
                self.output_dir = self.resume_output_dir
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.output_dir = self.output_dir_base / timestamp
            self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir


@dataclass
class ProcessingConfig:
    """
    Processing-related configuration settings.

    Parameters
    -----------
    max_workers : Optional[int]
        Maximum number of worker processes for parallel processing. Set to 0 for auto-detection.
    """

    max_workers: Optional[int]

    def __post_init__(self) -> None:
        """Validate processing settings after initialization."""
        self._validate_max_workers()

    @classmethod
    def from_env(cls) -> "ProcessingConfig":
        return cls(
            max_workers=int(os.getenv("MAX_WORKERS", "0")),
        )

    def _validate_max_workers(self) -> None:
        """Validate MAX_WORKERS value."""
        if self.max_workers == 0:
            self.max_workers = max(1, (os.cpu_count() or 1) - 1)
            return

        if self.max_workers is not None and self.max_workers < 0:
            raise ValueError("MAX_WORKERS must be 0 (auto) or >= 1")


@dataclass
class LoggingConfig:
    """
    Logging configuration settings.

    Parameters
    ----------
    log_level : str
        Logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    """
    log_level: str

    def __post_init__(self) -> None:
        """Validate and normalize log level after initialization."""
        self.log_level = self.log_level.strip().upper()
        self._validate_log_level()

    @classmethod
    def from_env(cls) -> "LoggingConfig":
        return cls(
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )

    def _validate_log_level(self) -> None:
        """Validate that LOG_LEVEL is one of the accepted values."""
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid LOG_LEVEL: {self.log_level}")


@dataclass
class Settings:
    """
    Settings container aggregating all configuration sections.

    Parameters
    ----------
    database : DatabaseConfig
        Database connection settings loaded from environment variables.
    data : DataConfig
        Data-related configuration settings.
    processing : ProcessingConfig
        Processing-related configuration settings.
    logging : LoggingConfig
        Logging configuration settings.
    """
    database: DatabaseConfig
    data: DataConfig
    processing: ProcessingConfig
    logging: LoggingConfig

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database=DatabaseConfig.from_env(),
            data=DataConfig.from_env(),
            processing=ProcessingConfig.from_env(),
            logging=LoggingConfig.from_env()
        )
    