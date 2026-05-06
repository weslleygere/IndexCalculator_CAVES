import logging

from src.config.logging import LogSetup
from src.config.settings import Settings
from src.pipeline import Pipeline


def main() -> None:
    """
    Run the PAM acoustic index pipeline.
    """
    LogSetup.setup_bootstrap_logging()
    logger = logging.getLogger(__name__)

    logging_setup: LogSetup | None = None

    try:
        settings = Settings.from_env()

        logging_setup = LogSetup(settings=settings)
        logging_setup.setup_logging()
        logging_setup.write_log_metadata()

        logger.info("Starting acoustic index pipeline...")

        pipeline = Pipeline(settings=settings)
        pipeline.execute()

        logger.info("Acoustic index pipeline completed successfully.")

    except Exception:
        logger.exception("An unexpected error occurred during pipeline execution.")

    finally:
        if logging_setup is not None:
            logging_setup.cleanup()


if __name__ == "__main__":
    main()
    