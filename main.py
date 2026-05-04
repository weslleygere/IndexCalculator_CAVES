import logging

from src.pipeline import Pipeline
from src.config.logging import LogSetup
from src.config.settings import Settings


def main() -> None:
    """
    Main entry point for the PAM pipeline.
    """
    
    LogSetup.setup_bootstrap_logging()
    logger = logging.getLogger(__name__)

    logging_setup = None

    try:
        settings = Settings.from_env()
        settings.data.create_output_dir()

        logging_setup = LogSetup(settings=settings)
        logging_setup.setup_logging()

        logger.info("Starting pipeline...")
        logging_setup.write_log_metadata()

        pipeline = Pipeline(settings=settings)
        pipeline.run()

        logger.info(f"Evaluation complete. Results saved to {settings.data.output_dir}")

    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)

    finally:
        if logging_setup is not None:
            logging_setup.cleanup()


if __name__ == "__main__":
    main()