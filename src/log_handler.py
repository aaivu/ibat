import logging


class LogHandler:
    def __init__(self) -> None:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        # Create a logger
        self._logger = logging.getLogger()
        self._logger.setLevel(logging.DEBUG)

        # Create a console handler and set the level to DEBUG
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        # Create a file handler and set the level to DEBUG
        file_handler = logging.FileHandler("example.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self._logger.addHandler(console_handler)
        self._logger.addHandler(file_handler)

        # Example usage
        # logger.debug("This is a debug message")
        # logger.info("This is an info message")
        # logger.warning("This is a warning message")
        # logger.error("This is an error message")
        # logger.critical("This is a critical message")
