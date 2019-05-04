import json
import logging.config
from pathlib import Path
import gym

from rl_agents.configuration import Configurable

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(levelname)s] %(message)s "
        },
        "detailed": {
            "format": "[%(name)s:%(levelname)s] %(message)s "
        }
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler"
        }
    },
    "loggers": {
        "": {
            "handlers": [
                "default"
            ],
            "level": "DEBUG",
            "propagate": True
        }
    }
}


def configure(config={}, gym_level=gym.logger.INFO):
    """
        Configure logging.

        Update the default configuration by a configuration file.
        Also configure the gym logger.

    :param config: logging configuration, or path to a configuration file
    :param gym_level: desired level for gym logger
    """
    if config:
        if isinstance(config, str):
            with Path(config).open() as f:
                config = json.load(f)
        Configurable.rec_update(logging_config, config)
    logging.config.dictConfig(logging_config)
    gym.logger.set_level(gym_level)


def add_file_handler(file_path):
    """
        Add a file handler to the root logger.

    :param Path file_path: log file path
    """
    configure({
        "handlers": {
            file_path.name: {
                "class": "logging.FileHandler",
                "filename": file_path,
                "level": "DEBUG",
                "formatter": "detailed",
                "mode": 'w'
            }
        },
        "loggers": {
            "": {
                "handlers": [
                    file_path.name,
                    *logging_config["handlers"]
                ]
            }
        }
    })
