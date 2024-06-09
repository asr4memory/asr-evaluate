"""
Application configuration.
"""

import os
import toml
from default_config import CONST_DEFAULT_CONFIG

combined_config = {}


def initialize_config():
    "Merges configuration from config.toml with defaults."
    global combined_config

    config_file_path = os.path.join(os.getcwd(), "config.toml")

    with open(config_file_path) as f:
        data = toml.load(f)
        combined_config = {
            "vtt_cleaning": CONST_DEFAULT_CONFIG["vtt_cleaning"] | data["vtt_cleaning"],
            "wer_calculation": CONST_DEFAULT_CONFIG["wer_calculation"]
            | data["wer_calculation"],
        }


def get_config() -> dict:
    "Returns app configuration as a dictionary."
    return combined_config


initialize_config()
