"""
Default configuration.
These settings are overridden by the settings in the config.toml file,
if present.
"""

CONST_DEFAULT_CONFIG = {
    "vtt_cleaning": {
        "input_directory": "",
        "output_directory": "",
    },
    "wer_calculation": {
        "reference_directory": "",
        "hypothesis_directory": "",
    },
    "custom_dataset_config": {
        "dataset_directory": "",
    },
}
