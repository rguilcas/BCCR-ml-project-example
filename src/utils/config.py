"""
File documentation:
This file defines a utility function for loading configuration files in YAML format.
"""

import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)