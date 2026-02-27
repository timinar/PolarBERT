import yaml
from typing import Any

def load_and_process_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Convert string numbers to proper types
    for section in config:
        for key, value in config[section].items():
            if isinstance(value, str):
                try:
                    if '.' in value:
                        config[section][key] = float(value)
                    else:
                        config[section][key] = int(value)
                except ValueError:
                    pass
    return config