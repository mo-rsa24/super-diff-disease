# src/utils/temp_config.py
import yaml, tempfile, os

def write_temp_config(base_path: str, overrides: dict) -> str:
    """
    Load YAML from base_path, merge overrides (shallow), write to a new tempfile.yaml,
    and return its path. Caller must delete this file when done, if desired.
    """
    with open(base_path, "r") as f:
        cfg = yaml.safe_load(f)

    # shallow‐merge only top‐level keys from overrides
    for k, v in overrides.items():
        cfg[k] = v

    # write to a new temp file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".yml", prefix="cfg_override_")
    os.close(tmp_fd)  # we only need the path
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f)

    return tmp_path
