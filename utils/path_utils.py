import os
from pathlib import Path

from yaml import Loader, load


DEFAULT_DATA_DIR = Path(os.environ.get("GENIE_DATA_DIR", "/data/Genie-Envisioner/data")).expanduser()
DEFAULT_STORAGE_ROOT = Path(os.environ.get("GENIE_STORAGE_ROOT", str(DEFAULT_DATA_DIR.parent))).expanduser()
DEFAULT_WEIGHTS_DIR = Path(os.environ.get("GENIE_WEIGHTS_DIR", str(DEFAULT_STORAGE_ROOT / "weights"))).expanduser()
DEFAULT_OUTPUTS_DIR = Path(os.environ.get("GENIE_OUTPUTS_DIR", str(DEFAULT_STORAGE_ROOT / "outputs"))).expanduser()
DEFAULT_LOGS_DIR = Path(os.environ.get("GENIE_LOGS_DIR", str(DEFAULT_STORAGE_ROOT / "logs"))).expanduser()

_LEGACY_ROOTS = (
    Path("/root/Genie-Envisioner"),
    Path("/home/yanzeyu/Genie-Envisioner"),
)

_PREFIX_REWRITES = (
    ("data", DEFAULT_DATA_DIR),
    ("weights", DEFAULT_WEIGHTS_DIR),
    ("outputs", DEFAULT_OUTPUTS_DIR),
    ("logs", DEFAULT_LOGS_DIR),
)


def storage_paths():
    return {
        "storage_root": str(DEFAULT_STORAGE_ROOT),
        "data_dir": str(DEFAULT_DATA_DIR),
        "weights_dir": str(DEFAULT_WEIGHTS_DIR),
        "outputs_dir": str(DEFAULT_OUTPUTS_DIR),
        "logs_dir": str(DEFAULT_LOGS_DIR),
    }


def apply_storage_env_defaults():
    paths = storage_paths()
    os.environ.setdefault("GENIE_STORAGE_ROOT", paths["storage_root"])
    os.environ.setdefault("GENIE_DATA_DIR", paths["data_dir"])
    os.environ.setdefault("GENIE_WEIGHTS_DIR", paths["weights_dir"])
    os.environ.setdefault("GENIE_OUTPUTS_DIR", paths["outputs_dir"])
    os.environ.setdefault("GENIE_LOGS_DIR", paths["logs_dir"])
    return paths


def rewrite_storage_path(value):
    if not isinstance(value, str):
        return value

    expanded = os.path.expanduser(value)
    for legacy_root in _LEGACY_ROOTS:
        for legacy_name, target_path in _PREFIX_REWRITES:
            legacy_prefix = str(legacy_root / legacy_name)
            if expanded == legacy_prefix:
                return str(target_path)
            if expanded.startswith(f"{legacy_prefix}/"):
                suffix = expanded[len(legacy_prefix) + 1 :]
                return str(target_path / suffix)
    return value


def resolve_storage_paths(obj):
    if isinstance(obj, dict):
        return {key: resolve_storage_paths(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [resolve_storage_paths(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(resolve_storage_paths(value) for value in obj)
    if isinstance(obj, str):
        return rewrite_storage_path(obj)
    return obj


def load_resolved_config(config_file):
    apply_storage_env_defaults()
    with open(config_file, "r") as handle:
        config = load(handle, Loader=Loader)
    return resolve_storage_paths(config)
