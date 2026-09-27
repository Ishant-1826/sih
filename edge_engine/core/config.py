"""
Configuration loader for IDR-GNSS-FUSION.

Loads YAML configuration files and provides typed access to parameters.
Supports default + override pattern for experiment configs.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "default.yaml"


class Config:
    """Hierarchical configuration with dot-notation access."""

    def __init__(self, data: dict):
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        """Get nested key using dot notation, e.g. 'sensor.imu_frequency'."""
        keys = key.split('.')
        val = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return super().__getattribute__(name)
        val = self._data.get(name)
        if isinstance(val, dict):
            return Config(val)
        return val

    def __repr__(self) -> str:
        return f"Config({self._data})"

    def to_dict(self) -> dict:
        return self._data.copy()


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML config. Uses default.yaml if None.
        overrides: Dict of overrides to apply on top.

    Returns:
        Config object with all parameters.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    if overrides:
        _deep_merge(data, overrides)

    return Config(data)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base
