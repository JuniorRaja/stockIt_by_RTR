"""Configuration management for Stocron by RTR."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional

_config: Optional[Dict[str, Any]] = None


def find_config_path() -> Path:
    """Find the configuration file path."""
    current_dir = Path(__file__).parent.parent.parent
    config_path = current_dir / "config" / "settings.yaml"
    
    if config_path.exists():
        return config_path
    
    cwd_config = Path.cwd() / "config" / "settings.yaml"
    if cwd_config.exists():
        return cwd_config
    
    raise FileNotFoundError("Configuration file not found.")


def load_config(config_path: Optional[str] = None, reload: bool = False) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    global _config
    
    if _config is not None and not reload:
        return _config
    
    path = Path(config_path) if config_path else find_config_path()
    
    with open(path, 'r') as f:
        _config = yaml.safe_load(f)
    
    return _config


def get_config(key: str, default: Any = None) -> Any:
    """Get a configuration value by dot-notation key."""
    config = load_config()
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value


def get_threshold(category: str, name: str, default: Any = None) -> Any:
    """Get threshold values."""
    return get_config(f'thresholds.{category}.{name}', default)


def get_risk_profile(risk_level: str) -> Dict[str, Any]:
    """Get risk profile configuration."""
    return get_config(f'risk_profiles.{risk_level.lower()}', {})


def get_signal_weights() -> Dict[str, float]:
    """Get weights for signal generation."""
    return get_config('signals.weights', {
        'governance': 0.25, 'financial': 0.30, 'valuation': 0.25,
        'market_behaviour': 0.15, 'ml_context': 0.05
    })


def get_macro_rsi_thresholds(default: Optional[Dict[str, Dict[str, int]]] = None) -> Dict[str, Dict[str, int]]:
    """Get market-regime RSI thresholds from config."""
    fallback = default or {
        'bull': {'oversold': 40, 'overbought': 80},
        'bear': {'oversold': 20, 'overbought': 60},
        'sideways': {'oversold': 30, 'overbought': 70},
    }
    thresholds = get_config('macro_config.rsi_thresholds', fallback)
    return thresholds or fallback


def get_red_flag_config(flag_type: str) -> Dict[str, Any]:
    """Get configuration for a specific red flag type."""
    return get_config(f'red_flags.{flag_type}', {})
