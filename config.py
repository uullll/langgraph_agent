import os
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
WORKSPACE.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = Path(os.getenv("AGENT_CONFIG_PATH", ROOT / "config.yaml"))


def _load_yaml_config() -> dict:
    if not CONFIG_PATH.exists() or yaml is None:
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


CONFIG = _load_yaml_config()


def get_config(path: str, default=None):
    current = CONFIG
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def get_setting(env_name: str, config_path: str, default=None, required: bool = False):
    value = os.getenv(env_name, get_config(config_path, default))
    if required and (value is None or value == ""):
        raise ValueError(
            f"Missing required setting: {env_name} or {config_path}. "
            f"Check AGENT_CONFIG_PATH={CONFIG_PATH} and ensure PyYAML is installed if using YAML."
        )
    return value


workspace_config = get_config("project.workspace", "workspace")
WORKSPACE = (ROOT / workspace_config).resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)