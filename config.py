from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
WORKSPACE.mkdir(parents=True, exist_ok=True)