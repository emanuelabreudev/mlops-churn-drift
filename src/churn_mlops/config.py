"""Carregamento da configuração central (configs/config.yaml).

Todos os caminhos relativos no YAML são resolvidos a partir da raiz do
projeto, para que os scripts funcionem de qualquer diretório de trabalho.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path or os.environ.get("CHURN_CONFIG", DEFAULT_CONFIG))
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve um caminho do config relativo à raiz do projeto."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
