"""Download do dataset Telco Customer Churn com verificação de integridade.

O arquivo é o mesmo publicado no Kaggle (blastchar/telco-customer-churn,
licença CC0), espelhado em repositório público. O SHA-256 esperado está
fixado em configs/config.yaml e em data/DATASET_CARD.md — se o mirror for
alterado, o download falha em vez de propagar dados silenciosamente
diferentes pelo pipeline.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import requests

from churn_mlops.config import load_config, resolve


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(cfg: dict | None = None, force: bool = False) -> Path:
    cfg = cfg or load_config()
    raw_path = resolve(cfg["data"]["raw_path"])
    expected = cfg["data"]["raw_sha256"]

    if raw_path.exists() and not force:
        digest = sha256_of(raw_path)
        if digest == expected:
            print(f"[download] dataset já presente e íntegro: {raw_path}")
            return raw_path
        print(f"[download] SHA-256 divergente ({digest[:12]}…), baixando novamente")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    url = cfg["data"]["raw_url"]
    print(f"[download] baixando {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    raw_path.write_bytes(resp.content)

    digest = sha256_of(raw_path)
    if digest != expected:
        raw_path.unlink()
        raise RuntimeError(
            f"SHA-256 do arquivo baixado ({digest}) difere do esperado ({expected}). "
            "O mirror pode ter sido alterado; verifique data/DATASET_CARD.md."
        )
    print(f"[download] ok: {raw_path} (sha256={digest[:12]}…)")
    return raw_path


if __name__ == "__main__":
    download(force="--force" in sys.argv)
