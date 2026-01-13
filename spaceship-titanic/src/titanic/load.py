from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

TARGET_COLUMN = "Transported"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir(base_dir: Optional[Path] = None) -> Path:
    root = project_root() if base_dir is None else Path(base_dir)
    return root / "data" / "raw"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(path)


def load_train_test(base_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    directory = data_dir(base_dir)
    train_df = _read_csv(directory / "train.csv")
    test_df = _read_csv(directory / "test.csv")
    return train_df, test_df


def load_submission_template(base_dir: Optional[Path] = None) -> pd.DataFrame:
    directory = data_dir(base_dir)
    return _read_csv(directory / "sample_submission.csv")
