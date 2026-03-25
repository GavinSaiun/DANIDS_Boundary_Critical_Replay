from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
LABEL_COL = "Label"


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_source(df: pd.DataFrame):
    train, temp = train_test_split(
        df,
        test_size=0.30,
        stratify=df[LABEL_COL],
        random_state=RANDOM_STATE,
    )

    val, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp[LABEL_COL],
        random_state=RANDOM_STATE,
    )

    return train, val, test


def split_target(df: pd.DataFrame):
    adapt, test = train_test_split(
        df,
        test_size=0.80,
        stratify=df[LABEL_COL],
        random_state=RANDOM_STATE,
    )

    adapt_train, adapt_val = train_test_split(
        adapt,
        test_size=0.50,
        stratify=adapt[LABEL_COL],
        random_state=RANDOM_STATE,
    )

    return adapt_train, adapt_val, test


def print_distribution(name: str, df: pd.DataFrame):
    print(f"\n{name} distribution:")
    print(df[LABEL_COL].value_counts().sort_index())


def main():
    repo_root = get_repo_root()

    sampled_dir = repo_root / "data" / "sampled"
    splits_dir = repo_root / "data" / "splits"
    ensure_dir(splits_dir)

    source_path = sampled_dir / "NF-ToN-IoT-v3_sampled_balanced.csv"
    target_path = sampled_dir / "NF-UNSW-NB15-v3_sampled_balanced.csv"

    source_df = pd.read_csv(source_path)
    target_df = pd.read_csv(target_path)

    # =====================
    # Source splits
    # =====================
    s_train, s_val, s_test = split_source(source_df)

    print_distribution("Source train", s_train)
    print_distribution("Source val", s_val)
    print_distribution("Source test", s_test)

    s_train.to_csv(splits_dir / "source_train.csv", index=False)
    s_val.to_csv(splits_dir / "source_val.csv", index=False)
    s_test.to_csv(splits_dir / "source_test.csv", index=False)

    # =====================
    # Target splits
    # =====================
    t_adapt_train, t_adapt_val, t_test = split_target(target_df)

    print_distribution("Target adapt train", t_adapt_train)
    print_distribution("Target adapt val", t_adapt_val)
    print_distribution("Target test", t_test)

    t_adapt_train.to_csv(splits_dir / "target_adapt_train.csv", index=False)
    t_adapt_val.to_csv(splits_dir / "target_adapt_val.csv", index=False)
    t_test.to_csv(splits_dir / "target_test.csv", index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()