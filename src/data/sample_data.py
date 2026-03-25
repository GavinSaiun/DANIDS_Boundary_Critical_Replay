from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


RANDOM_STATE = 42
CHUNK_SIZE = 200_000

LABEL_COL = "Label"
ATTACK_COL = "Attack"

DROP_COLS = [
    LABEL_COL,
    ATTACK_COL,
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
]

SOURCE_CSV_REL = Path(
    "Datasets/NF-ToN-IoT-v3/02934b58528a226b_NFV3DATA-A11964_A11964/data/NF-ToN-IoT-v3.csv"
)

TARGET_CSV_REL = Path(
    "Datasets/NF-UNSW-NB15-v3/f7546561558c07c5_NFV3DATA-A11964_A11964/data/NF-UNSW-NB15-v3.csv"
)


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_columns_to_use(csv_path: Path) -> List[str]:
    preview = pd.read_csv(csv_path, nrows=5)
    feature_cols = [c for c in preview.columns if c not in DROP_COLS]
    return feature_cols + [LABEL_COL, ATTACK_COL]


def clean_chunk(chunk: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    chunk = chunk.copy()

    # Clean metadata / target
    chunk[ATTACK_COL] = chunk[ATTACK_COL].astype(str).str.strip()
    chunk[LABEL_COL] = pd.to_numeric(chunk[LABEL_COL], errors="raise").astype("int8")

    # Force features numeric
    for col in feature_cols:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    return chunk


def collect_balanced_sample(
    csv_path: Path,
    dataset_name: str,
    n_benign: int,
    n_attack: int,
) -> pd.DataFrame:
    cols_to_use = get_columns_to_use(csv_path)
    feature_cols = [c for c in cols_to_use if c not in [LABEL_COL, ATTACK_COL]]

    benign_parts = []
    attack_parts = []

    benign_count = 0
    attack_count = 0

    print("=" * 80)
    print(f"Sampling from {dataset_name}")
    print(f"Path: {csv_path}")
    print(f"Target benign: {n_benign:,}")
    print(f"Target attack: {n_attack:,}")
    print("=" * 80)

    for i, chunk in enumerate(pd.read_csv(csv_path, usecols=cols_to_use, chunksize=CHUNK_SIZE), start=1):
        chunk = clean_chunk(chunk, feature_cols)

        benign_chunk = chunk[chunk[LABEL_COL] == 0]
        attack_chunk = chunk[chunk[LABEL_COL] == 1]

        if benign_count < n_benign:
            need_benign = n_benign - benign_count
            take_benign = benign_chunk.sample(
                n=min(need_benign, len(benign_chunk)),
                random_state=RANDOM_STATE,
                replace=False,
            )
            benign_parts.append(take_benign)
            benign_count += len(take_benign)

        if attack_count < n_attack:
            need_attack = n_attack - attack_count
            take_attack = attack_chunk.sample(
                n=min(need_attack, len(attack_chunk)),
                random_state=RANDOM_STATE,
                replace=False,
            )
            attack_parts.append(take_attack)
            attack_count += len(take_attack)

        print(
            f"Chunk {i:03d} | benign collected: {benign_count:,}/{n_benign:,} "
            f"| attack collected: {attack_count:,}/{n_attack:,}"
        )

        if benign_count >= n_benign and attack_count >= n_attack:
            break

    if benign_count < n_benign or attack_count < n_attack:
        raise ValueError(
            f"Not enough samples found in {dataset_name}. "
            f"Collected benign={benign_count:,}, attack={attack_count:,}"
        )

    sampled_df = pd.concat(benign_parts + attack_parts, axis=0, ignore_index=True)

    # Shuffle final sampled dataset
    sampled_df = sampled_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print("\nFinal sampled distribution:")
    print(sampled_df[LABEL_COL].value_counts().sort_index())
    print("\nAttack family breakdown:")
    print(sampled_df[ATTACK_COL].value_counts().head(20))

    return sampled_df


def main() -> None:
    repo_root = get_repo_root()

    source_csv = repo_root / SOURCE_CSV_REL
    target_csv = repo_root / TARGET_CSV_REL

    sampled_dir = repo_root / "data" / "sampled"
    ensure_dir(sampled_dir)

    # Source sample
    source_sample = collect_balanced_sample(
        csv_path=source_csv,
        dataset_name="NF-ToN-IoT-v3",
        n_benign=200_000,
        n_attack=200_000,
    )
    source_out = sampled_dir / "NF-ToN-IoT-v3_sampled_balanced.csv"
    source_sample.to_csv(source_out, index=False)
    print(f"\nSaved source sample to: {source_out}")

    # Target sample
    target_sample = collect_balanced_sample(
        csv_path=target_csv,
        dataset_name="NF-UNSW-NB15-v3",
        n_benign=100_000,
        n_attack=100_000,
    )
    target_out = sampled_dir / "NF-UNSW-NB15-v3_sampled_balanced.csv"
    target_sample.to_csv(target_out, index=False)
    print(f"\nSaved target sample to: {target_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()