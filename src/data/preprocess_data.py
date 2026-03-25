from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import pandas as pd


# =========================
# Configuration
# =========================
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


# =========================
# Helpers
# =========================
def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_preview(csv_path: Path, nrows: int = 10000) -> pd.DataFrame:
    return pd.read_csv(csv_path, nrows=nrows)


def build_feature_list(df: pd.DataFrame) -> List[str]:
    feature_cols = [col for col in df.columns if col not in DROP_COLS]
    return feature_cols


def validate_columns(source_df: pd.DataFrame, target_df: pd.DataFrame) -> None:
    source_cols = list(source_df.columns)
    target_cols = list(target_df.columns)

    if source_cols != target_cols:
        only_in_source = sorted(set(source_cols) - set(target_cols))
        only_in_target = sorted(set(target_cols) - set(source_cols))

        raise ValueError(
            "Source and target columns do not match.\n"
            f"Only in source: {only_in_source}\n"
            f"Only in target: {only_in_target}"
        )


def summarise_dataframe(df: pd.DataFrame, name: str, feature_cols: List[str]) -> None:
    print("=" * 80)
    print(f"{name} SUMMARY")
    print("=" * 80)
    print(f"Rows in preview: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Feature columns after dropping metadata/leakage: {len(feature_cols)}")

    print("\nDropped columns:")
    for col in DROP_COLS:
        print(f"  - {col}")

    print("\nFeature columns:")
    for col in feature_cols:
        print(f"  - {col}")

    print("\nTarget distribution in preview:")
    print(df[LABEL_COL].value_counts(dropna=False).sort_index())

    print("\nAttack distribution in preview:")
    print(df[ATTACK_COL].value_counts(dropna=False).head(20))

    object_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    print("\nObject columns remaining in features:")
    print(object_cols if object_cols else "  None")

    missing = df[feature_cols].isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("\nFeature columns with missing values:")
    if len(missing) == 0:
        print("  None")
    else:
        print(missing)


def convert_feature_types(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df = df.copy()

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="raise").astype("int8")
    df[ATTACK_COL] = df[ATTACK_COL].astype(str).str.strip()

    return df


def save_feature_manifest(feature_cols: List[str], out_path: Path) -> None:
    manifest_df = pd.DataFrame(
        {
            "feature_name": feature_cols,
            "index": list(range(len(feature_cols))),
        }
    )
    manifest_df.to_csv(out_path, index=False)


def save_preview_processed(
    df: pd.DataFrame, feature_cols: List[str], out_path: Path, dataset_name: str
) -> None:
    preview_df = df[feature_cols + [LABEL_COL, ATTACK_COL]].copy()
    preview_df.to_csv(out_path, index=False)
    print(f"\nSaved processed preview for {dataset_name}: {out_path}")


# =========================
# Main
# =========================
def main() -> None:
    repo_root = get_repo_root()

    source_csv = repo_root / SOURCE_CSV_REL
    target_csv = repo_root / TARGET_CSV_REL

    processed_dir = repo_root / "data" / "processed"
    ensure_dir(processed_dir)

    print("Loading previews...")
    source_df = load_preview(source_csv, nrows=10000)
    target_df = load_preview(target_csv, nrows=10000)

    print("Validating source/target schemas...")
    validate_columns(source_df, target_df)

    feature_cols = build_feature_list(source_df)

    source_df = convert_feature_types(source_df, feature_cols)
    target_df = convert_feature_types(target_df, feature_cols)

    summarise_dataframe(source_df, "NF-ToN-IoT-v3", feature_cols)
    print()
    summarise_dataframe(target_df, "NF-UNSW-NB15-v3", feature_cols)

    feature_manifest_path = processed_dir / "feature_manifest_v1.csv"
    save_feature_manifest(feature_cols, feature_manifest_path)
    print(f"\nSaved feature manifest: {feature_manifest_path}")

    source_preview_path = processed_dir / "NF-ToN-IoT-v3_preview_processed.csv"
    target_preview_path = processed_dir / "NF-UNSW-NB15-v3_preview_processed.csv"

    save_preview_processed(source_df, feature_cols, source_preview_path, "NF-ToN-IoT-v3")
    save_preview_processed(target_df, feature_cols, target_preview_path, "NF-UNSW-NB15-v3")

    print("\nDone.")
    print(f"Final feature count: {len(feature_cols)}")


if __name__ == "__main__":
    main()