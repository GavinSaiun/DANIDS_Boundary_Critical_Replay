from pathlib import Path
import pandas as pd


def inspect_dataset(name: str, csv_path: Path, preview_rows: int = 5, sample_rows: int = 10000):
    print("=" * 80)
    print(f"DATASET: {name}")
    print(f"PATH: {csv_path}")
    print("=" * 80)

    if not csv_path.exists():
        print(f"[ERROR] File does not exist: {csv_path}")
        return None

    # Small preview
    df_preview = pd.read_csv(csv_path, nrows=preview_rows)
    print("\n[1] Preview")
    print(df_preview.head())
    print("\nColumns:")
    print(df_preview.columns.tolist())

    # Slightly larger sample for schema inspection
    df_sample = pd.read_csv(csv_path, nrows=sample_rows)

    print("\n[2] Shape of sample")
    print(df_sample.shape)

    print("\n[3] Dtypes")
    print(df_sample.dtypes)

    print("\n[4] Missing values (top 20)")
    missing = df_sample.isna().sum().sort_values(ascending=False)
    print(missing.head(20))

    print("\n[5] Number of unique values per column (top 20)")
    nunique = df_sample.nunique(dropna=False).sort_values(ascending=False)
    print(nunique.head(20))

    # Try to detect likely label columns
    possible_label_cols = [col for col in df_sample.columns if "label" in col.lower() or "attack" in col.lower()]
    print("\n[6] Possible label columns")
    print(possible_label_cols if possible_label_cols else "None found automatically")

    for col in possible_label_cols:
        print(f"\nUnique values for candidate label column: {col}")
        print(df_sample[col].value_counts(dropna=False).head(20))

    return df_sample


def compare_columns(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str):
    print("\n" + "=" * 80)
    print("COLUMN COMPARISON")
    print("=" * 80)

    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    only_in_1 = sorted(cols1 - cols2)
    only_in_2 = sorted(cols2 - cols1)
    common = sorted(cols1 & cols2)

    print(f"\nColumns only in {name1}: {len(only_in_1)}")
    print(only_in_1)

    print(f"\nColumns only in {name2}: {len(only_in_2)}")
    print(only_in_2)

    print(f"\nCommon columns: {len(common)}")
    print(common)

    same_order = list(df1.columns) == list(df2.columns)
    print(f"\nSame column order? {same_order}")


def main():
    repo_root = Path(__file__).resolve().parents[2]

    ton_path = repo_root / "Datasets" / "NF-ToN-IoT-v3" / "02934b58528a226b_NFV3DATA-A11964_A11964" / "data" / "NF-ToN-IoT-v3.csv"
    unsw_path = repo_root / "Datasets" / "NF-UNSW-NB15-v3" / "f7546561558c07c5_NFV3DATA-A11964_A11964" / "data" / "NF-UNSW-NB15-v3.csv"

    ton_df = inspect_dataset("NF-ToN-IoT-v3", ton_path)
    unsw_df = inspect_dataset("NF-UNSW-NB15-v3", unsw_path)

    if ton_df is not None and unsw_df is not None:
        compare_columns(ton_df, unsw_df, "NF-ToN-IoT-v3", "NF-UNSW-NB15-v3")


if __name__ == "__main__":
    main()