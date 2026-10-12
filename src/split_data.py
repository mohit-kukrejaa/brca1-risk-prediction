import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from utils import PROJECT_ROOT, load_config, ensure_dir

def create_splits(df, config):
    split_config = config["split"]
    train_size = split_config["train_size"]
    val_size = split_config["validation_size"]
    test_size = split_config["test_size"]
    group_column = split_config["group_column"]
    random_state = split_config["random_state"]

    total_size = train_size + val_size + test_size

    if abs(total_size - 1.0) > 1e-8:
        raise ValueError("Train, validation, and test sizes must sum to 1.")

    # Train vs validation + test
    first_split = GroupShuffleSplit(n_splits=1,train_size=train_size,random_state=random_state)
    train_pos, temp_pos = next(first_split.split(df,groups=df[group_column]))

    train_idx = df.index[train_pos]
    temp_df = df.iloc[temp_pos]

    # Validation vs test
    rem_size = val_size + test_size
    relative_val_size = (val_size / rem_size)
    second_split = GroupShuffleSplit(n_splits=1,train_size=relative_val_size,random_state=random_state)

    val_pos, test_pos = next(second_split.split(temp_df,groups=temp_df[group_column]))

    val_idx = temp_df.index[val_pos]
    test_idx = temp_df.index[test_pos]

    splits = pd.Series(index=df.index,dtype="object")
    splits.loc[train_idx] = "train"
    splits.loc[val_idx] = "validation"
    splits.loc[test_idx] = "test"

    return splits

def validate_splits(df, splits, group_column):
    if splits.isna().any():
        raise ValueError("Some rows were not assigned to a split.")

    train_groups = set(df.loc[splits == "train", group_column])
    val_groups = set(df.loc[splits == "validation", group_column])
    test_groups = set(df.loc[splits == "test", group_column])

    if train_groups & val_groups:
        raise ValueError("Group overlap found between train and validation.")

    if train_groups & test_groups:
        raise ValueError("Group overlap found between train and test.")

    if val_groups & test_groups:
        raise ValueError("Group overlap found between validation and test.")

def print_split_summary(df, splits):
    summary_df = df.copy()
    summary_df["split"] = splits

    summary = (summary_df.groupby("split").agg(rows=("label", "size"),pathogenic=("label", "sum"),pathogenic_pct=("label", "mean")).reindex(["train", "validation", "test"]))
    summary["pathogenic_pct"] = (summary["pathogenic_pct"] * 100).round(1)

    print("\nSplit summary:")
    print(summary)

def save_splits(df, splits, group_column, output_path):
    if not df["variation_id"].is_unique:
        raise ValueError("variation_id must be unique.")

    split_df = df[["variation_id", group_column, "label"]].copy()
    split_df["split"] = splits

    ensure_dir(output_path.parent)

    split_df.to_csv(output_path,index=False)

def main():
    data_config = load_config()
    model_config = load_config("model_config.toml")
    data_path = (PROJECT_ROOT / data_config["paths"]["processed_data"])
    split_path = (PROJECT_ROOT / data_config["paths"]["split_data"])

    df = pd.read_csv(data_path)

    group_column = (model_config["split"]["group_column"])
    splits = create_splits(df,model_config)

    validate_splits(df,splits,group_column)
    print_split_summary(df,splits)
    save_splits(df,splits,group_column,split_path)

    print(f"\nSaved splits to {split_path}")

if __name__ == "__main__":
    main()