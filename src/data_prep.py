import pandas as pd

from utils import PROJECT_ROOT, load_config, ensure_dir
from features import add_engineered_features
from make_plots import (
    plot_label_distribution,
    plot_clinical_significance_distribution,
    plot_variant_position_by_class,
    plot_plddt_by_position,
    plot_domain_pathogenicity,
    plot_feature_correlation,
    plot_target_correlations,
    plot_consequence_pathogenicity
)

def validate_data(df, config):
    required_columns = {"clinical_significance","ref_aa","alt_aa","domain","plddt_at_pos","plddt_window11"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    positive_labels = config["data"]["positive_labels"]
    negative_labels = config["data"]["negative_labels"]
    valid_labels = positive_labels + negative_labels

    # Validate rows to be used
    valid_df = df[df["clinical_significance"].isin(valid_labels)]
    if valid_df.empty:
        raise ValueError("No valid clinical significance classes found.")

    feature_inputs = ["ref_aa","alt_aa","domain","plddt_at_pos","plddt_window11"]

    if valid_df[feature_inputs].isna().any().any():
        missing_counts = (valid_df[feature_inputs].isna().sum())
        missing_counts = missing_counts[missing_counts > 0]

        raise ValueError(f"Missing values in required feature columns:\n{missing_counts}")

    standard_aa = set("ARNDCQEGHILKMFPSTWYV")
    observed_aa = (set(valid_df["ref_aa"]) | set(valid_df["alt_aa"]))
    invalid_aa = observed_aa - standard_aa

    if invalid_aa:
        raise ValueError(f"Unexpected amino acids found: {invalid_aa}")

    valid_domains = {"RING", "BRCT1", "BRCT2", "OTHER"}
    invalid_domains = (set(valid_df["domain"]) - valid_domains)

    if invalid_domains:
        raise ValueError(f"Unexpected domains found: {invalid_domains}")

def prepare_data(df, config):

    validate_data(df, config)

    positive_labels = config["data"]["positive_labels"]
    negative_labels = config["data"]["negative_labels"]
    valid_labels = positive_labels + negative_labels

    # original raw classes
    df = df[df["clinical_significance"].isin(valid_labels)].copy()

    # binary target
    derived_label = (df["clinical_significance"].isin(positive_labels).astype(int))

    # Verify target labels
    if "label" in df.columns:
        if not (df["label"].to_numpy() == derived_label.to_numpy()).all():
            raise ValueError("Existing label does not match clinical_significance.")

    df["label"] = derived_label

    # Add engineered features
    df = add_engineered_features(df)

    return df

def plot_data_figures(df, config):
    figures_path = (PROJECT_ROOT / config["paths"]["data_figures"])
    ensure_dir(figures_path)

    plot_label_distribution(df, figures_path)
    plot_clinical_significance_distribution(df, figures_path)
    plot_variant_position_by_class(df, figures_path)
    plot_plddt_by_position(df, figures_path)
    plot_domain_pathogenicity(df, figures_path)
    plot_feature_correlation(df, figures_path)
    plot_target_correlations(df, figures_path)
    plot_consequence_pathogenicity(df, figures_path)

def main():
    config = load_config()
    raw_path = (PROJECT_ROOT / config["paths"]["raw_data"])
    processed_path = (PROJECT_ROOT / config["paths"]["processed_data"])

    df = pd.read_csv(raw_path)
    df = prepare_data(df, config)

    base_features = config["features"]["base"]
    engineered_features = config["features"]["engineered"]

    output_columns = (config["features"]["metadata"] + base_features + engineered_features + [config["data"]["target"]])

    missing_outputs = (set(output_columns) - set(df.columns))

    if missing_outputs:
        raise ValueError(f"Configured output columns not found: {missing_outputs}")

    df = df[output_columns]

    ensure_dir(processed_path.parent)
    df.to_csv(processed_path,index=False)

    plot_data_figures(df, config)
    print(f"Saved {df.shape[0]} rows and {df.shape[1]} columns to {processed_path}")

if __name__ == "__main__":
    main()