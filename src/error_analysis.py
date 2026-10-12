import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import PROJECT_ROOT,load_config,ensure_dir

def main():
    config = load_config()
    test_path = (PROJECT_ROOT / "outputs" / "test")
    output_path = (PROJECT_ROOT / "outputs" / "error_analysis")
    data_path = (PROJECT_ROOT / config["paths"]["processed_data"])

    ensure_dir(output_path)

    predictions = pd.read_csv(test_path / "test_predictions.csv")
    features = pd.read_csv(data_path)

    df = predictions.merge(features,on="variation_id",how="left",suffixes=("","_feature"))

    # Error summary
    error_summary = (predictions["error_type"].value_counts().reindex(["TP","TN","FP","FN"],fill_value=0).rename_axis("error_type").reset_index(name="count"))
    error_summary["percent"] = error_summary["count"] / len(predictions) * 100
    error_summary.to_csv(output_path / "error_summary.csv",index=False)

    # Errors by domain
    errors_by_domain = pd.crosstab(df["domain"],df["error_type"])
    errors_by_domain.to_csv(output_path / "errors_by_domain.csv")

    # Error distribution plot
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=predictions,x="error_type",order=["TP","TN","FP","FN"],ax=ax)
    ax.set_title("Test Prediction Outcomes")
    ax.set_xlabel("Prediction Outcome")
    ax.set_ylabel("Number of Variants")

    for container in ax.containers:
        ax.bar_label(container)

    fig.tight_layout()
    fig.savefig(output_path / "error_type_distribution.png",dpi=300,bbox_inches="tight")
    plt.close(fig)

    # Probability distribution by outcome
    fig, ax = plt.subplots(figsize=(9,6))
    sns.boxplot(data=predictions,x="error_type",y="probability",order=["TP","TN","FP","FN"],width=0.55,showfliers=False,ax=ax)
    ax.axhline(0.52,linestyle="--",color="black",label="Threshold = 0.52")
    ax.set_title("Predicted Probability by Test Outcome")
    ax.set_xlabel("Prediction Outcome")
    ax.set_ylabel("Predicted Pathogenic Probability")
    ax.set_ylim(0,1)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(output_path / "probability_by_error_type.png",dpi=300,bbox_inches="tight")
    plt.close(fig)

    # Representative variants
    tp = df[df["error_type"] == "TP"].sort_values("probability",ascending=False).head(1)
    fp = df[df["error_type"] == "FP"].sort_values("probability",ascending=False).head(1)
    fn = df[df["error_type"] == "FN"].sort_values("probability",ascending=True).head(1)
    tn = df[df["error_type"] == "TN"].sort_values("probability",ascending=True).head(1)

    representatives = pd.concat([tp,fp,fn,tn])

    columns = ["variation_id","protein_change","protein_position","clinical_significance","domain","label","prediction","probability","error_type",
               "is_stop_gained","is_frameshift","is_splice_region","plddt_at_pos","plddt_window11",
               "hydro_delta","charge_delta","volume_delta","grantham_distance","grantham_domain","blosum62_score"]

    columns = [column for column in columns if column in representatives.columns]

    representatives[columns].to_csv(output_path / "representative_variants.csv",index=False)

    print("\nError Summary", error_summary.to_string(index=False))
    print("\nErrors by Domain", errors_by_domain)
    print("\nRepresentative Variants", representatives[columns].to_string(index=False))

if __name__ == "__main__":
    main()