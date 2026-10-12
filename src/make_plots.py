from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve

import seaborn as sns
sns.set_theme(style="whitegrid")

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

LABEL_NAMES = {0: "Benign / Likely Benign", 1: "Pathogenic / Likely Pathogenic"}

CORRELATION_FEATURES = ["protein_position","distance_to_domain","plddt_at_pos","plddt_window11","hydro_delta","charge_delta","volume_delta",
                        "blosum62_score","grantham_distance","is_stop_gained","is_frameshift","is_splice_region",
                        "in_functional_domain","grantham_plddt","blosum_unfavorable_plddt","grantham_domain"]

def save_figure(fig, figures_path, filename):
    output_path = Path(figures_path) / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_label_distribution(df, figures_path):
    plot_df = df.copy()
    plot_df["class"] = plot_df["label"].map(LABEL_NAMES)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.countplot(data=plot_df,x="class",ax=ax)
    ax.set_title("BRCA1 Variant Class Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Variants")

    for container in ax.containers:
        ax.bar_label(container)

    save_figure(fig,figures_path,"label_distribution.png")

def plot_variant_position_by_class(df, figures_path):
    plot_df = df.copy()
    plot_df["class"] = plot_df["label"].map(LABEL_NAMES)

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.histplot(data=plot_df,x="protein_position",hue="class",bins=40,element="step",common_norm=False,ax=ax)
    ax.set_title("BRCA1 Variant Position by Class")
    ax.set_xlabel("Protein Position")
    ax.set_ylabel("Variant Count")
    
    save_figure(fig,figures_path,"variant_position_by_class.png")

def plot_plddt_by_position(df, figures_path):
    plot_df = (df[["protein_position", "plddt_at_pos", "label"]].drop_duplicates().copy())
    plot_df["class"] = plot_df["label"].map(LABEL_NAMES)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=plot_df,x="protein_position",y="plddt_at_pos",hue="class",alpha=0.65,s=35,ax=ax)
    ax.set_title("pLDDT Across BRCA1 Protein Positions")
    ax.legend(title="Variant class")
    ax.set_xlabel("Protein Position")
    ax.set_ylabel("pLDDT")

    save_figure(fig,figures_path,"plddt_by_position.png")

def plot_clinical_significance_distribution(df, figures_path):
    order = ["Benign","Likely benign","Likely pathogenic","Pathogenic"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x="clinical_significance", order=order, ax=ax)
    ax.set_title("Clinical Significance Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Variants")
    ax.tick_params(axis="x", rotation=20)

    for container in ax.containers:
        ax.bar_label(container)

    save_figure(fig,figures_path,"clinical_significance_distribution.png")

def plot_domain_pathogenicity(df, figures_path):
    domain_rates = (df.groupby("domain")["label"].mean().mul(100).reindex(["RING", "OTHER", "BRCT1", "BRCT2"]).dropna())

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=domain_rates.index,y=domain_rates.values,ax=ax)
    ax.set_title("Pathogenic Variant Proportion by BRCA1 Domain")
    ax.set_xlabel("Domain")
    ax.set_ylabel("Variants Labeled Pathogenic (%)")
    ax.set_ylim(0, 100)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%")

    save_figure(fig,figures_path,"pathogenicity_by_domain.png")

def plot_feature_correlation(df, figures_path):
    correlations = df[CORRELATION_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(correlations,cmap="coolwarm",annot=True,fmt=".2f",center=0,vmin=-1,vmax=1,square=True,ax=ax)
    ax.set_title("Feature Correlation Matrix")

    save_figure(fig,figures_path,"feature_correlation.png")

def plot_target_correlations(df, figures_path):
    correlations = (df[CORRELATION_FEATURES + ["label"]].corr()["label"].drop("label").sort_values())
    colors = ["green" if value > 0 else "red" for value in correlations.values]

    fig, ax = plt.subplots(figsize=(18, 10))
    bars = ax.barh(correlations.index,correlations.values,color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Feature Correlation with Pathogenicity")
    ax.set_xlabel("Pearson Correlation with Label")
    ax.set_ylabel("Feature")
    ax.bar_label(bars,labels=[f"{value:.2f}" for value in correlations.values],padding=3)

    save_figure(fig,figures_path,"target_correlation.png")

def plot_consequence_pathogenicity(df, figures_path):
    consequence_columns = {"is_stop_gained": "Stop gained", "is_frameshift": "Frameshift","is_splice_region": "Splice region"}
    rows = []

    for column, name in consequence_columns.items():
        for value, status in [(0, "Absent"), (1, "Present")]:
            rate = (df.loc[df[column] == value, "label"].mean() * 100)
            rows.append({"consequence": name,"status": status,"pathogenic_rate": rate})

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=plot_df,x="consequence", y="pathogenic_rate",hue="status",ax=ax)
    ax.set_title("Pathogenicity by Variant Consequence")
    ax.set_xlabel("")
    ax.set_ylabel("Variants Labeled Pathogenic (%)")
    ax.set_ylim(0, 100)

    save_figure(fig,figures_path,"consequence_pathogenicity.png")

# Training and evaluation plots
def plot_validation_pr_curves(predictions, figures_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    for run_name, pred_df in predictions.items():
        precision, recall, _ = precision_recall_curve(pred_df["label"],pred_df["probability"])
        ap = average_precision_score(pred_df["label"],pred_df["probability"])
        ax.plot(recall,precision,label=f"{run_name} (AP={ap:.3f})")

    ax.set_title("Validation Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8)

    save_figure(fig,figures_path,"validation_pr_curves.png")

def plot_validation_roc_curves(predictions, figures_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    for run_name, pred_df in predictions.items():
        fpr, tpr, _ = roc_curve(pred_df["label"],pred_df["probability"])
        auc = roc_auc_score(pred_df["label"],pred_df["probability"])
        ax.plot(fpr,tpr,label=f"{run_name} (AUC={auc:.3f})")

    ax.plot([0, 1],[0, 1],linestyle="--",linewidth=1)
    ax.set_title("Validation ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)

    save_figure(fig,figures_path, "validation_roc_curves.png")

def plot_experiment_comparison(results_df, figures_path):
    plot_df = results_df.copy()
    plot_df["run"] = (plot_df["model"] + " | " + plot_df["experiment"])
    plot_df = plot_df.sort_values("val_average_precision", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(plot_df["run"],plot_df["val_average_precision"])
    ax.bar_label(bars,labels=[f"{value:.3f}" for value in plot_df["val_average_precision"]],padding=3)
    ax.set_title("Validation Average Precision by Experiment")
    ax.set_xlabel("Average Precision")
    ax.set_ylabel("")

    save_figure(fig,figures_path,"experiment_comparison.png")

def plot_threshold_metrics(threshold_df, selected_threshold, figures_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    metrics = ["precision","recall","specificity","f1","balanced_accuracy"]

    for metric in metrics:
        ax.plot(threshold_df["threshold"],threshold_df[metric],label=metric.replace("_", " ").title())

    ax.axvline(selected_threshold,linestyle="--",color="black",label=f"Selected threshold = {selected_threshold:.2f}")
    ax.set_title("Validation Metrics Across Classification Thresholds")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1)
    ax.legend()

    save_figure(fig,figures_path,"threshold_metrics.png")

def plot_calibration(y_true,probabilities,figures_path,title="Calibration Curve",filename="calibration_curve.png"):
    fraction_positive, mean_predicted = calibration_curve(y_true,probabilities,n_bins=10,strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_predicted,fraction_positive,marker="o",label="Model")
    ax.plot([0, 1],[0, 1],linestyle="--",color="black",label="Perfect calibration")
    ax.set_title(title)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Pathogenic Fraction")
    ax.legend()

    save_figure(fig,figures_path,filename)

# Test plots
def plot_test_roc_curve(y_true, probabilities, figures_path):
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc = roc_auc_score(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.set_title("Test ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    save_figure(fig, figures_path, "test_roc_curve.png")

def plot_test_pr_curve(y_true, probabilities, figures_path):
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    ap = average_precision_score(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"Average Precision = {ap:.3f}")
    ax.set_title("Test Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()

    save_figure(fig, figures_path, "test_pr_curve.png")

def plot_test_confusion_matrix(y_true, predictions, figures_path):
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign","Pathogenic"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.grid(False)
    ax.set_title("Test Confusion Matrix")

    save_figure(fig, figures_path, "test_confusion_matrix.png")

# Explainability plot
def plot_feature_importance(importance_df, figures_path):
    plot_df = importance_df.sort_values("importance", ascending=True).copy()
    colors = ["green" if coef > 0 else "red" for coef in plot_df["coefficient"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["feature"], plot_df["importance"], color=colors)
    ax.set_title("Feature Importance for Final Logistic Regression Model")
    ax.set_xlabel("Absolute Standardized Coefficient")
    ax.set_ylabel("Feature")

    for i, coef in enumerate(plot_df["coefficient"]):
        ax.text(plot_df["importance"].iloc[i] + 0.01, i, f"{coef:.3f}", va="center")

    save_figure(fig, figures_path, "feature_importance_bar.png")