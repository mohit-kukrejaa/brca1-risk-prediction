from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
sns.set_theme(style="whitegrid")

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