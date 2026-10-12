import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss
)

from make_plots import (
    plot_threshold_metrics,
    plot_calibration,
    plot_test_roc_curve,
    plot_test_pr_curve,
    plot_test_confusion_matrix,
    plot_feature_importance
)

from utils import PROJECT_ROOT,load_config,ensure_dir

def choose_best_model(data_config):
    summary_path = (PROJECT_ROOT / data_config["paths"]["tables"] / "best_models_summary.csv")
    summary = pd.read_csv(summary_path)
    best_row = summary.loc[summary["val_average_precision"].idxmax()]

    return best_row

def calculate_threshold_metrics(y_true,probabilities,threshold_config):
    thresholds = np.round(np.arange(threshold_config["min"],threshold_config["max"] + threshold_config["step"],threshold_config["step"]),2)
    rows = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true,predictions,labels=[0, 1]).ravel()
        specificity = (tn / (tn + fp) if (tn + fp) > 0 else 0)

        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_true,predictions,zero_division=0),
            "recall": recall_score(y_true,predictions,zero_division=0),
            "specificity": specificity,
            "f1": f1_score(y_true,predictions,zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true,predictions)
        })

    return pd.DataFrame(rows)

def select_threshold(threshold_df,metric):
    if metric == "balanced_accuracy":
        best_index = threshold_df["balanced_accuracy"].idxmax()
        return threshold_df.loc[best_index]

    if metric == "sensitivity_target":
        candidates = threshold_df[threshold_df["recall"] >= 0.75]

        if candidates.empty:
            return threshold_df.loc[threshold_df["recall"].idxmax()]

        best_index = candidates["specificity"].idxmax()
        return threshold_df.loc[best_index]

def calculate_metrics(y_true,probabilities,predictions):
    tn, fp, fn, tp = confusion_matrix(y_true,predictions,labels=[0, 1]).ravel()
    specificity = (tn / (tn + fp) if (tn + fp) > 0 else 0)
    metrics = {
        "average_precision": average_precision_score(y_true,probabilities),
        "roc_auc": roc_auc_score(y_true,probabilities),
        "precision": precision_score(y_true,predictions,zero_division=0),
        "recall": recall_score(y_true,predictions,zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true,predictions,zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true,predictions),
        "brier_score": brier_score_loss(y_true,probabilities),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp
    }

    return metrics

def load_test_data(data_config):
    data_path = (PROJECT_ROOT / data_config["paths"]["processed_data"])
    split_path = (PROJECT_ROOT / data_config["paths"]["split_data"])

    df = pd.read_csv(data_path)
    splits = pd.read_csv(split_path)
    df = df.merge(splits[["variation_id","split"]],on="variation_id",how="left",validate="one_to_one")
    test_df = df[df["split"] == "test"].copy()

    if test_df.empty:
        raise ValueError("Test set is empty.")

    return test_df

def get_feature_importance(model,feature_names,model_name):
    final_model = model.named_steps["model"]

    if model_name == "logistic_regression":
        values = final_model.coef_[0]
        coefficients = values
        importance = np.abs(values)

    elif model_name in {"random_forest","xgboost"}:
        values = final_model.feature_importances_
        coefficients = values
        importance = values

    else:
        raise ValueError(f"Feature importance not supported for {model_name}")

    if len(feature_names) != len(values):
        raise ValueError("Feature names do not match the number of final model features.")

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "importance": importance
    })

    return importance_df.sort_values("importance",ascending=False)

def add_error_types(test_predictions):
    conditions = [
        (test_predictions["label"] == 1) & (test_predictions["prediction"] == 1),
        (test_predictions["label"] == 0) & (test_predictions["prediction"] == 0),
        (test_predictions["label"] == 0) & (test_predictions["prediction"] == 1),
        (test_predictions["label"] == 1) & (test_predictions["prediction"] == 0)
    ]

    choices = ["TP","TN","FP","FN"]
    test_predictions["error_type"] = np.select(conditions,choices,default="Unknown")

    return test_predictions

def run_test_evaluation(data_config,best_model_row,selected_threshold,tables_path,test_path):
    model_name = best_model_row["model"]
    experiment_name = best_model_row["experiment"]

    model_path = (PROJECT_ROOT / data_config["paths"]["models"] / model_name / "best_model.joblib")
    input_feature_path = (tables_path / model_name / "input_features.csv")
    selected_feature_path = (tables_path / model_name / "selected_features.csv")

    model = joblib.load(model_path)
    input_features = pd.read_csv(input_feature_path)["feature"].tolist()
    selected_features = pd.read_csv(selected_feature_path)["feature"].tolist()

    test_df = load_test_data(data_config)

    missing_features = set(input_features) - set(test_df.columns)
    if missing_features:
        raise ValueError(f"Missing test features: {missing_features}")

    X_test = test_df[input_features]
    y_test = test_df["label"]

    if X_test.isna().any().any():
        raise ValueError("Missing values found in test features.")

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= selected_threshold).astype(int)

    metrics = calculate_metrics(y_test,probabilities,predictions)
    metrics["model"] = model_name
    metrics["experiment"] = experiment_name
    metrics["threshold"] = selected_threshold

    pd.DataFrame([metrics]).to_csv(test_path / "test_metrics.csv",index=False)
    prediction_columns = ["variation_id","protein_change","protein_position","ref_aa","alt_aa","clinical_significance","domain","label"]
    prediction_columns = [column for column in prediction_columns if column in test_df.columns]

    test_predictions = test_df[prediction_columns].copy()
    test_predictions["probability"] = probabilities
    test_predictions["prediction"] = predictions
    test_predictions["correct"] = (predictions == y_test.to_numpy()).astype(int)
    test_predictions = add_error_types(test_predictions)
    test_predictions.to_csv(test_path / "test_predictions.csv",index=False)

    confusion_df = pd.DataFrame(confusion_matrix(y_test,predictions,labels=[0, 1]),index=["Actual Benign","Actual Pathogenic"],columns=["Predicted Benign","Predicted Pathogenic"])
    confusion_df.to_csv(test_path / "test_confusion_matrix.csv")

    plot_test_roc_curve(y_test,probabilities,test_path)
    plot_test_pr_curve(y_test,probabilities,test_path)
    plot_test_confusion_matrix(y_test,predictions,test_path)
    plot_calibration(y_test,probabilities,test_path,title="Test Calibration Curve",filename="test_calibration_curve.png")

    importance_df = get_feature_importance(model,selected_features,model_name)
    importance_df.to_csv(test_path / "feature_importance.csv",index=False)
    plot_feature_importance(importance_df,test_path)

    print("\nFinal Test Results")
    print(f"Model: {model_name}")
    print(f"Experiment: {experiment_name}")
    print(f"Threshold: {selected_threshold:.2f}")
    print(f"Average Precision: {metrics['average_precision']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Specificity: {metrics['specificity']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")
    print(f"Brier score: {metrics['brier_score']:.3f}")
    print(f"Confusion matrix: TN={metrics['true_negative']}, FP={metrics['false_positive']}, FN={metrics['false_negative']}, TP={metrics['true_positive']}")

def main():
    data_config = load_config()
    model_config = load_config("model_config.toml")
    best_model = choose_best_model(data_config)
    model_name = best_model["model"]

    print(f"Selected model: {model_name}")
    print(f"Validation AP: {best_model['val_average_precision']:.3f}")

    tables_path = (PROJECT_ROOT / data_config["paths"]["tables"])
    figures_path = (PROJECT_ROOT / data_config["paths"]["model_figures"])
    threshold_tables = (tables_path / "threshold")
    threshold_figures = (figures_path / "threshold")
    calibration_figures = (figures_path / "calibration")
    test_path = (PROJECT_ROOT / "outputs" / "test")

    ensure_dir(threshold_tables)
    ensure_dir(threshold_figures)
    ensure_dir(calibration_figures)
    ensure_dir(test_path)

    prediction_path = (tables_path / model_name / "validation_predictions.csv")
    validation_predictions = pd.read_csv(prediction_path)

    y_val = validation_predictions["label"]
    val_probabilities = validation_predictions["probability"]

    threshold_df = calculate_threshold_metrics(y_val,val_probabilities,model_config["threshold"])
    selected = select_threshold(threshold_df,model_config["threshold"]["metric"])
    selected_threshold = float(selected["threshold"])

    threshold_df.to_csv(threshold_tables / "threshold_results.csv",index=False)
    selected.to_frame().T.to_csv(threshold_tables / "selected_threshold.csv",index=False)

    plot_threshold_metrics(threshold_df,selected_threshold,threshold_figures)
    plot_calibration(y_val,val_probabilities,calibration_figures,title="Validation Calibration Curve",filename="validation_calibration_curve.png")

    validation_brier = brier_score_loss(y_val,val_probabilities)

    print(f"\nSelected threshold: {selected_threshold:.2f}")
    print(f"Balanced accuracy: {selected['balanced_accuracy']:.3f}")
    print(f"Recall: {selected['recall']:.3f}")
    print(f"Precision: {selected['precision']:.3f}")
    print(f"Specificity: {selected['specificity']:.3f}")
    print(f"F1: {selected['f1']:.3f}")
    print(f"Brier score: {validation_brier:.3f}")

    run_test_evaluation(data_config,best_model,selected_threshold,tables_path,test_path)

if __name__ == "__main__":
    main()