import json
import joblib
import pandas as pd

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)

from make_plots import plot_validation_pr_curves, plot_validation_roc_curves, plot_experiment_comparison
from utils import PROJECT_ROOT, load_config, ensure_dir

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_training_data(data_config):
    data_path = PROJECT_ROOT / data_config["paths"]["processed_data"]
    split_path = PROJECT_ROOT / data_config["paths"]["split_data"]

    df = pd.read_csv(data_path)
    splits = pd.read_csv(split_path)

    if not df["variation_id"].is_unique:
        raise ValueError("variation_id must be unique in processed data.")

    if not splits["variation_id"].is_unique:
        raise ValueError("variation_id must be unique in split data.")

    df = df.merge(splits[["variation_id", "split"]],on="variation_id",how="left",validate="one_to_one")

    if df["split"].isna().any():
        raise ValueError("Some variants do not have a split.")

    return df

def get_experiments(data_config):
    base_features = data_config["features"]["base"]
    engineered_features = data_config["features"]["engineered"]
    all_features = list(dict.fromkeys(base_features + engineered_features))

    experiments = {"base": {"features": base_features,"use_selection": False},
        "base_plus_engineered": {"features": all_features,"use_selection": False},
        "model_selected": {"features": all_features,"use_selection": True}}

    return experiments

def convert_config_value(value):
    if value == "none":
        return None

    if isinstance(value, list):
        return [None if item == "none" else item for item in value]

    return value

def get_model_parameters(model_settings):
    fixed_params = {}
    search_params = {}

    for key, value in model_settings.items():
        if key in {"enabled","selection","search","n_iter"}:
            continue

        value = convert_config_value(value)

        if isinstance(value,list):
            search_params[f"model__{key}"] = value
        else:
            fixed_params[key] = value

    return fixed_params, search_params

def build_selector(model_name, random_state):
    if model_name == "logistic_regression":
        selector_model = LogisticRegression(l1_ratio=1.0,solver="liblinear",max_iter=2000,random_state=random_state)

    elif model_name == "random_forest":
        selector_model = RandomForestClassifier(n_estimators=200,random_state=random_state,n_jobs=-1)

    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        selector_model = XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,random_state=random_state,n_jobs=-1,eval_metric="logloss")

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return SelectFromModel(selector_model)

def build_model(model_name, model_config, random_state, use_selection):
    fixed_params, search_params = get_model_parameters(model_config)
    selection_config = model_config.get("selection",{})

    if model_name == "logistic_regression":
        defaults = {"max_iter": 2000, "random_state": random_state}
        defaults.update(fixed_params)
        
        model = LogisticRegression(**defaults)
        pipeline = [("scaler", StandardScaler())]

    elif model_name == "random_forest":
        defaults = {"random_state": random_state, "n_jobs": -1}
        defaults.update(fixed_params)

        model = RandomForestClassifier(**defaults)
        pipeline = []

    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        defaults = {"random_state": random_state, "n_jobs": -1, "eval_metric": "logloss"}
        defaults.update(fixed_params)

        model = XGBClassifier(**defaults)
        pipeline = []

    else:
        raise ValueError(f"Unknown model: {model_name}")

    if use_selection:
        selector = build_selector(model_name,random_state)

        pipeline.append(("selector", selector))
        thresholds = selection_config.get("thresholds",["median", "mean"])
        search_params["selector__threshold"] = thresholds

    pipeline.append(("model", model))
    pipeline = Pipeline(pipeline)

    return pipeline, search_params

def build_search(pipeline,search_params,model_settings,scoring,refit_metric,cv,random_state):
    search_type = model_settings.get("search","grid")
    if search_type == "random":
        n_iter = model_settings.get("n_iter",20)

        return RandomizedSearchCV(estimator=pipeline,param_distributions=search_params,n_iter=n_iter,scoring=scoring,refit=refit_metric,cv=cv,random_state=random_state,n_jobs=-1)

    if search_type == "grid":
        return GridSearchCV(estimator=pipeline,param_grid=search_params,scoring=scoring,refit=refit_metric,cv=cv,n_jobs=-1)

def evaluate_validation(model, X_val, y_val):
    probabilities = model.predict_proba(X_val)[:, 1]

    # default threshold of 0.5 for binary classification
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_val,predictions,labels=[0, 1]).ravel()
    specificity = (tn / (tn + fp) if (tn + fp) > 0 else 0)

    metrics = {"val_average_precision": average_precision_score(y_val, probabilities),
        "val_roc_auc": roc_auc_score(y_val, probabilities),
        "val_precision_0.5": precision_score(y_val,predictions,zero_division=0),
        "val_recall_0.5": recall_score(y_val,predictions,zero_division=0),
        "val_specificity_0.5": specificity,
        "val_f1_0.5": f1_score(y_val,predictions, zero_division=0),
        "val_balanced_accuracy_0.5": balanced_accuracy_score(y_val,predictions),
        "val_brier_score":brier_score_loss(y_val,probabilities)
        }

    return metrics, probabilities, predictions

def get_selected_features(model, feature_columns):
    if "selector" not in model.named_steps:
        return feature_columns

    selector = model.named_steps["selector"]
    selected_features = [feature for feature, selected in zip(feature_columns,selector.get_support()) if selected]

    return selected_features

def train_experiment(model_name,model_settings,experiment_name,experiment_config,train_df,val_df,model_config):
    feature_columns = experiment_config["features"]
    use_selection = experiment_config["use_selection"]

    training_config = model_config["training"]
    split_config = model_config["split"]

    random_state = split_config["random_state"]
    group_column = split_config["group_column"]

    cv_folds = training_config["cv_folds"]
    refit_metric = training_config["refit_metric"]
    metric_names = training_config["metrics"]

    missing_features = (set(feature_columns) - set(train_df.columns))
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")
    
    X_train = train_df[feature_columns]
    y_train = train_df["label"]
    groups = train_df[group_column]

    X_val = val_df[feature_columns]
    y_val = val_df["label"]

    if X_train.isna().any().any():
        raise ValueError(f"Missing values found in training features for {experiment_name}.")

    if X_val.isna().any().any():
        raise ValueError(f"Missing values found in validation features for {experiment_name}.")

    pipeline, param_grid = build_model(model_name,model_settings,random_state,use_selection)
    cv = StratifiedGroupKFold(n_splits=cv_folds,shuffle=True,random_state=random_state)
    scoring = {metric: metric for metric in metric_names}

    search = build_search(pipeline=pipeline,search_params=param_grid,model_settings=model_settings,scoring=scoring,refit_metric=refit_metric,cv=cv,random_state=random_state)

    print(f"\nTraining {model_name} | {experiment_name} | {len(X_train)} training samples | {len(X_val)} validation samples")

    search.fit(X_train,y_train,groups=groups)
    best_model = search.best_estimator_
    
    metrics, probabilities, predictions = (evaluate_validation(best_model,X_val,y_val))
    selected_features = get_selected_features(best_model,feature_columns)
    best_index = search.best_index_
    cv_std = search.cv_results_[
        f"std_test_{refit_metric}"
    ][best_index]

    result = {"model": model_name,"experiment": experiment_name,"features": len(feature_columns), "selected_features": len(selected_features),
        "cv_best_score": search.best_score_,"cv_score_std": cv_std,"best_params": json.dumps(search.best_params_,default=str)}

    result.update(metrics)

    validation_predictions = pd.DataFrame({"label": y_val.to_numpy(),"probability": probabilities,"prediction_0.5": predictions})

    print(f"Best CV {refit_metric}: {search.best_score_:.3f} ± {cv_std:.3f}")
    print(f"Selected features: {len(selected_features)} / {len(feature_columns)}")
    print(f"Validation AP: {metrics['val_average_precision']:.3f}")

    return {"result": result,"model": best_model,"predictions": validation_predictions, "input_features": feature_columns, "selected_features": selected_features}

def save_best_model(model_name,best_run,models_path,tables_path):
    model_path = (models_path / model_name)
    table_path = (tables_path / model_name)
    ensure_dir(model_path)
    ensure_dir(table_path)

    joblib.dump(best_run["model"],model_path / "best_model.joblib")

    best_run["predictions"].to_csv(table_path / "validation_predictions.csv",index=False)

    pd.DataFrame({"feature": best_run["input_features"]}).to_csv(table_path / "input_features.csv",index=False)
    pd.DataFrame({"feature": best_run["selected_features"]}).to_csv(table_path / "selected_features.csv",index=False)

def main():
    data_config = load_config()
    model_config = load_config("model_config.toml")

    df = load_training_data(data_config)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    experiments = get_experiments(data_config)
    enabled_experiments = model_config["training"].get("experiments",list(experiments.keys()))

    models_path = (PROJECT_ROOT / data_config["paths"]["models"])
    tables_path = (PROJECT_ROOT / data_config["paths"]["tables"])
    figures_path = (PROJECT_ROOT / data_config["paths"]["model_figures"])
    # experiment_tables = (tables_path / "experiments")
    validation_figures = (figures_path / "validation")

    ensure_dir(models_path)
    ensure_dir(tables_path)
    ensure_dir(figures_path)
    # ensure_dir(experiment_tables)
    ensure_dir(validation_figures)

    results = []
    best_results = []
    best_predictions = {}

    for model_name, model_settings in (model_config["models"].items()):
        if not model_settings.get("enabled", False):
            print(f"\nSkipping {model_name}")
            continue
        model_runs = []

        for experiment_name in enabled_experiments:
            if experiment_name not in experiments:
                raise ValueError(f"Unknown experiment: {experiment_name}")

            run = train_experiment(model_name=model_name,model_settings=model_settings,experiment_name=experiment_name,experiment_config=experiments[experiment_name],
                train_df=train_df,val_df=val_df,model_config=model_config)

            model_runs.append(run)
            results.append(run["result"])

        best_run = max(model_runs,key=lambda x: x["result"]["cv_best_score"])
        best_result = best_run["result"].copy()
        best_result["best_for_model"] = True
        best_results.append(best_result)

        save_best_model(model_name,best_run,models_path,tables_path)
        best_predictions[model_name] = best_run["predictions"][["label", "probability"]]

        print(f"Best {model_name}: {best_result['experiment']} | CV AP = {best_result['cv_best_score']:.3f} | Validation AP = {best_result['val_average_precision']:.3f}")
    
    if not results:
        raise ValueError("No training experiments were run.")
    results_df = pd.DataFrame(results)
    best_results_df = pd.DataFrame(best_results)
    results_df.to_csv(tables_path / "experiments_results.csv",index=False)
    best_results_df.to_csv(tables_path / "best_models_summary.csv",index=False)

    plot_validation_pr_curves(best_predictions,validation_figures)
    plot_validation_roc_curves(best_predictions,validation_figures)
    plot_experiment_comparison(results_df,validation_figures)

if __name__ == "__main__":
    main()