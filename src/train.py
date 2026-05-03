import time
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from src.models import get_all_models, save_model

RANDOM_STATE = 42


def train_model(model_info, X_train, y_train, X_test, y_test, tune_hyperparams=False):
    name = model_info["name"]
    model = model_info["model"]
    is_unsupervised = model_info["unsupervised"]

    print(f"\n{'=' * 50}")
    print(f"Training: {name}")
    print(f"{'=' * 50}")

    start_time = time.time()

    if is_unsupervised:
        model.fit(X_train)
        train_pred_raw = model.predict(X_train)
        test_pred_raw = model.predict(X_test)

        train_pred = np.where(train_pred_raw == -1, 1, 0)
        test_pred = np.where(test_pred_raw == -1, 1, 0)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
    else:
        if tune_hyperparams and model_info["param_dist"]:
            print(f"  Performing hyperparameter tuning (RandomizedSearchCV 3-fold, 5 iterations)...")
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=model_info["param_dist"],
                n_iter=5,
                cv=cv,
                scoring="accuracy",
                n_jobs=1,
                verbose=0,
                random_state=RANDOM_STATE
            )
            random_search.fit(X_train, y_train)

            model = random_search.best_estimator_
            print(f"  Best params: {random_search.best_params_}")
            print(f"  Best CV accuracy: {random_search.best_score_:.4f}")
        else:
            model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")

    elapsed = time.time() - start_time
    print(f"  Training time: {elapsed:.2f}s")

    save_model(model, name)

    result = {
        "model_name": name,
        "model": model,
        "train_predictions": train_pred,
        "test_predictions": test_pred,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "training_time": elapsed,
        "unsupervised": is_unsupervised
    }

    if not is_unsupervised:
        if hasattr(model, "predict_proba"):
            try:
                train_proba = model.predict_proba(X_train)[:, 1]
                test_proba = model.predict_proba(X_test)[:, 1]
                result["train_proba"] = train_proba
                result["test_proba"] = test_proba
            except Exception:
                result["train_proba"] = None
                result["test_proba"] = None
        else:
            result["train_proba"] = None
            result["test_proba"] = None
    else:
        result["train_proba"] = None
        result["test_proba"] = None

    return result


def train_all_models(data_splits, tune_hyperparams=False):
    print("\n" + "=" * 60)
    print("MODEL TRAINING PIPELINE")
    print("=" * 60)

    X_train = data_splits["X_train_pca"]
    X_test = data_splits["X_test_pca"]
    y_train = data_splits["y_train_res"]
    y_test = data_splits["y_test_bin"]

    X_train_raw = data_splits["X_train_raw"]
    X_test_raw = data_splits["X_test_raw"]
    y_train_raw = data_splits["y_train_bin"]

    models = get_all_models()
    results = []

    for model_info in models:
        if model_info["unsupervised"]:
            result = train_model(
                model_info,
                X_train_raw, y_train_raw,
                X_test_raw, y_test,
                tune_hyperparams=False
            )
        else:
            result = train_model(
                model_info,
                X_train, y_train,
                X_test, y_test,
                tune_hyperparams=tune_hyperparams
            )
        results.append(result)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - ALL MODELS")
    print("=" * 60)

    return results
