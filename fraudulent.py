import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    average_precision_score, f1_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
SMOTE_RATIO = 1.0
DATA_PATH = "data/creditcard.csv"
COLS_TO_SCALE = ['Amount', 'Time']

np.random.seed(RANDOM_STATE)

def load_data(path):
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"\nClass Distribution:\n{df['Class'].value_counts()}")
    print(f"\nFraud Rate: {df['Class'].mean() * 100:.3f}%")
    return df

def plot_class_distribution(df):
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=df['Class'], palette="Blues_r")
    ax.bar_label(ax.containers[0])
    plt.title("Fraud vs Non-Fraud Distribution")
    plt.xlabel("Class (0=Legitimate, 1=Fraud)")
    plt.tight_layout()
    plt.show()

def prepare_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[COLS_TO_SCALE] = scaler.fit_transform(X_train[COLS_TO_SCALE])
    X_test_scaled[COLS_TO_SCALE] = scaler.transform(X_test[COLS_TO_SCALE])

    return X_train_scaled, X_test_scaled, y_train, y_test

def create_smote_pipeline(model, model_name):
    return ImbPipeline([
        ('smote', SMOTE(sampling_strategy=SMOTE_RATIO, random_state=RANDOM_STATE)),
        (model_name, model)
    ])

def tune_random_forest(X_train, y_train, cv):
    pipeline = create_smote_pipeline(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), 'rf'
    )
    params = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [10, 20, 30, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__class_weight': ['balanced', 'balanced_subsample', None]
    }

    search = RandomizedSearchCV(
        pipeline, params, n_iter=20, cv=cv,
        scoring='f1', random_state=RANDOM_STATE, n_jobs=-1
    )
    search.fit(X_train, y_train)

    print(f"Best RF params: {search.best_params_}")
    print(f"Best RF F1: {search.best_score_:.4f}")
    return search.best_estimator_

def tune_xgboost(X_train, y_train, cv):
    pipeline = create_smote_pipeline(
        XGBClassifier(
            eval_metric='logloss', random_state=RANDOM_STATE,
            tree_method='gpu_hist', device='cuda'
        ), 'xgb'
    )
    params = {
        'xgb__n_estimators': [100, 200, 300],
        'xgb__max_depth': [3, 5, 7, 10],
        'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'xgb__subsample': [0.6, 0.8, 1.0],
        'xgb__colsample_bytree': [0.6, 0.8, 1.0],
        'xgb__scale_pos_weight': [1, 5, 10]
    }

    search = RandomizedSearchCV(
        pipeline, params, n_iter=20, cv=cv,
        scoring='f1', random_state=RANDOM_STATE, n_jobs=1
    )
    search.fit(X_train, y_train)

    print(f"Best XGB params: {search.best_params_}")
    print(f"Best XGB F1: {search.best_score_:.4f}")
    return search.best_estimator_

def train_logistic_regression(X_train, y_train):
    pipeline = create_smote_pipeline(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), 'lr'
    )
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_models(models, X_test, y_test):
    results = {}

    for name, pipeline in models.items():
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)

        results[name] = {
            'y_prob': y_prob,
            'y_pred': y_pred,
            'roc_auc': roc_auc_score(y_test, y_prob),
            'pr_auc': average_precision_score(y_test, y_prob)
        }

        print(f"\n{'='*50}")
        print(f"{name.upper()}")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {results[name]['roc_auc']:.4f}")
        print(f"PR-AUC: {results[name]['pr_auc']:.4f}")

    return results

def optimize_thresholds(results, y_test):
    thresholds = np.arange(0.1, 0.9, 0.01)

    for name, data in results.items():
        scores = [f1_score(y_test, (data['y_prob'] >= t).astype(int)) for t in thresholds]
        optimal_idx = np.argmax(scores)
        results[name]['optimal_thresh'] = thresholds[optimal_idx]
        results[name]['optimal_f1'] = scores[optimal_idx]
        print(f"{name}: Optimal Threshold = {thresholds[optimal_idx]:.2f}, F1 = {scores[optimal_idx]:.4f}")

    return results, thresholds

def plot_threshold_curves(results, y_test, thresholds):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (name, data) in zip(axes, results.items()):
        scores = [f1_score(y_test, (data['y_prob'] >= t).astype(int)) for t in thresholds]
        ax.plot(thresholds, scores, 'b-', linewidth=2)
        ax.axvline(x=data['optimal_thresh'], color='r', linestyle='--',
                   label=f"Optimal: {data['optimal_thresh']:.2f}")
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1 Score')
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_feature_importance(rf_pipeline, xgb_pipeline, feature_names):
    rf_model = rf_pipeline.named_steps['rf']
    xgb_model = xgb_pipeline.named_steps['xgb']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    models = [(rf_model, 'Random Forest', 'steelblue'), (xgb_model, 'XGBoost', 'darkorange')]

    for ax, (model, name, color) in zip(axes, models):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True).tail(15)

        ax.barh(importance['feature'], importance['importance'], color=color)
        ax.set_xlabel('Importance')
        ax.set_title(f'{name} - Top 15 Features')

    plt.tight_layout()
    plt.show()

def plot_roc_pr_curves(results, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, data in results.items():
        RocCurveDisplay.from_predictions(
            y_test, data['y_prob'], name=f"{name} ({data['roc_auc']:.3f})", ax=axes[0]
        )
        PrecisionRecallDisplay.from_predictions(
            y_test, data['y_prob'], name=f"{name} ({data['pr_auc']:.3f})", ax=axes[1]
        )

    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_title('ROC Curves')
    axes[0].legend(loc='lower right')
    axes[1].set_title('Precision-Recall Curves')
    axes[1].legend(loc='lower left')

    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (name, data) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, data['y_pred'], normalize='true')
        ConfusionMatrixDisplay(cm, display_labels=['Legitimate', 'Fraud']).plot(
            ax=ax, cmap='Blues', values_format='.2%'
        )
        ax.set_title(name)

    plt.tight_layout()
    plt.show()

def print_summary(results):
    summary = pd.DataFrame({
        'Model': list(results.keys()),
        'ROC-AUC': [r['roc_auc'] for r in results.values()],
        'PR-AUC': [r['pr_auc'] for r in results.values()],
        'Optimal Threshold': [r['optimal_thresh'] for r in results.values()],
        'F1 at Optimal': [r['optimal_f1'] for r in results.values()]
    }).sort_values('PR-AUC', ascending=False)

    print(summary.to_string(index=False))
    return summary

def main():
    df = load_data(DATA_PATH)
    plot_class_distribution(df)

    X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(df)
    feature_names = X_train_scaled.columns.tolist()

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    models = {
        'Logistic Regression': train_logistic_regression(X_train_scaled, y_train),
        'Random Forest': tune_random_forest(X_train_scaled, y_train, cv),
        'XGBoost': tune_xgboost(X_train_scaled, y_train, cv)
    }

    results = evaluate_models(models, X_test_scaled, y_test)
    results, thresholds = optimize_thresholds(results, y_test)

    plot_threshold_curves(results, y_test, thresholds)
    plot_feature_importance(models['Random Forest'], models['XGBoost'], feature_names)
    plot_roc_pr_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    print_summary(results)


if __name__ == "__main__":
    main()
