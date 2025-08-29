# %%
import os
import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
roc_auc_score, confusion_matrix, classification_report, roc_curve)
import joblib


# Optional: XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


CSV_PATH = './heart_disease_uci.csv' # Change to your dataset
OUTPUT_DIR = 'outputs'
RANDOM_STATE = 42
TEST_SIZE = 0.2


os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
def save_fig(fig, name, dpi=150):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    print(f"Saved: {path}")


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

    # Print classification report
    print('---', name, '---')
    print(f'Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}')
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{name} - Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    save_fig(fig, f'{name}_confusion_matrix.png')
    plt.show()

    # ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        ax.plot([0,1], [0,1], linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{name} - ROC Curve')
        ax.legend()
        save_fig(fig, f'{name}_roc.png')
        plt.show()

    return {
        'name': name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc
    }


# %%
print('Loading CSV from', CSV_PATH)
df = pd.read_csv(CSV_PATH)
print('Shape:', df.shape)
df.head()

# %%
for col in ['id', 'dataset']:
    if col in df.columns:
        df.drop(columns=col, inplace=True)


if 'num' in df.columns:
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df.drop(columns=['num'], inplace=True)
else:
    raise ValueError('Expected `num` column as target indicator')


print(df.info())

# %%
print(df.isnull().sum())
thresh = 0.4 * len(df)
cols_to_drop = [c for c in df.columns if df[c].isnull().sum() > thresh]
df.drop(columns=cols_to_drop, inplace=True)
print('Dropped columns:', cols_to_drop)


num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
if 'target' in num_cols:
    num_cols.remove('target')


# Imputation
num_imputer = SimpleImputer(strategy='median')
if num_cols:
    df[num_cols] = num_imputer.fit_transform(df[num_cols])


cat_imputer = SimpleImputer(strategy='most_frequent')
if cat_cols:
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# %%
for c in cat_cols:
    if df[c].nunique() == 2:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
    else:
        if df[c].nunique() <= 6:
            dummies = pd.get_dummies(df[c].astype(str), prefix=c)
            df = pd.concat([df.drop(columns=[c]), dummies], axis=1)
        else:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))


features = [c for c in df.columns if c != 'target']

# %%
corr = df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


sns.countplot(x='target', data=df)
plt.title('Target Distribution')
plt.show()


for c in num_cols:
    sns.boxplot(x='target', y=c, data=df)
    plt.title(f'{c} by Target')
    plt.show()

# %%
X = df[features]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


scaler = StandardScaler()
if num_cols:
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])


joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))

# %%
models = {}


lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train, y_train)
models['LogisticRegression'] = lr


rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
models['RandomForest'] = rf


if XGBOOST_AVAILABLE:
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb
else:
    print('XGBoost not available.')

# %%
# ---------- Model evaluation ----------
results = []
for name, model in models.items():
    res = evaluate_model(name, model, X_test, y_test)
    results.append(res)

# Save results to CSV
results_df = pd.DataFrame(results).sort_values(by='f1', ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_results.csv'), index=False)
print('\nModel results:')
print(results_df)

# 🔹 Save Feature Importances for ALL models that support it
for name, model in models.items():
    if hasattr(model, "feature_importances_"):  # Tree-based models (RF, XGB, etc.)
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False).head(20)
        
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax)
        ax.set_title(f'{name} - Feature Importances')
        save_fig(fig, f'{name}_feature_importances.png')
        plt.close(fig)
        print(f"✅ Saved feature importances for {name}")

# 🔹 Save ALL models
for name, model in models.items():
    model_path = os.path.join(OUTPUT_DIR, f"{name}_model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved model: {model_path}")

# 🔹 Save BEST model separately
best_model_name = results_df.iloc[0]['name']
best_model = models[best_model_name]
joblib.dump(best_model, os.path.join(OUTPUT_DIR, f'best_model_{best_model_name}.joblib'))
print(f"\n✅ Best model saved: {best_model_name}")

# Save test predictions for best model
preds = best_model.predict(X_test)
probs = None
if hasattr(best_model, 'predict_proba'):
    probs = best_model.predict_proba(X_test)[:,1]

out = X_test.copy()
out['actual'] = y_test.values
out['pred'] = preds
if probs is not None:
    out['proba'] = probs
out.to_csv(os.path.join(OUTPUT_DIR, 'test_predictions.csv'), index=False)
print('\n✅ Test predictions saved.')

print('\n🎉 All done. Check the outputs folder for:')
print('- model_results.csv')
print('- ALL trained models (.joblib)')
print(f'- best_model_{best_model_name}.joblib')
print('- test_predictions.csv')
print('- feature importance plots for all models that support it')
print('- confusion matrices & ROC plots')


# %%
# ========= Model Comparison Visualization =========
import matplotlib.pyplot as plt
import seaborn as sns

# Use the results_df you already created
metrics = ["accuracy", "precision", "recall", "f1", "auc"]

plt.figure(figsize=(14,8))

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    sns.barplot(x="name", y=metric, data=results_df, palette="viridis")
    plt.title(f"Model {metric.capitalize()} Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
plt.show()
print("✅ Saved: model_comparison.png")


# %%
import pickle

# Load model
with open("./outputs/best_model_XGBoost.joblib", "rb") as f:
    model = pickle.load(f)
# Get feature importance dictionary
feature_importances = model.get_booster().get_score(importance_type="weight")

# Display features
print("Features used by the model:")
for feat, score in feature_importances.items():
    print(f"{feat}: {score}")




