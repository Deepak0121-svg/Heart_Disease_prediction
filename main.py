# %%
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)
import joblib

# Paths and constants
CSV_PATH = './heart_disease_uci.csv'  # Change to your dataset
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
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Print classification report
    print('---', name, '---')
    print(f'Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}')
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{name} - Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    save_fig(fig, f'{name}_confusion_matrix.png')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--')
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
# Drop unnecessary columns
for col in ['id', 'dataset']:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# Convert target
if 'num' in df.columns:
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df.drop(columns=['num'], inplace=True)
else:
    raise ValueError('Expected `num` column as target indicator')

print(df.info())

# %%
# Handle missing values
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

# Encode categorical
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
# Train-test split
X = df[features]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Standardize numeric columns
scaler = StandardScaler()
if num_cols:
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))

# %%
# Train Random Forest ONLY
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

# Evaluate Random Forest
results = []
res = evaluate_model("RandomForest", rf, X_test, y_test)
results.append(res)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_results.csv'), index=False)
print('\nModel results:')
print(results_df)

# Save feature importances
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax)
ax.set_title('RandomForest - Feature Importances')
save_fig(fig, 'RandomForest_feature_importances.png')
plt.close(fig)
print("✅ Saved feature importances for RandomForest")

# Save model
model_path = os.path.join(OUTPUT_DIR, "RandomForest_model.joblib")
joblib.dump(rf, model_path)
print(f"Saved model: {model_path}")

# Save predictions
preds = rf.predict(X_test)
probs = rf.predict_proba(X_test)[:, 1]
out = X_test.copy()
out['actual'] = y_test.values
out['pred'] = preds
out['proba'] = probs
out.to_csv(os.path.join(OUTPUT_DIR, 'test_predictions.csv'), index=False)
print('\n✅ Test predictions saved.')

print('\n🎉 All done. Check the outputs folder for:')
print('- model_results.csv')
print('- RandomForest_model.joblib')
print('- test_predictions.csv')
print('- feature importance plots')
print('- confusion matrix & ROC curve')
