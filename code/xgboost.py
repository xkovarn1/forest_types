# XGBoost training pipeline with per-class weighting (GPU)
import os
import json
import rasterio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import xgboost as xgb
from collections import Counter

# --- CONFIG ---
TIF_PATH = 'dataset.tif'
TARGET_BAND_INDEX = 157      # 1-based
BACKGROUND_CLASS = 0
TEST_SIZE = 0.2
RANDOM_STATE = 42
OUTPUT_DIR = 'results_xgb_pipeline_weighted'
USE_PCA = False
VARIANCE_THRESHOLD = 0.95
HANDLE_NAN = False
NAN_INPUTER = 'median'

CLASS_MERGE_MAP = {}
REMOVE_CLASSES = []

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(OUTPUT_DIR, 'training.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
print("Logging to:", os.path.join(OUTPUT_DIR, 'training.log'))

def merge_classes(y_array, merge_map):
    y_new = y_array.copy()
    for old_class, new_class in merge_map.items():
        y_new[y_new == old_class] = new_class
    return y_new

# --- Load raster ---
logging.info("Loading raster data...")
with rasterio.open(TIF_PATH) as src:
    data = src.read()
    band_count = src.count
logging.info(f"Raster bands: {band_count}, shape: {data.shape}")
assert TARGET_BAND_INDEX <= band_count, "Target band index too large"

target = data[TARGET_BAND_INDEX - 1]
feature_indices = [i for i in range(band_count) if i != (TARGET_BAND_INDEX - 1)]
features = data[feature_indices, :, :]

mask = target != BACKGROUND_CLASS
X = features[:, mask].T
y = target[mask]

y_merged = merge_classes(y, CLASS_MERGE_MAP)

if REMOVE_CLASSES:
    keep_mask = ~np.isin(y_merged, REMOVE_CLASSES)
    X = X[keep_mask]
    y_merged = y_merged[keep_mask]

vt = VarianceThreshold(threshold=0.0)
X = vt.fit_transform(X)
logging.info(f"Features after zero-variance removal: {X.shape[1]}")

if HANDLE_NAN:
    from sklearn.impute import SimpleImputer
    if np.isnan(X).any():
        logging.warning("Data contains NaNs, applying imputer.")
        imputer = SimpleImputer(strategy=NAN_INPUTER)
        X = imputer.fit_transform(X)
else:
    missing_rows = np.isnan(X).any(axis=1)
    if missing_rows.any():
        X = X[~missing_rows]
        y_merged = y_merged[~missing_rows]
        logging.warning(f"Dropped rows with NaNs: {missing_rows.sum()}")

unique_classes = sorted(np.unique(y_merged))
class_to_new = {orig: idx for idx, orig in enumerate(unique_classes)}
new_to_class = {v: int(k) for k,v in class_to_new.items()}
y_remapped = np.array([class_to_new[yy] for yy in y_merged])

class_counts = Counter(y_remapped)
total = sum(class_counts.values())
num_classes = len(class_counts)
class_weight = {cls: total/(num_classes*count) for cls, count in class_counts.items()}
sample_weights = np.array([class_weight[yy] for yy in y_remapped])

logging.info(f"Class mapping: {class_to_new}")
logging.info(f"Class counts: {class_counts}")

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y_remapped, sample_weights, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_remapped
)

if USE_PCA:
    pca = PCA(n_components=VARIANCE_THRESHOLD, random_state=RANDOM_STATE)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    logging.info(f"PCA kept {pca.n_components_} components.")

dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
dtest = xgb.DMatrix(X_test, label=y_test, weight=w_test)

params = {
    'objective': 'multi:softprob',  # changed to softprob to get probabilities
    'num_class': num_classes,
    'tree_method': 'gpu_hist',
    'max_depth': 12,
    'eta': 0.1,
    'seed': RANDOM_STATE,
    'eval_metric': 'mlogloss'
}
logging.info(f"XGBoost params: {params}")

evals = [(dtrain, 'train'), (dtest, 'eval')]
bst = xgb.train(params, dtrain, num_boost_round=400, evals=evals, early_stopping_rounds=20)

# Predict probabilities (shape: [n_samples, num_classes])
y_proba = bst.predict(dtest)
# Predicted classes
y_pred = np.argmax(y_proba, axis=1)

# Save predictions for later analysis
np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
np.save(os.path.join(OUTPUT_DIR, 'y_pred.npy'), y_pred)
np.save(os.path.join(OUTPUT_DIR, 'y_proba.npy'), y_proba)

# Classification report
report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(report).transpose().to_csv(os.path.join(OUTPUT_DIR, 'classification_report.csv'))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm).to_csv(os.path.join(OUTPUT_DIR, 'confusion_matrix.csv'))

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
plt.close()

# Feature importance
importance = bst.get_score(importance_type='gain')
feat_names = list(importance.keys())
feat_values = list(importance.values())
feat_imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': feat_values})
feat_imp_df.sort_values(by='Importance', ascending=False, inplace=True)
feat_imp_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importances.csv'), index=False)

plt.figure(figsize=(12,5))
sns.barplot(x='Feature', y='Importance', data=feat_imp_df)
plt.xticks(rotation=90)
plt.title('Feature Importances')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importances.png'))
plt.close()

# --- ROC AUC per class & macro-average ---
# One-hot encode true labels for ROC AUC computation
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test, classes=list(range(num_classes)))

roc_auc_dict = {}
plt.figure(figsize=(12, 10))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    roc_auc_dict[i] = roc_auc
    plt.plot(fpr, tpr, lw=2, label=f'Class {new_to_class[i]} (AUC = {roc_auc:.3f})')

# Macro-average AUC
macro_auc = roc_auc_score(y_test_bin, y_proba, average='macro')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Multi-class ROC Curves (Macro AUC = {macro_auc:.3f})')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'))
plt.close()

# Save ROC AUC scores
with open(os.path.join(OUTPUT_DIR, 'roc_auc_scores.json'), 'w') as f:
    json.dump({'per_class': {str(new_to_class[k]): float(v) for k,v in roc_auc_dict.items()},
               'macro_auc': float(macro_auc)}, f)

# Save model & mappings & parameters
bst.save_model(os.path.join(OUTPUT_DIR, 'model.xgb'))
with open(os.path.join(OUTPUT_DIR, 'class_mapping.json'), 'w') as f:
    json.dump({
        'class_to_new': {str(k): int(v) for k,v in class_to_new.items()},
        'new_to_class': {str(k): int(v) for k,v in new_to_class.items()}
    }, f)
with open(os.path.join(OUTPUT_DIR, 'xgb_params.json'), 'w') as f:
    json.dump(params, f)

logging.info("Pipeline done. Results in: %s", OUTPUT_DIR)
print("Done! See:", OUTPUT_DIR)
