# xgboost_forest_pipeline_cv.py
import os
import json
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
import xgboost as xgb

# -------------------------------
# CONFIGURATION
# -------------------------------
TIF_PATH = 'dataset.tif'
TARGET_BAND_INDEX = 157      # 1-based index of target band
BACKGROUND_CLASS = 0
N_SPLITS = 5                 # Number of CV folds
RANDOM_STATE = 42
OUTPUT_DIR = 'results_xgb_cv'
USE_PCA = False
VARIANCE_THRESHOLD = 0.95
HANDLE_NAN = False
NAN_INPUTER = 'median'
CLASS_MERGE_MAP = {}
REMOVE_CLASSES = []

# -------------------------------
# INITIALIZATION
# -------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(OUTPUT_DIR, 'training.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
print("Logging to:", os.path.join(OUTPUT_DIR, 'training.log'))

# -------------------------------
# FUNCTIONS
# -------------------------------
def merge_classes(y_array, merge_map):
    y_new = y_array.copy()
    for old_class, new_class in merge_map.items():
        y_new[y_new == old_class] = new_class
    return y_new

def compute_sample_weights(y, num_classes):
    class_counts = Counter(y)
    total = sum(class_counts.values())
    return {cls: total/(num_classes*count) for cls, count in class_counts.items()}

def plot_confusion_matrix(cm, classes, fold_idx):
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Fold {fold_idx+1}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_fold_{fold_idx+1}.png'))
    plt.close()

def plot_roc_curves(y_test_bin, y_proba, classes, fold_idx):
    plt.figure(figsize=(12, 10))
    roc_auc_dict = {}
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        roc_auc_dict[str(cls)] = roc_auc
        plt.plot(fpr, tpr, lw=2, label=f'Class {cls} (AUC={roc_auc:.3f})')
    macro_auc = roc_auc_score(y_test_bin, y_proba, average='macro')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multi-class ROC Curves - Fold {fold_idx+1} (Macro AUC={macro_auc:.3f})')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'roc_curves_fold_{fold_idx+1}.png'))
    plt.close()
    return roc_auc_dict, macro_auc

# -------------------------------
# LOAD AND PREPARE DATA
# -------------------------------
logging.info("Loading raster data...")
with rasterio.open(TIF_PATH) as src:
    data = src.read()
band_count = data.shape[0]
assert TARGET_BAND_INDEX <= band_count, "Target band index too large"

target = data[TARGET_BAND_INDEX - 1]
features = np.delete(data, TARGET_BAND_INDEX - 1, axis=0)
mask = target != BACKGROUND_CLASS

X = features[:, mask].T
y = target[mask]
logging.info(f"Initial data shape: {X.shape}")

y_merged = merge_classes(y, CLASS_MERGE_MAP)

if REMOVE_CLASSES:
    keep_mask = ~np.isin(y_merged, REMOVE_CLASSES)
    X = X[keep_mask]
    y_merged = y_merged[keep_mask]

# Remove low variance features
vt = VarianceThreshold(threshold=0.0)
X = vt.fit_transform(X)
logging.info(f"Features after variance thresholding: {X.shape}")

# Handle missing values
if HANDLE_NAN:
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy=NAN_INPUTER)
    X = imputer.fit_transform(X)
else:
    missing_rows = np.isnan(X).any(axis=1)
    if missing_rows.any():
        logging.warning(f"Dropping rows with NaNs: {missing_rows.sum()}")
        X = X[~missing_rows]
        y_merged = y_merged[~missing_rows]

# Remap classes to sequential IDs
unique_classes = sorted(np.unique(y_merged))
class_to_new = {int(orig): int(idx) for idx, orig in enumerate(unique_classes)}
new_to_class = {int(v): int(k) for k, v in class_to_new.items()}
y_remapped = np.array([class_to_new[yy] for yy in y_merged])
num_classes = len(unique_classes)

logging.info(f"Class mapping: {class_to_new}")

# -------------------------------
# CROSS-VALIDATION
# -------------------------------
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_metrics = []

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y_remapped)):
    logging.info(f"Starting fold {fold_idx+1}/{N_SPLITS}")
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_remapped[train_idx], y_remapped[test_idx]

    if USE_PCA:
        pca = PCA(n_components=VARIANCE_THRESHOLD, random_state=RANDOM_STATE)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        logging.info(f"PCA kept {pca.n_components_} components.")

    # Compute sample weights
    sample_weights = compute_sample_weights(y_train, num_classes)
    w_train = np.array([sample_weights[yy] for yy in y_train])

    # XGBoost training
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'tree_method': 'hist',
        'max_depth': 12,
        'eta': 0.1,
        'seed': RANDOM_STATE,
        'eval_metric': 'mlogloss'
    }
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round=400, evals=evals, early_stopping_rounds=20)

    # Predictions
    y_proba = bst.predict(dtest)
    y_pred = np.argmax(y_proba, axis=1)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(OUTPUT_DIR, f'classification_report_fold_{fold_idx+1}.csv')
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, [new_to_class[i] for i in range(num_classes)], fold_idx)

    # ROC curves
    y_test_bin = label_binarize(y_test, classes=list(range(num_classes)))
    roc_auc_dict, macro_auc = plot_roc_curves(y_test_bin, y_proba,
                                              [new_to_class[i] for i in range(num_classes)], fold_idx)

    # Save fold metrics
    fold_metrics.append({
        'fold': fold_idx+1,
        'macro_auc': macro_auc,
        'roc_auc_per_class': roc_auc_dict
    })

    # Save model
    bst.save_model(os.path.join(OUTPUT_DIR, f'model_fold_{fold_idx+1}.xgb'))

# -------------------------------
# SAVE OVERALL RESULTS
# -------------------------------
with open(os.path.join(OUTPUT_DIR, 'fold_metrics.json'), 'w') as f:
    json.dump(fold_metrics, f, indent=2)

with open(os.path.join(OUTPUT_DIR, 'class_mapping.json'), 'w') as f:
    json.dump({'class_to_new': class_to_new, 'new_to_class': new_to_class}, f, indent=2)

with open(os.path.join(OUTPUT_DIR, 'xgb_params.json'), 'w') as f:
    json.dump(params, f, indent=2)

logging.info("All folds completed successfully. Results saved.")
print(f"Done! See results in: {OUTPUT_DIR}")
