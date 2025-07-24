# Multimodal Satellite Data and Extreme Gradient Boosting for Ecological Forest Site Classification

This repository contains the complete pipeline used in our paper. Our workflow combines Google Earth Engine (GEE) for large-scale data preparation, and machine learning models based on XGBoost to classify forest categories.

---

## Repository Structure

---

## Description of Scripts

| Script | Purpose |
|-------|--------|
| **xgboost.py** | Train a single XGBoost classifier on prepared raster data. Saves trained model, metrics, and plots. |
| **cross_valid_xgb.py** | Train and evaluate XGBoost with k-fold cross-validation (default: StratifiedKFold). Saves fold-wise metrics, ROC curves, confusion matrices, and trained models. |
| **gee_export.js** | Google Earth Engine script to export satellite data covering the study area, including multiple spectral bands or indices for machine learning. |
| *merge.py** | Merge the training shapefile with the raster. Adds last band as target class. |

---

##  Data

- `final_roi_data.zip`  
  Contains a shapefile defining the study area / region of interest divided into specific ecological forest types.
- `forest_categories.csv`  
  A CSV file listing the forest categories used in the paper. Please note there is no official translation so we used Google Translate API.
- `band_descriptions.py`  
  Contains a python dictionary with a band index (starts from 1 bacause of Rasterio) and its short description. Those bands returns the `gee_export.js`.

> Note: Actual satellite raster data (e.g., `dataset.tif`) must be prepared and exported separately using the provided GEE script.

---
