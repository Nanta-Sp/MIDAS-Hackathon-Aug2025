# Detroit Computer Vision - Blight Classification Models

This directory contains machine learning models for predicting blight severity levels using Detroit property survey data from the Detroit Land Bank Authority (DLBA).

## Project Context

The models work with blight survey data containing property condition assessments across Detroit. The task is multi-class classification to predict blight severity levels (0-3) based on property condition indicators.

**Data Source:** DLBA survey data (20250527_DLBA_survey_data_UM_Detroit.xlsx)
**Problem Type:** Multi-class classification (4 classes)
**Features:** Property condition indicators (roof, openings, occupancy, etc.)
**Target:** Blight severity levels derived from FIELD_DETERMINATION

## Models

### `xgboost_baseline.py`
A baseline XGBoost classifier for 4-class blight prediction.

**Features:**
- Multi-class classification (0=No Blight, 1=Noticeable, 2=Significant, 3=Extreme)
- Feature encoding and preprocessing
- Comprehensive evaluation metrics
- Feature importance analysis
- Confusion matrix visualization
- Predictions with PARCEL_ID tracking

**Key Metrics:**
- Uses accuracy, balanced accuracy, macro/weighted F1, Cohen's kappa, MCC
- Train/validation/test split (70/10/20)
- Per-class performance analysis

### `xgboost_optimized1.py`  
An advanced XGBoost model with hyperparameter optimization and feature engineering.

**Additional Features:**
- Bayesian hyperparameter optimization (Optuna)
- Advanced feature engineering (interaction terms)
- Feature selection (mutual information)
- Stratified K-fold cross-validation
- Model persistence and comprehensive logging
- Enhanced visualizations

**Optimization:**
- 10+ hyperparameters tuned automatically
- Tree-structured Parzen Estimator (TPE) sampling
- Early stopping and regularization
- Class imbalance handling

## Directory Structure

```
2_detroit_computer_vision/
├── models/                          # Model scripts and requirements
│   ├── xgboost_baseline.py         # Baseline XGBoost model
│   ├── xgboost_optimized1.py       # Optimized XGBoost model
│   ├── requirements-xgboost_baseline.txt
│   └── requirements-xgboost_optimized1.txt
├── training_data/                   # Processed training datasets
│   ├── blight_features.csv         # Feature matrix
│   └── blight_labels.csv           # Target labels
├── deliverables/                    # Model outputs and artifacts
│   ├── xgboost_baseline/           # Baseline model results
│   └── xgboost_optimized1/         # Optimized model results
└── eda/                            # Exploratory data analysis notebooks
```

## Data

**Target Variable:** FIELD_DETERMINATION mapped to 4 classes:
- 0: No Action (Salvage), NAP (Salvage), Other Resolution Pathways, Vacant (Not Blighted)
- 1: Noticeable Evidence of Blight  
- 2: Significant Evidence of Blight
- 3: Extreme Evidence of Blight

**Features:** Property condition indicators:
- `IS_OCCUPIED` - Whether property is occupied
- `FIRE_DAMAGE_CONDITION` - Fire damage assessment
- `ROOF_CONDITION` - Roof condition rating
- `OPENINGS_CONDITION` - Doors/windows condition
- `IS_OPEN_TO_TRESPASS` - Accessibility to trespassers

**Dataset Size:** ~98,320 property records
**Class Distribution:** 49% class 1, 27% class 0, 20% class 2, 4% class 3

## Usage

### Run Baseline Model
```bash
cd models/
pip install -r requirements-xgboost_baseline.txt
python xgboost_baseline.py
```

### Run Optimized Model  
```bash
cd models/
pip install -r requirements-xgboost_optimized1.txt
python xgboost_optimized1.py
```

## Outputs

Each model generates comprehensive artifacts in `deliverables/[model_name]/`:

**Data Analysis:**
- `data_analytics.json` - Dataset statistics
- `label_distribution.png` - Class distribution plots
- `data_summary.png` - Summary statistics

**Model Performance:**
- `evaluation_results.json` - All metrics and classification report  
- `confusion_matrix.png` - Prediction accuracy visualization
- `metrics_comparison.png` - Performance metrics comparison

**Feature Analysis:**
- `feature_importance.csv` - Feature importance scores
- `feature_importance.png` - Feature importance visualization

**Predictions:**
- `test_predictions.csv` - Test set predictions with PARCEL_IDs and probabilities

**Additional (Optimized Model):**
- `optimization_results.json` - Hyperparameter optimization results
- `cv_results.json` - Cross-validation results with confidence intervals
- `best_model.pkl` - Serialized trained model

## Requirements

**Core Dependencies:**
- pandas >= 2.0.0
- numpy >= 1.24.0  
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

**Additional (Optimized Model):**
- optuna >= 3.4.0
- joblib >= 1.3.0

## Performance

**Baseline Model Results:**
- Accuracy: ~62.6%
- Balanced Accuracy: ~49.7% 
- Macro F1: ~51.4%
- Weighted F1: ~60.5%

**Key Findings:**
- Model performs well on classes 0 and 1 (F1 ~0.66-0.70)
- Struggles with minority classes 2 and 3 (F1 ~0.35)
- OPENINGS_CONDITION is the most important feature (60% importance)
- ROOF_CONDITION and IS_OCCUPIED are secondary features

The models handle class imbalance through balanced evaluation metrics, stratified sampling, and proper train/validation/test splits.