# Project 2: Detroit Computer Vision for Building Habitability

## Project Description

Computer vision tools for building habitability using Detroit imagery spanning 1999-2024. This project aims to develop tools to assess building conditions and habitability using visual analysis of Detroit's built environment over time.

**Current Implementation**: Multi-class blight classification using Detroit Land Bank Authority survey data as a foundation for future computer vision work.

**Data Source:** DLBA survey data (20250527_DLBA_survey_data_UM_Detroit.xlsx)
**Problem Type:** Multi-class classification (4 classes)
**Features:** Property condition indicators (roof, openings, occupancy, etc.)
**Target:** Blight severity levels derived from FIELD_DETERMINATION

## Models

### `xgboost_baseline.py`
Baseline XGBoost classifier for 4-class blight prediction with comprehensive evaluation metrics and feature importance analysis.

### `xgboost_optimized1.py`  
Advanced XGBoost model with Bayesian hyperparameter optimization (Optuna), feature engineering, and K-fold cross-validation.

## Quick Start

**Option 1: Use main environment (covers all projects)**
```bash
# From repository root
pip install -r requirements.txt  # or: conda env create -f environment.yml
cd 2_detroit_computer_vision/models/
python xgboost_baseline.py
```

**Option 2: Use model-specific requirements (minimal install)**
```bash
cd 2_detroit_computer_vision/models/
pip install -r requirements-xgboost_baseline.txt
python xgboost_baseline.py

# For optimized model
pip install -r requirements-xgboost_optimized1.txt  
python xgboost_optimized1.py
```

**Outputs**: All results saved to `deliverables/[model_name]/` with visualizations and metrics.

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

## Potential Future Computer Vision Components

**Planned Features**:
- Aerial imagery analysis for building condition assessment
- Street-level imagery processing for habitability metrics
- Temporal analysis of building deterioration (1999-2024)
- Integration of survey data with visual analysis

**Potential Tech Stack**: OpenCV, PyTorch/TensorFlow, satellite/aerial imagery APIs, computer vision models