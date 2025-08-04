import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import (train_test_split, StratifiedKFold, 
                                   learning_curve, validation_curve)
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             f1_score, cohen_kappa_score, matthews_corrcoef,
                             balanced_accuracy_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class XGBoostOptimizer:
    """
    Advanced XGBoost classifier with hyperparameter optimization,
    feature engineering, and comprehensive evaluation.
    """
    
    def __init__(self, output_dir='../deliverables/xgboost_optimized1', random_state=42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.cv_results = None
        self.feature_names = None
        
    def load_data(self, features_path, labels_path):
        """Load and merge feature and label data."""
        print("Loading data...")
        features_df = pd.read_csv(features_path)
        labels_df = pd.read_csv(labels_path)
        
        # Merge on PARCEL_ID
        data = features_df.merge(labels_df, on='PARCEL_ID', how='inner')
        
        # Drop any unnamed columns
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        
        print(f"Loaded {len(data)} samples with {len(data.columns)-2} features")
        return data
    
    def engineer_features(self, df, fit_transform=True):
        """Advanced feature engineering with interaction terms and transformations."""
        print("Engineering features...")
        df_engineered = df.copy()
        
        # Drop DATE_SURVEYED - not predictive
        if 'DATE_SURVEYED' in df_engineered.columns:
            df_engineered = df_engineered.drop('DATE_SURVEYED', axis=1)
        
        # Encode categorical variables
        categorical_cols = ['IS_OCCUPIED', 'FIRE_DAMAGE_CONDITION', 'ROOF_CONDITION', 
                           'OPENINGS_CONDITION', 'IS_OPEN_TO_TRESPASS']
        
        if fit_transform:
            self.label_encoders = {}
        
        for col in categorical_cols:
            if col in df_engineered.columns:
                if fit_transform:
                    le = LabelEncoder()
                    df_engineered[col] = df_engineered[col].fillna('Unknown').astype(str)
                    df_engineered[col] = le.fit_transform(df_engineered[col])
                    self.label_encoders[col] = le
                else:
                    # Use existing encoder
                    df_engineered[col] = df_engineered[col].fillna('Unknown').astype(str)
                    # Handle unseen categories
                    for category in df_engineered[col].unique():
                        if category not in self.label_encoders[col].classes_:
                            df_engineered[col] = df_engineered[col].replace(category, 'Unknown')
                    df_engineered[col] = self.label_encoders[col].transform(df_engineered[col])
        
        # Create interaction features
        if len(df_engineered.columns) >= 2:
            # Most important interactions based on baseline model
            if 'OPENINGS_CONDITION' in df_engineered.columns and 'ROOF_CONDITION' in df_engineered.columns:
                df_engineered['openings_roof_interaction'] = (
                    df_engineered['OPENINGS_CONDITION'] * df_engineered['ROOF_CONDITION']
                )
            
            if 'IS_OCCUPIED' in df_engineered.columns and 'OPENINGS_CONDITION' in df_engineered.columns:
                df_engineered['occupied_openings_interaction'] = (
                    df_engineered['IS_OCCUPIED'] * df_engineered['OPENINGS_CONDITION']
                )
        
        # Feature scaling for interaction terms
        if fit_transform:
            self.scaler = StandardScaler()
            interaction_cols = [col for col in df_engineered.columns if 'interaction' in col]
            if interaction_cols:
                df_engineered[interaction_cols] = self.scaler.fit_transform(df_engineered[interaction_cols])
        else:
            interaction_cols = [col for col in df_engineered.columns if 'interaction' in col]
            if interaction_cols and hasattr(self, 'scaler'):
                df_engineered[interaction_cols] = self.scaler.transform(df_engineered[interaction_cols])
        
        if fit_transform:
            self.feature_names = df_engineered.columns.tolist()
        
        return df_engineered
    
    def select_features(self, X, y, k=15):
        """Feature selection using mutual information."""
        print(f"Selecting top {k} features...")
        
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        self.feature_selector = selector
        self.selected_features = selected_features
        
        print(f"Selected features: {selected_features}")
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def get_class_weights(self, y):
        """Calculate class weights for imbalanced data."""
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, class_weights))
    
    def objective(self, trial, X, y, cv_folds=5):
        """Optuna objective function for hyperparameter optimization."""
        
        # Hyperparameter search space
        params = {
            'objective': 'multi:softprob',
            'num_class': 4,
            'eval_metric': 'mlogloss',
            'random_state': self.random_state,
            'verbosity': 0,
            
            # Tunable parameters
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0)
        }
        
        # Stratified K-fold cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
            
            y_pred = model.predict(X_val_fold)
            f1_macro = f1_score(y_val_fold, y_pred, average='macro')
            cv_scores.append(f1_macro)
        
        return np.mean(cv_scores)
    
    def optimize_hyperparameters(self, X, y, n_trials=100, cv_folds=5):
        """Bayesian optimization of hyperparameters using Optuna."""
        print(f"Optimizing hyperparameters with {n_trials} trials...")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        study.optimize(
            lambda trial: self.objective(trial, X, y, cv_folds),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        self.best_params.update({
            'objective': 'multi:softprob',
            'num_class': 4,
            'eval_metric': 'mlogloss',
            'random_state': self.random_state,
            'verbosity': 0
        })
        
        print(f"Best macro F1-score: {study.best_value:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        # Save optimization results
        with open(self.output_dir / 'optimization_results.json', 'w') as f:
            json.dump({
                'best_value': study.best_value,
                'best_params': self.best_params,
                'n_trials': n_trials,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        return study
    
    def cross_validate(self, X, y, cv_folds=5):
        """Comprehensive cross-validation with detailed metrics."""
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_results = {
            'accuracy': [], 'balanced_accuracy': [], 'macro_f1': [], 
            'weighted_f1': [], 'kappa': [], 'mcc': [],
            'per_class_f1': [[] for _ in range(4)]
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"  Fold {fold + 1}/{cv_folds}...")
            
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model with best parameters
            model = xgb.XGBClassifier(**self.best_params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
            
            # Predictions
            y_pred = model.predict(X_val_fold)
            
            # Calculate metrics
            cv_results['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            cv_results['balanced_accuracy'].append(balanced_accuracy_score(y_val_fold, y_pred))
            cv_results['macro_f1'].append(f1_score(y_val_fold, y_pred, average='macro'))
            cv_results['weighted_f1'].append(f1_score(y_val_fold, y_pred, average='weighted'))
            cv_results['kappa'].append(cohen_kappa_score(y_val_fold, y_pred))
            cv_results['mcc'].append(matthews_corrcoef(y_val_fold, y_pred))
            
            # Per-class F1 scores
            per_class_f1 = f1_score(y_val_fold, y_pred, average=None)
            for i, f1 in enumerate(per_class_f1):
                cv_results['per_class_f1'][i].append(f1)
        
        # Calculate mean and std for each metric
        cv_summary = {}
        for metric, values in cv_results.items():
            if metric != 'per_class_f1':
                cv_summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        # Per-class F1 summary
        cv_summary['per_class_f1'] = {}
        for i in range(4):
            cv_summary['per_class_f1'][f'class_{i}'] = {
                'mean': np.mean(cv_results['per_class_f1'][i]),
                'std': np.std(cv_results['per_class_f1'][i])
            }
        
        self.cv_results = cv_summary
        
        # Print results
        print("\nCross-Validation Results:")
        print("=" * 50)
        for metric, stats in cv_summary.items():
            if metric != 'per_class_f1':
                print(f"{metric.replace('_', ' ').title()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        print("\nPer-Class F1 Scores:")
        for class_name, stats in cv_summary['per_class_f1'].items():
            print(f"{class_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Save CV results
        with open(self.output_dir / 'cv_results.json', 'w') as f:
            json.dump(cv_summary, f, indent=2)
        
        return cv_summary
    
    def train_final_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train the final model with optimized parameters."""
        print("Training final model...")
        
        self.best_model = xgb.XGBClassifier(**self.best_params)
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.best_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        # Save model
        joblib.dump(self.best_model, self.output_dir / 'best_model.pkl')
        
        return self.best_model
    
    def evaluate_model(self, X_test, y_test, parcel_ids_test=None):
        """Comprehensive model evaluation."""
        print("Evaluating final model...")
        
        # Predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
            'kappa': cohen_kappa_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'per_class_f1': f1_score(y_test, y_pred, average=None).tolist()
        }
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"\n{'='*50}")
        print("FINAL MODEL PERFORMANCE")
        print(f"{'='*50}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"Cohen's Kappa: {metrics['kappa']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        
        print(f"\nPer-Class F1 Scores:")
        for i, f1 in enumerate(metrics['per_class_f1']):
            print(f"  Class {i}: {f1:.4f}")
        
        # Save predictions with parcel IDs
        predictions_df = None
        if parcel_ids_test is not None:
            predictions_df = pd.DataFrame({
                'PARCEL_ID': parcel_ids_test,
                'true_label': y_test,
                'predicted_label': y_pred
            })
            
            # Add probability columns
            for i in range(y_pred_proba.shape[1]):
                predictions_df[f'prob_class_{i}'] = y_pred_proba[:, i]
            
            predictions_df.to_csv(self.output_dir / 'test_predictions.csv', index=False)
        
        # Save evaluation results
        evaluation_results = {
            'metrics': metrics,
            'classification_report': report,
            'cv_results': self.cv_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        self.create_evaluation_plots(y_test, y_pred, y_pred_proba, metrics)
        
        return metrics, predictions_df
    
    def create_evaluation_plots(self, y_test, y_pred, y_pred_proba, metrics):
        """Create comprehensive evaluation visualizations."""
        
        # 1. Metrics comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Overall metrics
        overall_metrics = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'kappa', 'mcc']
        metric_values = [metrics[m] for m in overall_metrics]
        
        bars = ax1.bar(range(len(overall_metrics)), metric_values, 
                      color=['blue', 'green', 'orange', 'red', 'purple', 'brown'])
        ax1.set_xticks(range(len(overall_metrics)))
        ax1.set_xticklabels([m.replace('_', '\n').title() for m in overall_metrics], rotation=45)
        ax1.set_ylim(0, 1)
        ax1.set_title('Overall Performance Metrics')
        
        # Add value labels
        for bar, val in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Per-class F1 scores
        per_class_f1 = metrics['per_class_f1']
        colors = ['green', 'yellow', 'orange', 'red']
        bars2 = ax2.bar(range(len(per_class_f1)), per_class_f1, color=colors)
        ax2.set_xticks(range(len(per_class_f1)))
        ax2.set_xticklabels([f'Class {i}' for i in range(len(per_class_f1))])
        ax2.set_ylim(0, 1)
        ax2.set_title('Per-Class F1 Scores')
        
        # Add value labels
        for bar, val in zip(bars2, per_class_f1):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Confusion matrices
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title('Confusion Matrix (Counts)')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax4)
        ax4.set_title('Confusion Matrix (Normalized)')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'evaluation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature importance plot
        if hasattr(self.best_model, 'feature_importances_'):
            self.plot_feature_importance()
    
    def plot_feature_importance(self):
        """Plot feature importance analysis."""
        if self.best_model is None:
            return
        
        # Get feature importance
        if hasattr(self, 'selected_features'):
            feature_names = self.selected_features
        else:
            feature_names = self.feature_names
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['feature'][:20], importance_df['importance'][:20])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save importance data
        importance_df.to_csv(self.output_dir / 'feature_importance.csv', index=False)
        
        return importance_df
    
    def analyze_data(self, data):
        """Perform comprehensive data analysis with visualizations (matching baseline)."""
        print("Analyzing dataset...")
        
        analytics = {
            'total_samples': len(data),
            'feature_count': len([col for col in data.columns if col not in ['PARCEL_ID', 'BLIGHT_LABEL']]),
            'features': [col for col in data.columns if col not in ['PARCEL_ID', 'BLIGHT_LABEL']]
        }
        
        # Label distribution analysis
        label_dist = data['BLIGHT_LABEL'].value_counts().sort_index()
        analytics['label_distribution'] = label_dist.to_dict()
        analytics['label_percentages'] = (label_dist / len(data) * 100).round(2).to_dict()
        
        print(f"\nDataset Analytics:")
        print(f"Total samples: {analytics['total_samples']}")
        print(f"Features: {analytics['feature_count']}")
        print("\nLabel distribution:")
        for label, count in analytics['label_distribution'].items():
            pct = analytics['label_percentages'][label]
            print(f"  Label {label}: {count} ({pct}%)")
        
        # Save analytics
        with open(self.output_dir / 'data_analytics.json', 'w') as f:
            json.dump(analytics, f, indent=2)
        
        # Create visualizations (matching baseline exactly)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Subplot 1: Count distribution
        colors = ['green', 'yellow', 'orange', 'red']
        bars = ax1.bar(label_dist.index, label_dist.values, color=colors)
        ax1.set_xlabel('Blight Label')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Blight Labels (Counts)')
        ax1.set_xticks(label_dist.index)
        
        # Add count labels on bars
        for bar, count in zip(bars, label_dist.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 100,
                    f'{count:,}', ha='center', va='bottom')
        
        # Subplot 2: Percentage distribution (pie chart)
        percentages = [analytics['label_percentages'][i] for i in sorted(analytics['label_percentages'].keys())]
        labels = [f'Label {i}\n({pct}%)' for i, pct in zip(label_dist.index, percentages)]
        ax2.pie(percentages, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Distribution of Blight Labels (Percentages)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'label_distribution.png', dpi=300)
        plt.close()
        
        # Create summary statistics plot
        fig, ax = plt.subplots(figsize=(8, 6))
        summary_data = {
            'Total Samples': analytics['total_samples'],
            'Number of Features': analytics['feature_count'],
            'Class 0 (No Blight)': analytics['label_distribution'][0],
            'Class 1 (Noticeable)': analytics['label_distribution'][1],
            'Class 2 (Significant)': analytics['label_distribution'][2],
            'Class 3 (Extreme)': analytics['label_distribution'][3]
        }
        
        ax.axis('off')
        y_pos = 0.9
        for key, value in summary_data.items():
            ax.text(0.1, y_pos, f'{key}:', fontsize=12, fontweight='bold')
            ax.text(0.6, y_pos, f'{value:,}', fontsize=12)
            y_pos -= 0.12
        
        ax.set_title('Dataset Summary Statistics', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'data_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return analytics

    def run_full_pipeline(self, features_path, labels_path, n_trials=50, cv_folds=5):
        """Run the complete optimization pipeline."""
        print("🚀 Starting XGBoost Optimization Pipeline")
        print("=" * 60)
        
        # Load data
        data = self.load_data(features_path, labels_path)
        
        # Analyze data (matching baseline)
        analytics = self.analyze_data(data)
        
        # Extract parcel IDs
        parcel_ids = data['PARCEL_ID']
        
        # Prepare features and target
        X = self.engineer_features(data.drop(['PARCEL_ID', 'BLIGHT_LABEL'], axis=1))
        y = data['BLIGHT_LABEL']
        
        # Feature selection
        X_selected = self.select_features(X, y, k=min(15, len(X.columns)))
        
        # Split data
        X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
            X_selected, y, np.arange(len(X_selected)), 
            test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X_temp, y_temp, idx_temp,
            test_size=0.125, random_state=self.random_state, stratify=y_temp
        )
        
        parcel_ids_test = parcel_ids.iloc[idx_test]
        
        print(f"\nData split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples") 
        print(f"  Test: {len(X_test)} samples")
        
        # Hyperparameter optimization
        study = self.optimize_hyperparameters(X_train, y_train, n_trials, cv_folds)
        
        # Cross-validation
        cv_results = self.cross_validate(X_temp, y_temp, cv_folds)
        
        # Train final model
        final_model = self.train_final_model(X_train, y_train, X_val, y_val)
        
        # Final evaluation
        metrics, predictions_df = self.evaluate_model(X_test, y_test, parcel_ids_test)
        
        print(f"\n🎉 Pipeline completed! Results saved to: {self.output_dir}")
        
        return final_model, metrics, predictions_df, analytics

def main():
    """Main execution function."""
    # Configuration
    features_path = "../training_data/blight_features.csv"
    labels_path = "../training_data/blight_labels.csv"
    
    # Initialize optimizer
    optimizer = XGBoostOptimizer(output_dir='../deliverables/xgboost_optimized1')
    
    # Run optimization pipeline
    model, metrics, predictions, analytics = optimizer.run_full_pipeline(
        features_path=features_path,
        labels_path=labels_path,
        n_trials=5,  # Quick test - increase for production
        cv_folds=5
    )
    
    return optimizer, model, metrics, predictions, analytics

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import optuna
    except ImportError:
        print("Installing optuna...")
        import subprocess
        subprocess.run(["pip", "install", "optuna"])
        import optuna
    
    optimizer, model, metrics, predictions, analytics = main()