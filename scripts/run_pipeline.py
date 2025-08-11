import os
import sys
import yaml
import argparse
import pandas as pd

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.generate_data import generate_traffic_data
from src.features.feature_engineering import engineer_features
from src.models.train_model import train_model
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from datetime import datetime, timezone
import json
from src.visualization.visualize import (
    plot_class_distribution, plot_traffic_volume, 
    plot_protocol_distribution, plot_congestion_over_time,
    plot_confusion_matrix, plot_feature_importance
)
from src.utils.logger import get_logger

logger = get_logger()

def run_pipeline(config_path='config/config.yaml', generate_new_data=False):
    """
    Run the complete traffic analysis pipeline
    
    Args:
        config_path: Path to configuration file
        generate_new_data: Whether to generate new data or use existing
    """
    logger.info("Starting traffic analysis pipeline")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Step 1: Data Generation/Loading
    raw_data_path = config['data']['raw_path']
    
    if generate_new_data or not os.path.exists(raw_data_path):
        logger.info("Generating new traffic data")
        df = generate_traffic_data(n_samples=1000, output_path=raw_data_path)
    else:
        logger.info(f"Loading existing data from {raw_data_path}")
        df = pd.read_csv(raw_data_path, parse_dates=['timestamp'])
    
    # Step 2: Exploratory Data Analysis
    logger.info("Performing exploratory data analysis")
    plot_class_distribution(df)
    plot_traffic_volume(df)
    plot_protocol_distribution(df)
    plot_congestion_over_time(df)
    
    # Step 3: Feature Engineering
    logger.info("Performing feature engineering")
    processed_df = engineer_features(df)
    
    # Step 4: Model Training
    logger.info("Training the model")
    X = processed_df.drop('congestion', axis=1)
    y = processed_df['congestion']
    
    model, X_test, y_test, y_pred = train_model(X, y, config_path=config_path)

    # Compute metrics for dashboard
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        'precision': round(float(precision_score(y_test, y_pred)), 4),
        'recall': round(float(recall_score(y_test, y_pred)), 4),
        'f1_score': round(float(f1_score(y_test, y_pred)), 4),
        'auc_roc': round(float(roc_auc_score(y_test, y_pred)), 4),
        'samples': int(len(X)),
        'features': int(X.shape[1]),
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }
    os.makedirs('models', exist_ok=True)
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to models/metrics.json")

    # Append to history
    history_path = 'models/metrics_history.json'
    try:
        if os.path.exists(history_path):
            with open(history_path, 'r') as hf:
                history = json.load(hf)
            if not isinstance(history, list):
                history = []
        else:
            history = []
    except Exception:
        history = []
    history.append(metrics)
    # keep only last 100 entries
    history = history[-100:]
    with open(history_path, 'w') as hf:
        json.dump(history, hf, indent=2)
    logger.info("Appended metrics to metrics_history.json (size=%d)" % len(history))
    
    # Step 5: Model Evaluation Visualization
    logger.info("Creating model evaluation visualizations")
    plot_confusion_matrix(y_test, y_pred)
    importance_df = plot_feature_importance(model, X.columns)

    # Global SHAP (tree-based models only)
    try:
        import shap
        tree_like = any(k in type(model).__name__.lower() for k in ['forest','boost','xgb','gbm','tree'])
        if tree_like:
            sample = X.sample(n=min(500, len(X)), random_state=42)
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(sample)
            # Handle list per-class; choose positive class if available
            if isinstance(sv, list):
                sv_use = sv[1] if len(sv) > 1 else sv[0]
            else:
                sv_use = sv
            import numpy as _np
            vals = _np.abs(_np.array(sv_use))
            if vals.ndim == 3:  # sometimes (classes, samples, features)
                vals = vals[0]
            mean_abs = vals.mean(axis=0)
            ordering = mean_abs.argsort()[::-1]
            global_imp = [
                {'feature': str(X.columns[i]), 'mean_abs_shap': float(mean_abs[i])}
                for i in ordering
            ]
            with open('models/shap_global.json','w') as gf:
                json.dump({'generated_at': datetime.now(timezone.utc).isoformat(), 'items': global_imp}, gf, indent=2)
            logger.info('Saved global SHAP importance to models/shap_global.json')
    except Exception as e:
        logger.warning(f"Global SHAP computation skipped: {e}")
    
    logger.info("Pipeline completed successfully")
    return model, importance_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the traffic analysis pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--generate-data', action='store_true',
                        help='Generate new data instead of using existing')

    args = parser.parse_args()
    
    # Run the pipeline
    model, importance_df = run_pipeline(
        config_path=args.config,
        generate_new_data=args.generate_data
    )
    
    # Print top 5 important features
    print("\nTop 5 Important Features:")
    print(importance_df.head(5))