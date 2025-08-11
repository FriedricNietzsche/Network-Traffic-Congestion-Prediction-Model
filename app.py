from flask import Flask, request, render_template, jsonify, send_from_directory, abort, Response, redirect, url_for, stream_with_context
import os
import sys
import pandas as pd
import threading
import time
from datetime import datetime, timezone
import shutil

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.predict import predict_congestion
import yaml
try:  # optional dependency
    from flask_cors import CORS  # type: ignore
    _CORS_AVAILABLE = True
except Exception:  # pragma: no cover
    _CORS_AVAILABLE = False
from src.utils.logger import get_logger
from datetime import datetime

application = Flask(__name__)
app = application

logger = get_logger()

# Simple metrics loader
import json
METRICS_PATH = 'models/metrics.json'
MODEL_PATH = 'models/congestion_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
VERSIONS_DIR = 'models/versions'
GLOBAL_SHAP_PATH = 'models/shap_global.json'

def load_metrics():
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def list_model_versions():
    if not os.path.isdir(VERSIONS_DIR):
        return []
    return sorted([f for f in os.listdir(VERSIONS_DIR) if f.startswith('model_') and f.endswith('.pkl')], reverse=True)

def list_model_versions_detailed(limit: int = 10):
    """Return list of version artifacts with metrics if available."""
    if not os.path.isdir(VERSIONS_DIR):
        return []
    versions = []
    for fn in sorted(os.listdir(VERSIONS_DIR), reverse=True):
        if not fn.startswith('model_') or not fn.endswith('.pkl'):
            continue
        ts = fn[len('model_'):-4]
        metrics_file = os.path.join(VERSIONS_DIR, f'metrics_{ts}.json')
        metrics = None
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file,'r') as mf:
                    metrics = json.load(mf)
            except Exception:
                metrics = None
        versions.append({'model_file': fn, 'timestamp': ts, 'metrics': metrics})
        if len(versions) >= limit:
            break
    return versions

# Load API config
try:
    with open('config/config.yaml', 'r') as f:
        _cfg = yaml.safe_load(f)
    if _CORS_AVAILABLE and _cfg.get('api', {}).get('enable_cors'):
        CORS(app)
except Exception:  # fall back silently
    _cfg = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    metrics = load_metrics()
    plot_dir = 'visualization'
    plots = []
    if os.path.isdir(plot_dir):
        for fn in sorted(os.listdir(plot_dir)):
            if fn.lower().endswith('.png'):
                plots.append(fn)  # store only filename
    versions = list_model_versions()
    return render_template('dashboard.html', metrics=metrics, plots=plots, versions=versions, config=_cfg)

@app.route('/visualization/<path:filename>')
def visualization_file(filename):
    safe_dir = 'visualization'
    if not os.path.isdir(safe_dir):
        abort(404)
    return send_from_directory(safe_dir, filename)

@app.route('/api/metrics')
def api_metrics():
    return jsonify(load_metrics())

@app.route('/api/metrics/history')
def api_metrics_history():
    path = 'models/metrics_history.json'
    if os.path.exists(path):
        try:
            import json
            with open(path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                # Deduplicate by trained_at timestamp keeping first occurrence order
                seen = set()
                deduped = []
                for row in data:
                    ts = row.get('trained_at')
                    if ts in seen:
                        continue
                    seen.add(ts)
                    deduped.append(row)
                # Keep only last 50 entries
                deduped = deduped[-50:]
                return jsonify(deduped)
        except Exception:
            pass
    return jsonify([])

@app.route('/download/model')
def download_model():
    if os.path.exists(MODEL_PATH):
        return send_from_directory('models', os.path.basename(MODEL_PATH), as_attachment=True)
    return Response('Model not found', status=404)

@app.route('/download/scaler')
def download_scaler():
    if os.path.exists(SCALER_PATH):
        return send_from_directory('models', os.path.basename(SCALER_PATH), as_attachment=True)
    return Response('Scaler not found', status=404)

@app.route('/download/metrics')
def download_metrics():
    if os.path.exists(METRICS_PATH):
        return send_from_directory('models', os.path.basename(METRICS_PATH), as_attachment=True)
    return Response('Metrics not found', status=404)

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('predict.html')
    else:
        try:
            # Collect input data from the form
            input_data = {
                'packet_size': int(request.form.get('packet_size')),
                'bytes_sent': int(request.form.get('bytes_sent')),
                'source_ip': request.form.get('source_ip'),
                'dest_ip': request.form.get('dest_ip'),
                'protocol': request.form.get('protocol'),
                'timestamp_seconds': int(request.form.get('timestamp_seconds')),
                "hour": int(request.form.get('hour'))

            }
            
            # Make prediction
            outcome = predict_congestion(input_data, return_proba=True)
            error = None
            result = None
            proba = None
            if isinstance(outcome, tuple):
                result, proba = outcome
            elif isinstance(outcome, (int, float)):
                result = int(outcome)
            elif isinstance(outcome, str):
                error = outcome
            else:
                error = "Unexpected prediction output format"

            if error:
                return render_template('predict.html', error=error)

            congestion_status = "Congested" if result == 1 else "Normal"
            return render_template('predict.html', result=congestion_status, probability=f"{proba:.3f}" if proba is not None else None)
        
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return render_template('predict.html', error="Invalid input or server error.")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        threshold = float(request.args.get('threshold', _cfg.get('api', {}).get('default_threshold', 0.5)))
        outcome = predict_congestion(data, return_proba=True)
        if isinstance(outcome, tuple):
            result, proba = outcome
            # Apply custom threshold to probability if available
            if proba is not None:
                result = 1 if proba >= threshold else 0
            return jsonify({
                'prediction': int(result),
                'label': 'Congested' if result == 1 else 'Normal',
                'probability': proba,
                'threshold': threshold
            })
        if isinstance(outcome, (int, float)):
            result = int(outcome)
            return jsonify({
                'prediction': result,
                'label': 'Congested' if result == 1 else 'Normal',
                'threshold': threshold
            })
        return jsonify({'error': outcome}), 400
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/versions')
def api_versions():
    return jsonify(list_model_versions())

@app.route('/api/versions/detail')
def api_versions_detail():
    return jsonify(list_model_versions_detailed())

@app.route('/download/version/<path:filename>')
def download_version(filename):
    target = os.path.join(VERSIONS_DIR, filename)
    if os.path.exists(target):
        return send_from_directory(VERSIONS_DIR, filename, as_attachment=True)
    return Response('Version not found', status=404)

@app.route('/api/explain', methods=['POST'])
def api_explain():
    # Basic SHAP explanation for a single instance
    try:
        import importlib, numpy as np
        shap = importlib.import_module('shap')
        from src.features.feature_engineering import engineer_features
        from src.models.predict import load_artifacts
        data = request.get_json(force=True)
        model, scaler, X_ref = load_artifacts('config/config.yaml')
        single_df = pd.DataFrame([data])
        if 'timestamp_seconds' in single_df.columns and 'timestamp' not in single_df.columns:
            single_df['timestamp'] = pd.to_datetime(single_df['timestamp_seconds'], unit='s')
        processed = engineer_features(single_df, is_training=False, scaler_path=SCALER_PATH)
        for col in X_ref.columns:
            if col not in processed.columns:
                processed[col] = 0
        processed = processed[X_ref.columns]
        # Restrict to tree-based models for now
        is_tree = any(name in type(model).__name__.lower() for name in ['forest','boost','xgb','gbm','tree'])
        if not is_tree:
            return jsonify({'error': 'SHAP explanation currently supported only for tree-based models.'}), 400
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(processed)
        # sv can be list (per-class) or array
        if isinstance(sv, list):
            if len(sv) >= 2:
                sv_use = sv[1][0]  # positive class
            else:
                sv_use = sv[0][0]
        else:
            sv_use = sv[0] if getattr(sv, 'ndim', 1) > 1 else sv
        values = np.array(sv_use, dtype=float).ravel()
        # Pair with columns (cast to str) and sort
        pairs = list(zip([str(c) for c in X_ref.columns], values))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top = pairs[:10]
        result = [{'feature': f, 'shap_value': float(v)} for f, v in top]
        # Collapse paired cyclical features optionally
        collapse = request.args.get('collapse_pairs', 'true').lower() == 'true'
        if collapse:
            cyc_pairs = [('hour_sin','hour_cos')]
            collapsed = []
            used = set()
            for a,b in cyc_pairs:
                fa = next((r for r in result if r['feature']==a), None)
                fb = next((r for r in result if r['feature']==b), None)
                if fa or fb:
                    mag = 0.0
                    if fa: mag += abs(fa['shap_value']); used.add(a)
                    if fb: mag += abs(fb['shap_value']); used.add(b)
                    collapsed.append({'feature': f'{a[:-4]}_cycle', 'shap_value': mag})
            for r in result:
                if r['feature'] not in used:
                    collapsed.append(r)
            result = collapsed
        # Actual model probability
        try:
            prob = float(model.predict_proba(processed)[0][1])
        except Exception:
            prob = None
        expected_raw = explainer.expected_value
        if isinstance(expected_raw, (list, tuple, np.ndarray)):
            if isinstance(expected_raw, (list, tuple)) and len(expected_raw) >= 2:
                expected_value = float(expected_raw[1])
            elif isinstance(expected_raw, np.ndarray) and expected_raw.size >= 2:
                expected_value = float(expected_raw.flat[-1])
            else:
                expected_value = float(expected_raw[0]) if isinstance(expected_raw, (list, tuple)) else float(expected_raw)
        else:
            expected_value = float(expected_raw)
        # Fallback if any value could not be cast
        if not result:
            fi = getattr(model, 'feature_importances_', None)
            if fi is not None:
                fi = np.asarray(fi)
                top_idx = np.argsort(fi)[-10:][::-1]
                result = [{'feature': str(X_ref.columns[i]), 'importance': float(fi[i])} for i in top_idx]
        n = request.args.get('n')
        if n:
            try:
                k = int(n)
                if k > 0:
                    result = result[:k]
            except Exception:
                pass
        # Feature values for display (only for first row)
        feature_values = {}
        try:
            row_vals = processed.iloc[0]
            for r in result:
                f = r['feature']
                if f in row_vals.index:
                    val = row_vals[f]
                    try:
                        feature_values[f] = float(val)
                    except Exception:
                        feature_values[f] = str(val)
        except Exception:
            pass
        # Decision with optional threshold param
        threshold = None
        decision = None
        if prob is not None:
            try:
                threshold = float(request.args.get('threshold', _cfg.get('api', {}).get('default_threshold', 0.5)))
                decision = int(prob >= threshold)
            except Exception:
                pass
        n = request.args.get('n')
        if n:
            try:
                k = int(n)
                if k > 0:
                    result = result[:k]
            except Exception:
                pass
        return jsonify({'features': result, 'expected_value': expected_value, 'probability': prob, 'threshold': threshold, 'decision': decision, 'feature_values': feature_values})
    except Exception as e:
        logger.error(f"Explain error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/toggle-theme')
def toggle_theme():
    resp = redirect(request.referrer or url_for('index'))
    current = request.cookies.get('theme', 'light')
    resp.set_cookie('theme', 'dark' if current != 'dark' else 'light', max_age=60*60*24*365)
    return resp

@app.route('/api/explain/global')
def api_explain_global():
    path = 'models/shap_global.json'
    if os.path.exists(path):
        try:
            with open(path,'r') as f:
                import json
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    return jsonify({'error': 'Global SHAP not available'}), 404

@app.route('/api/train', methods=['POST'])
def api_train():
    """Trigger retraining pipeline from the web UI.
    Query param generate_data=true regenerates synthetic data first.
    """
    generate_flag = request.args.get('generate_data', 'false').lower() == 'true'
    try:
        from scripts.run_pipeline import run_pipeline
        run_pipeline(generate_new_data=generate_flag)
        metrics = load_metrics()
        return jsonify({
            'status': 'ok',
            'generated_new_data': generate_flag,
            'trained_at': metrics.get('trained_at'),
            'metrics': metrics
        })
    except Exception as e:
        logger.error(f"Training trigger failed: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

# ---------------------- Async training + SSE -----------------------
TRAIN_STATUS = {
    'state': 'idle',
    'message': 'Idle',
    'progress': 0.0,
    'started_at': None,
    'updated_at': None,
    'error': None
}
TRAIN_LOCK = threading.Lock()
TRAIN_CANCEL_FLAG = {'cancel': False}

def _update_train(state=None, message=None, progress=None, error=None):
    with TRAIN_LOCK:
        if state: TRAIN_STATUS['state'] = state
        if message is not None: TRAIN_STATUS['message'] = message
        if progress is not None: TRAIN_STATUS['progress'] = float(progress)
        if error is not None: TRAIN_STATUS['error'] = error
        TRAIN_STATUS['updated_at'] = datetime.utcnow().isoformat()
        if state == 'starting':
            TRAIN_STATUS['started_at'] = TRAIN_STATUS['updated_at']

def _background_train(generate_new_data: bool):
    try:
        _update_train(state='starting', message='Starting training', progress=0.0)
        # Import inside thread to avoid blocking main import path
        import yaml, json
        from src.data.generate_data import generate_traffic_data
        from src.features.feature_engineering import engineer_features
        from src.models.train_model import train_model
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        config_path = 'config/config.yaml'
        _update_train(message='Loading config', progress=2)
        with open(config_path,'r') as f:
            config = yaml.safe_load(f)
        raw_data_path = config['data']['raw_path']
        _update_train(state='running')
        import time as _t
        phase_start = _t.time()
        phase_times = {}

        if generate_new_data or not os.path.exists(raw_data_path):
            _update_train(message='Generating data', progress=5)
            df = generate_traffic_data(n_samples=1000, output_path=raw_data_path)
        else:
            _update_train(message='Loading data', progress=5)
            df = pd.read_csv(raw_data_path, parse_dates=['timestamp'])
        phase_times['data'] = round(_t.time() - phase_start, 3)
        if TRAIN_CANCEL_FLAG['cancel']:
            _update_train(state='cancelled', message='Cancelled after data load', progress=0)
            return
        phase_start = _t.time()
        _update_train(message='Feature engineering', progress=25)
        processed_df = engineer_features(df)
        phase_times['feature_engineering'] = round(_t.time() - phase_start, 3)
        if TRAIN_CANCEL_FLAG['cancel']:
            _update_train(state='cancelled', message='Cancelled after feature engineering', progress=0)
            return
        phase_start = _t.time()
        _update_train(message='Training model', progress=50)
        X = processed_df.drop('congestion', axis=1)
        y = processed_df['congestion']
        model, X_test, y_test, y_pred = train_model(X, y, config_path=config_path)
        phase_times['training'] = round(_t.time() - phase_start, 3)
        if TRAIN_CANCEL_FLAG['cancel']:
            _update_train(state='cancelled', message='Cancelled after training', progress=0)
            return
        phase_start = _t.time()
        _update_train(message='Scoring model', progress=75)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics = {
            'precision': round(float(precision_score(y_test, y_pred)), 4),
            'recall': round(float(recall_score(y_test, y_pred)), 4),
            'f1_score': round(float(f1_score(y_test, y_pred)), 4),
            'auc_roc': round(float(roc_auc_score(y_test, y_pred)), 4),
            'samples': int(len(X)),
            'features': int(X.shape[1]),
            'trained_at': datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
        }
        os.makedirs('models', exist_ok=True)
        with open(METRICS_PATH,'w') as f:
            json.dump(metrics,f,indent=2)
        # Update history
        history_path = 'models/metrics_history.json'
        try:
            if os.path.exists(history_path):
                with open(history_path,'r') as hf:
                    hist = json.load(hf)
                if not isinstance(hist,list): hist=[]
            else:
                hist=[]
        except Exception:
            hist=[]
        hist.append(metrics); hist = hist[-100:]
        with open(history_path,'w') as hf: json.dump(hist,hf,indent=2)
        phase_times['scoring'] = round(_t.time() - phase_start, 3)
        if TRAIN_CANCEL_FLAG['cancel']:
            _update_train(state='cancelled', message='Cancelled after scoring', progress=0)
            return
        phase_start = _t.time()
        _update_train(message='Computing global SHAP', progress=85)
        # Global SHAP (best effort)
        try:
            import shap, numpy as _np
            if any(k in type(model).__name__.lower() for k in ['forest','boost','xgb','gbm','tree']):
                sample = X.sample(n=min(400,len(X)), random_state=42)
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(sample)
                if isinstance(sv, list):
                    sv_use = sv[1] if len(sv)>1 else sv[0]
                else:
                    sv_use = sv
                vals = _np.abs(_np.array(sv_use))
                if vals.ndim==3: vals=vals[0]
                mean_abs = vals.mean(axis=0)
                ordering = mean_abs.argsort()[::-1]
                global_imp = [{'feature': str(X.columns[i]), 'mean_abs_shap': float(mean_abs[i])} for i in ordering]
                # Save global shap aggregate
                with open(GLOBAL_SHAP_PATH,'w') as gf:
                    json.dump({'generated_at': datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(), 'items': global_imp}, gf, indent=2)
        except Exception as e:
            logger.warning(f"Async SHAP skipped: {e}")
            global_imp = []
        phase_times['shap'] = round(_t.time() - phase_start, 3)
        if TRAIN_CANCEL_FLAG['cancel']:
            _update_train(state='cancelled', message='Cancelled after SHAP', progress=0)
            return
        phase_start = _t.time()
        _update_train(message='Saving version', progress=92)
        # Versioning
        try:
            with open('config/config.yaml','r') as cf: cfg2 = yaml.safe_load(cf) or {}
            vcfg = (cfg2.get('versioning') or {})
            if vcfg.get('enable', True):
                os.makedirs(VERSIONS_DIR, exist_ok=True)
                ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                model_file = os.path.join(VERSIONS_DIR, f'model_{ts}.pkl')
                import pickle
                with open(model_file,'wb') as mf: pickle.dump(model,mf)
                # copy scaler if exists
                if os.path.exists(SCALER_PATH):
                    shutil.copy2(SCALER_PATH, os.path.join(VERSIONS_DIR, f'scaler_{ts}.pkl'))
                # enrich metrics with phase times & shap top
                phase_times['version_save'] = round(_t.time() - phase_start, 3)
                total_duration = sum(v for v in phase_times.values())
                metrics['phase_durations'] = phase_times
                metrics['total_duration_sec'] = total_duration
                # add shap top features
                if 'global_imp' in locals() and global_imp:
                    metrics['shap_top'] = global_imp[:10]
                with open(os.path.join(VERSIONS_DIR, f'metrics_{ts}.json'),'w') as mf:
                    json.dump(metrics, mf, indent=2)
                # prune
                keep = int(vcfg.get('keep_last',5))
                model_versions = sorted([f for f in os.listdir(VERSIONS_DIR) if f.startswith('model_') and f.endswith('.pkl')])
                if len(model_versions) > keep:
                    for old in model_versions[:-keep]:
                        try:
                            ts_old = old[len('model_'):-4]
                            os.remove(os.path.join(VERSIONS_DIR, old))
                            mfile = os.path.join(VERSIONS_DIR, f'metrics_{ts_old}.json')
                            if os.path.exists(mfile): os.remove(mfile)
                            sfile = os.path.join(VERSIONS_DIR, f'scaler_{ts_old}.pkl')
                            if os.path.exists(sfile): os.remove(sfile)
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"Versioning save failed: {e}")
        _update_train(state='completed', message='Training complete', progress=100.0)
    except Exception as e:  # pragma: no cover
        logger.error(f"Async training failed: {e}")
        _update_train(state='error', message='Training failed', progress=0, error=str(e))

@app.route('/api/train/async', methods=['POST'])
def api_train_async():
    generate_flag = request.args.get('generate_data','false').lower()=='true'
    with TRAIN_LOCK:
        if TRAIN_STATUS['state'] in ('running','starting'):
            return jsonify({'status':'busy','message':'Training already in progress'}), 409
        TRAIN_STATUS['state']='starting'
        TRAIN_STATUS['message']='Queued'
        TRAIN_STATUS['progress']=0.0
        TRAIN_STATUS['error']=None
    t = threading.Thread(target=_background_train, args=(generate_flag,), daemon=True)
    t.start()
    return jsonify({'status':'started','generate_new_data':generate_flag})

@app.route('/api/train/events')
def api_train_events():
    def event_stream():
        last_state = None
        while True:
            with TRAIN_LOCK:
                payload = dict(TRAIN_STATUS)
            if payload['state'] in ('idle','completed','error') and last_state == payload['state'] and payload['state'] != 'starting':
                # emit final then break
                if payload['state'] in ('completed','error'):
                    yield f"data: {json.dumps(payload)}\n\n"
                    break
            if payload != last_state:
                yield f"data: {json.dumps(payload)}\n\n"
                last_state = payload['state']
            if payload['state'] in ('completed','error'):
                break
            time.sleep(1)
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

@app.route('/api/train/cancel', methods=['POST'])
def api_train_cancel():
    with TRAIN_LOCK:
        if TRAIN_STATUS['state'] not in ('starting','running'):
            return jsonify({'status':'idle'}), 400
        TRAIN_CANCEL_FLAG['cancel'] = True
    return jsonify({'status':'cancelling'})

@app.route('/api/versions/rollback/<timestamp>', methods=['POST'])
def api_versions_rollback(timestamp):
    # Accept timestamps in format YYYYMMDD_HHMMSS (with underscore) or pure digits
    import re
    if not re.fullmatch(r'\d{8}_\d{6}|\d+', timestamp):
        return jsonify({'error':'Invalid timestamp format'}), 400
    model_file = os.path.join(VERSIONS_DIR, f'model_{timestamp}.pkl')
    if not os.path.exists(model_file):
        return jsonify({'error':'Model version not found'}), 404
    scaler_file = os.path.join(VERSIONS_DIR, f'scaler_{timestamp}.pkl')
    metrics_file = os.path.join(VERSIONS_DIR, f'metrics_{timestamp}.json')
    shap_file = os.path.join(VERSIONS_DIR, f'shap_global_{timestamp}.json')
    try:
        shutil.copy2(model_file, MODEL_PATH)
        if os.path.exists(scaler_file):
            shutil.copy2(scaler_file, SCALER_PATH)
        if os.path.exists(metrics_file):
            shutil.copy2(metrics_file, METRICS_PATH)
        if os.path.exists(shap_file):
            shutil.copy2(shap_file, GLOBAL_SHAP_PATH)
        # append rollback event to history
        hist = []
        history_path = 'models/metrics_history.json'
        try:
            if os.path.exists(history_path):
                with open(history_path,'r') as hf: hist = json.load(hf)
                if not isinstance(hist,list): hist=[]
        except Exception:
            hist=[]
        rollback_entry = {'event':'rollback','rolled_to':timestamp,'at': datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()}
        hist.append(rollback_entry)
        hist = hist[-150:]
        with open(history_path,'w') as hf: json.dump(hist,hf,indent=2)
        return jsonify({'status':'ok','rolled_to':timestamp})
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    host = _cfg.get('api', {}).get('default_host', '0.0.0.0') if '_cfg' in globals() else '0.0.0.0'
    port = _cfg.get('api', {}).get('default_port', 5000) if '_cfg' in globals() else 5000
    app.run(debug=True, host=host, port=port)
