from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

app = Flask(__name__)

# Define model directory and timestamp
MODEL_DIR = 'models'
TIMESTAMP = '20251005_153914'  # Replace with the timestamp from your model_training.py output

# Load regression model and metadata
reg_metadata_path = os.path.join(MODEL_DIR, f'regression_metadata_{TIMESTAMP}.json')
if not os.path.exists(reg_metadata_path):
    raise FileNotFoundError(f"Regression metadata file not found: {reg_metadata_path}")
with open(reg_metadata_path, 'r') as f:
    content = f.read().strip()
    if not content:
        raise ValueError("Regression metadata JSON file is empty")
    reg_metadata = json.loads(content)

reg_model_path = os.path.join(MODEL_DIR, f'regression_model_{TIMESTAMP}.pkl')
if not os.path.exists(reg_model_path):
    raise FileNotFoundError(f"Regression model file not found: {reg_model_path}")
reg_model = joblib.load(reg_model_path)
reg_features = reg_metadata['features']

# Load classification model and metadata
class_metadata_path = os.path.join(MODEL_DIR, f'classification_metadata_{TIMESTAMP}.json')
if not os.path.exists(class_metadata_path):
    raise FileNotFoundError(f"Classification metadata file not found: {class_metadata_path}")
with open(class_metadata_path, 'r') as f:
    content = f.read().strip()
    if not content:
        raise ValueError("Classification metadata JSON file is empty")
    class_metadata = json.loads(content)

class_model_path = os.path.join(MODEL_DIR, f'classification_model_{TIMESTAMP}.pkl')
if not os.path.exists(class_model_path):
    raise FileNotFoundError(f"Classification model file not found: {class_model_path}")
class_dict = joblib.load(class_model_path)
class_model = class_dict['model']
class_threshold = class_dict['threshold']  # Uses optimal Precision-Recall threshold (e.g., 0.400)
class_features = class_metadata['features']

# Ensure features match
if set(reg_features) != set(class_features):
    raise ValueError("Feature mismatch between regression and classification models!")
ALL_FEATURES = reg_features

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'period_data' not in data:
            return jsonify({'error': 'Missing "period_data" key. Provide list of dicts with exogenous vars.'}), 400
        
        period_data = data['period_data']
        period = data.get('period', 4)
        if len(period_data) != period:
            return jsonify({'error': f'period_data must have exactly {period} entries.'}), 400
        
        task = data.get('task', 'both').lower()
        
        # Generate week_starts
        start_date = datetime(2025, 10, 4)
        week_starts = [start_date + timedelta(weeks=i) for i in range(period)]
        
        # Prepare DataFrame
        df_list = []
        for i, (week_start, row_data) in enumerate(zip(week_starts, period_data)):
            row = {'week_start': week_start}
            for key in ['temp_c', 'rh_pct', 'rain_mm', 'wind10_kmh', 'soil_moisture_top_m3m3']:
                row[key] = row_data.get(key, 0)
            row['Combined positive'] = row_data.get('Combined_positive', np.nan)
            df_list.append(row)
        
        df = pd.DataFrame(df_list)
        
        # Compute static features
        df['month'] = df['week_start'].dt.month
        df['week_of_year'] = df['week_start'].dt.isocalendar().week.astype(float)
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sin_week'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['cos_week'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        df['sin_month_2'] = np.sin(4 * np.pi * df['month'] / 12)
        df['cos_month_2'] = np.cos(4 * np.pi * df['month'] / 12)
        
        # Initialize ratio features
        df['ratio'] = 1.0
        df['ratio_lag_1'] = df['ratio'].shift(1).fillna(1.0)
        df['ratio_lag_2'] = df['ratio'].shift(2).fillna(1.0)
        
        # Sequential prediction
        results = {}
        for i in range(len(df)):
            if i == 0:
                df.loc[i, 'Combined positive'] = df.loc[i, 'Combined positive'] if not pd.isna(df.loc[i, 'Combined positive']) else 0
            else:
                if task in ['regression', 'both'] and 'regression' in results and len(results['regression']) > i-1:
                    df.loc[i, 'Combined positive'] = results['regression'][i-1]
                else:
                    df.loc[i, 'Combined positive'] = 0
            
            # Recompute rolling features
            for window in [4, 8, 12]:
                if len(df) >= window:
                    df[f'Combined_positive_roll_mean_{window}'] = df['Combined positive'].shift(1).rolling(window=window, min_periods=1).mean().fillna(0)
                    df[f'Combined_positive_roll_std_{window}'] = df['Combined positive'].shift(1).rolling(window=window, min_periods=1).std().fillna(0)
                else:
                    df[f'Combined_positive_roll_mean_{window}'] = 0
                    df[f'Combined_positive_roll_std_{window}'] = 0
            
            # Recompute interactions
            df['rain_soil_interaction'] = df['rain_mm'] * df['soil_moisture_top_m3m3']
            df['temp_rh_interaction'] = df['temp_c'] * df['rh_pct']
            df['temp_combined_interaction'] = df['temp_c'] * df['Combined positive']
            
            # Update ratio for classification
            if i < len(df) - 1 and task in ['regression', 'both'] and 'regression' in results and len(results['regression']) > i:
                df.loc[i, 'ratio'] = results['regression'][i] / df.loc[i, 'Combined positive'] if df.loc[i, 'Combined positive'] != 0 else 1.0
                df['ratio_lag_1'] = df['ratio'].shift(1).fillna(1.0)
                df['ratio_lag_2'] = df['ratio'].shift(2).fillna(1.0)
            
            # Select features
            X = df[ALL_FEATURES].astype(float)
            
            # Predict
            if task in ['regression', 'both']:
                reg_preds = reg_model.predict(X.iloc[[i]]).tolist()
                results.setdefault('regression', []).extend(reg_preds)
            
            if task in ['classification', 'both']:
                class_probs = class_model.predict_proba(X.iloc[[i]])[:, 1]
                class_preds = (class_probs >= class_threshold).astype(int).tolist()
                results.setdefault('classification', []).extend(class_preds)
                results.setdefault('class_probabilities', []).extend(class_probs.tolist())
        
        results['threshold'] = class_threshold if task in ['classification', 'both'] else None
        
        return jsonify({
            'period_weeks': period,
            'week_starts': [ws.strftime('%Y-%m-%d') for ws in week_starts],
            'predictions': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)