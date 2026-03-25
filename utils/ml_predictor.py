"""
utils/ml_predictor.py
=====================
ML Prediction Service — XGBoost + LSTM (Keras)
Models trained on 26 features: 9 weather + 17 time (NO lag features)
Regions: DELHI, BRPL, BYPL, NDPL, NDMC, MES
"""

import numpy as np
import joblib
import os
from datetime import datetime

# Keras import with fallback
try:
    from tensorflow import keras
except ImportError:
    import keras


class MLPredictor:
    """
    Loads XGBoost (.pkl) and LSTM (.keras) models for all 6 Delhi regions.
    Uses only weather + time features — no historical lag data required.
    """

    # ── Exact feature order used during training (DO NOT change order) ──
    WEATHER_FEATURES = [
        'Temperature (°C)',
        'Relative Humidity (%)',
        'Apparent Temperature (°C)',
        'Precipitation (mm)',
        'Wind Speed (m/s)',
        'Cloud Cover Total (%)',
        'Cloud Cover Low (%)',
        'Cloud Cover Mid (%)',
        'Cloud Cover High (%)',
    ]

    TIME_FEATURES = [
        'Hour', 'Minute', 'DayOfWeek', 'DayOfYear', 'Month',
        'WeekOfYear', 'IsWeekend', 'Quarter', 'Season',
        'Hour_sin', 'Hour_cos',
        'Month_sin', 'Month_cos',
        'DoW_sin', 'DoW_cos',
        'Min_sin', 'Min_cos',
    ]

    ALL_FEATURES = WEATHER_FEATURES + TIME_FEATURES   # 26 features total
    SEQUENCE_LEN = 12                                  # 12 x 5 min = 60 min

    def __init__(self, model_dir='saved_models'):
        self.model_dir = model_dir
        self.regions   = ['DELHI', 'BRPL', 'BYPL', 'NDPL', 'NDMC', 'MES']
        self.models    = {}   # { region: {'xgb': ..., 'lstm': ...} }
        self.scalers   = {}   # { region: {'X': ..., 'y': ...} }
        self._load_all()

    # ──────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────

    def _load_all(self):
        """Load XGBoost + LSTM models and scalers for every region."""
        for region in self.regions:
            try:
                xgb_path  = os.path.join(self.model_dir, f'xgb_{region}.pkl')
                lstm_path = os.path.join(self.model_dir, f'lstm_{region}.keras')
                sx_path   = os.path.join(self.model_dir, f'lstm_scaler_X_{region}.pkl')
                sy_path   = os.path.join(self.model_dir, f'lstm_scaler_y_{region}.pkl')

                self.models[region] = {
                    'xgb':  joblib.load(xgb_path),
                    'lstm': keras.models.load_model(lstm_path),
                }
                self.scalers[region] = {
                    'X': joblib.load(sx_path),
                    'y': joblib.load(sy_path),
                }
                print(f'Loaded models: {region}')

            except FileNotFoundError as e:
                print(f'Missing model file for {region}: {e}')
            except Exception as e:
                print(f'Error loading models for {region}: {e}')

    # ──────────────────────────────────────────────
    # Feature engineering (mirrors training notebook)
    # ──────────────────────────────────────────────

    @staticmethod
    def _season(month):
        if month in [12, 1, 2]:   return 0   # Winter
        elif month in [3, 4, 5]:  return 1   # Spring
        elif month in [6, 7, 8]:  return 2   # Summer/Monsoon
        else:                     return 3   # Post-monsoon

    def build_feature_vector(self, date_obj, hour, minute,
                              temperature, apparent_temperature,
                              humidity, wind_speed,
                              precipitation=0.0,
                              cloud_total=50.0, cloud_low=20.0,
                              cloud_mid=15.0,  cloud_high=10.0):
        """
        Build the 26-feature numpy vector.
        Returns shape (1, 26) — ready to feed into XGB or scaled for LSTM.
        """
        dow     = date_obj.weekday()
        doy     = date_obj.timetuple().tm_yday
        month   = date_obj.month
        woy     = int(date_obj.isocalendar()[1])
        weekend = int(dow >= 5)
        quarter = (month - 1) // 3 + 1
        season  = self._season(month)

        # Cyclical encodings
        h_sin  = np.sin(2 * np.pi * hour   / 24)
        h_cos  = np.cos(2 * np.pi * hour   / 24)
        m_sin  = np.sin(2 * np.pi * month  / 12)
        m_cos  = np.cos(2 * np.pi * month  / 12)
        d_sin  = np.sin(2 * np.pi * dow    / 7)
        d_cos  = np.cos(2 * np.pi * dow    / 7)
        mi_sin = np.sin(2 * np.pi * minute / 60)
        mi_cos = np.cos(2 * np.pi * minute / 60)

        vec = np.array([
            # 9 weather features
            temperature, humidity, apparent_temperature,
            precipitation, wind_speed,
            cloud_total, cloud_low, cloud_mid, cloud_high,
            # 17 time features
            hour, minute, dow, doy, month,
            woy, weekend, quarter, season,
            h_sin, h_cos,
            m_sin, m_cos,
            d_sin, d_cos,
            mi_sin, mi_cos,
        ], dtype=np.float32)

        return vec.reshape(1, -1)   # (1, 26)

    # ──────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────

    def predict(self, region, date_obj, hour, minute,
                temperature, apparent_temperature,
                humidity, wind_speed,
                precipitation=0.0,
                cloud_total=50.0, cloud_low=20.0,
                cloud_mid=15.0,   cloud_high=10.0):
        """
        Predict electricity demand for one data point.

        Returns dict: { xgb, lstm, ensemble, confidence }
        """
        try:
            if region not in self.models:
                raise ValueError(f'Models not loaded for region: {region}')

            # Build feature vector
            X = self.build_feature_vector(
                date_obj=date_obj, hour=hour, minute=minute,
                temperature=temperature,
                apparent_temperature=apparent_temperature,
                humidity=humidity, wind_speed=wind_speed,
                precipitation=precipitation,
                cloud_total=cloud_total, cloud_low=cloud_low,
                cloud_mid=cloud_mid, cloud_high=cloud_high,
            )  # (1, 26)

            m = self.models[region]

            # XGBoost prediction (direct, no scaling needed)
            xgb_pred = float(m['xgb'].predict(X)[0])

            # LSTM prediction (needs scaling + sequence reshape)
            sx   = self.scalers[region]['X']
            sy   = self.scalers[region]['y']
            X_sc = sx.transform(X)                           # (1, 26) scaled
            # Repeat row to form sequence: (1, SEQUENCE_LEN, 26)
            X_seq = np.repeat(X_sc[:, np.newaxis, :],
                              self.SEQUENCE_LEN, axis=1)
            y_sc      = m['lstm'].predict(X_seq, verbose=0)
            lstm_pred = float(sy.inverse_transform(y_sc)[0][0])

            # Ensemble average
            ensemble = (xgb_pred + lstm_pred) / 2.0

            # Confidence: how close are the two models?
            diff       = abs(xgb_pred - lstm_pred)
            confidence = float(min(0.99, max(0.5,
                            1.0 - diff / (abs(ensemble) + 1e-5))))

            return {
                'xgb':        round(xgb_pred,  2),
                'lstm':       round(lstm_pred,  2),
                'ensemble':   round(ensemble,   2),
                'confidence': round(confidence, 3),
            }

        except Exception as e:
            print(f'Prediction error [{region}]: {e}')
            fallback = self._fallback_demand(
                temperature, hour, date_obj.month)
            return {
                'xgb': fallback, 'lstm': fallback,
                'ensemble': fallback, 'confidence': 0.5,
            }

    def predict_all_regions(self, date_obj, hour, minute,
                            temperature, apparent_temperature,
                            humidity, wind_speed, **kwargs):
        """Predict ensemble demand for all regions at once."""
        return {
            region: self.predict(
                region=region,
                date_obj=date_obj, hour=hour, minute=minute,
                temperature=temperature,
                apparent_temperature=apparent_temperature,
                humidity=humidity, wind_speed=wind_speed,
                **kwargs,
            )['ensemble']
            for region in self.regions
        }

    # ──────────────────────────────────────────────
    # Fallback
    # ──────────────────────────────────────────────

    @staticmethod
    def _fallback_demand(temperature, hour, month):
        """Simple heuristic when models fail to load."""
        base = 2500
        if month in [6, 7, 8]:    base += 1500
        elif month in [12, 1, 2]: base -= 600
        if temperature > 38:      base += int((temperature - 38) * 80)
        elif temperature < 12:    base += int((12 - temperature) * 50)
        if 0 <= hour < 5:         base -= 800
        elif 14 <= hour < 21:     base += 500
        return max(500, base)
