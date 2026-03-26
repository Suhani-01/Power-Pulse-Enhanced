"""
utils/ml_predictor.py - v3
XGBoost + LightGBM + Ridge Regression Ensemble
No TensorFlow/Keras — all .pkl files, deploy friendly
29 features: 9 weather + 17 time + 3 interaction
"""

import numpy as np
import joblib
import os


class MLPredictor:

    W_XGB   = 0.45
    W_LGB   = 0.45
    W_RIDGE = 0.10

    WEATHER_FEATURES = [
        'Temperature (°C)', 'Relative Humidity (%)', 'Apparent Temperature (°C)',
        'Precipitation (mm)', 'Wind Speed (m/s)', 'Cloud Cover Total (%)',
        'Cloud Cover Low (%)', 'Cloud Cover Mid (%)', 'Cloud Cover High (%)',
    ]
    TIME_FEATURES = [
        'Hour', 'Minute', 'DayOfWeek', 'DayOfYear', 'Month',
        'WeekOfYear', 'IsWeekend', 'Quarter', 'Season',
        'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos',
        'DoW_sin', 'DoW_cos', 'Min_sin', 'Min_cos',
    ]
    INTERACTION_FEATURES = ['Temp_Humidity', 'Temp_squared', 'Heat_Index']
    ALL_FEATURES = WEATHER_FEATURES + TIME_FEATURES + INTERACTION_FEATURES

    def __init__(self, model_dir='saved_models'):
        self.model_dir = model_dir
        self.regions   = ['DELHI', 'BRPL', 'BYPL', 'NDPL', 'NDMC', 'MES']
        self.models    = {}
        self._load_all()

    def _load_all(self):
        for region in self.regions:
            try:
                self.models[region] = {
                    'xgb':   joblib.load(os.path.join(self.model_dir, f'xgb_{region}.pkl')),
                    'lgb':   joblib.load(os.path.join(self.model_dir, f'lgb_{region}.pkl')),
                    'ridge': joblib.load(os.path.join(self.model_dir, f'ridge_{region}.pkl')),
                }
                print(f'Loaded models: {region}')
            except FileNotFoundError as e:
                print(f'Missing model file for {region}: {e}')
            except Exception as e:
                print(f'Error loading models for {region}: {e}')

    @staticmethod
    def _season(month):
        if month in [12, 1, 2]:  return 0
        elif month in [3, 4, 5]: return 1
        elif month in [6, 7, 8]: return 2
        else:                    return 3

    def build_feature_vector(self, date_obj, hour, minute,
                              temperature, apparent_temperature,
                              humidity, wind_speed,
                              precipitation=0.0, cloud_total=50.0,
                              cloud_low=20.0, cloud_mid=15.0, cloud_high=10.0):
        dow     = date_obj.weekday()
        doy     = date_obj.timetuple().tm_yday
        month   = date_obj.month
        woy     = int(date_obj.isocalendar()[1])
        weekend = int(dow >= 5)
        quarter = (month - 1) // 3 + 1
        season  = self._season(month)

        vec = np.array([
            # 9 weather
            temperature, humidity, apparent_temperature,
            precipitation, wind_speed,
            cloud_total, cloud_low, cloud_mid, cloud_high,
            # 17 time
            hour, minute, dow, doy, month, woy, weekend, quarter, season,
            np.sin(2*np.pi*hour/24),   np.cos(2*np.pi*hour/24),
            np.sin(2*np.pi*month/12),  np.cos(2*np.pi*month/12),
            np.sin(2*np.pi*dow/7),     np.cos(2*np.pi*dow/7),
            np.sin(2*np.pi*minute/60), np.cos(2*np.pi*minute/60),
            # 3 interaction
            temperature * humidity / 100,
            temperature ** 2,
            apparent_temperature - temperature,
        ], dtype=np.float32)
        return vec.reshape(1, -1)

    def predict(self, region, date_obj, hour, minute,
                temperature, apparent_temperature, humidity, wind_speed,
                precipitation=0.0, cloud_total=50.0,
                cloud_low=20.0, cloud_mid=15.0, cloud_high=10.0):
        try:
            if region not in self.models:
                raise ValueError(f'Models not loaded for region: {region}')

            X = self.build_feature_vector(
                date_obj=date_obj, hour=hour, minute=minute,
                temperature=temperature, apparent_temperature=apparent_temperature,
                humidity=humidity, wind_speed=wind_speed,
                precipitation=precipitation, cloud_total=cloud_total,
                cloud_low=cloud_low, cloud_mid=cloud_mid, cloud_high=cloud_high,
            )
            m = self.models[region]

            xgb_pred   = float(m['xgb'].predict(X)[0])
            lgb_pred   = float(m['lgb'].predict(X)[0])
            X_sc       = m['ridge']['scaler'].transform(X)
            ridge_pred = float(m['ridge']['model'].predict(X_sc)[0])
            ensemble   = self.W_XGB*xgb_pred + self.W_LGB*lgb_pred + self.W_RIDGE*ridge_pred

            diff       = abs(xgb_pred - lgb_pred)
            confidence = float(min(0.99, max(0.5, 1.0 - diff/(abs(ensemble)+1e-5))))

            return {
                'xgb':        round(xgb_pred,  2),
                'lgb':        round(lgb_pred,  2),
                'ridge':      round(ridge_pred, 2),
                'ensemble':   round(ensemble,   2),
                'confidence': round(confidence, 3),
            }
        except Exception as e:
            print(f'Prediction error [{region}]: {e}')
            fb = self._fallback(temperature, hour, date_obj.month)
            return {'xgb':fb,'lgb':fb,'ridge':fb,'ensemble':fb,'confidence':0.5}

    def predict_all_regions(self, date_obj, hour, minute,
                            temperature, apparent_temperature,
                            humidity, wind_speed, **kwargs):
        return {
            r: self.predict(region=r, date_obj=date_obj, hour=hour, minute=minute,
                            temperature=temperature, apparent_temperature=apparent_temperature,
                            humidity=humidity, wind_speed=wind_speed, **kwargs)['ensemble']
            for r in self.regions
        }

    @staticmethod
    def _fallback(temperature, hour, month):
        base = 2500
        if month in [6,7,8]:    base += 1500
        elif month in [12,1,2]: base -= 600
        if temperature > 38:    base += int((temperature-38)*80)
        elif temperature < 12:  base += int((12-temperature)*50)
        if 0 <= hour < 5:       base -= 800
        elif 14 <= hour < 21:   base += 500
        return max(500, base)
