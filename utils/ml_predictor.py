import numpy as np
import pandas as pd
import joblib
import os

class MLPredictor:
    # Ensemble weights: 45% XGBoost, 45% LightGBM, 10% Ridge
    W_XGB   = 0.45
    W_LGB   = 0.45
    W_RIDGE = 0.10

    # Lists of features used for training and prediction
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
        self.models    = {}  # Dictionary to hold models; loaded only when needed
        self._verify_files() # Check if .pkl files exist before starting

    def _verify_files(self):
        """Checks disk for model files without loading them into memory."""
        for region in self.regions:
            xgb_path   = os.path.join(self.model_dir, f'xgb_{region}.pkl')
            lgb_path   = os.path.join(self.model_dir, f'lgb_{region}.pkl')
            ridge_path = os.path.join(self.model_dir, f'ridge_{region}.pkl')

            if all(os.path.exists(p) for p in [xgb_path, lgb_path, ridge_path]):
                self.models[region] = None  # Mark as available
                print(f'Detected models for: {region}')
            else:
                print(f'⚠️ Missing model files for {region}')

    def _load_region(self, region):
        """Loads model files into RAM for a specific region if not already loaded."""
        if self.models.get(region) is not None:
            return  # Skip if already in memory

        try:
            self.models[region] = {
                'xgb':   joblib.load(os.path.join(self.model_dir, f'xgb_{region}.pkl')),
                'lgb':   joblib.load(os.path.join(self.model_dir, f'lgb_{region}.pkl')),
                'ridge': joblib.load(os.path.join(self.model_dir, f'ridge_{region}.pkl')),
            }
            print(f'✅ Successfully loaded {region} models into memory.')
        except Exception as e:
            print(f'❌ Error loading models for {region}: {e}')
            self.models[region] = None

    @staticmethod
    def _season(month):
        """Maps month integer to a seasonal index (0=Winter, 1=Spring, 2=Summer, 3=Autumn)."""
        if month in [12, 1, 2]:  return 0
        elif month in [3, 4, 5]: return 1
        elif month in [6, 7, 8]: return 2
        else:                    return 3

    def build_feature_vector(self, date_obj, hour, minute,
                              temperature, apparent_temperature,
                              humidity, wind_speed,
                              precipitation=0.0, cloud_total=50.0,
                              cloud_low=20.0, cloud_mid=15.0, cloud_high=10.0):
        """Transforms raw inputs into the specific format the ML models expect."""
        dow     = date_obj.weekday()
        doy     = date_obj.timetuple().tm_yday
        month   = date_obj.month
        woy     = int(date_obj.isocalendar()[1])
        weekend = int(dow >= 5)
        quarter = (month - 1) // 3 + 1
        season  = self._season(month)

        vec = np.array([
            # Weather raw data
            temperature, humidity, apparent_temperature,
            precipitation, wind_speed,
            cloud_total, cloud_low, cloud_mid, cloud_high,

            # Time features & Cyclic encoding (sin/cos ensures 23:00 is close to 00:00)
            hour, minute, dow, doy, month, woy, weekend, quarter, season,
            np.sin(2*np.pi*hour/24),   np.cos(2*np.pi*hour/24),
            np.sin(2*np.pi*month/12),  np.cos(2*np.pi*month/12),
            np.sin(2*np.pi*dow/7),     np.cos(2*np.pi*dow/7),
            np.sin(2*np.pi*minute/60), np.cos(2*np.pi*minute/60),

            # Engineered Interaction features
            temperature * humidity / 100,
            temperature ** 2,
            apparent_temperature - temperature,

        ], dtype=np.float32)

        return vec.reshape(1, -1)

    def predict(self, region, date_obj, hour, minute,
                temperature, apparent_temperature, humidity, wind_speed,
                precipitation=0.0, cloud_total=50.0,
                cloud_low=20.0, cloud_mid=15.0, cloud_high=10.0):
        """Generates a weighted ensemble prediction for a specific region."""
        try:
            if region not in self.models:
                raise ValueError(f'Models not available for region: {region}')

            # Load models now if they weren't loaded during startup (Lazy Loading)
            self._load_region(region)

            if self.models[region] is None:
                raise ValueError(f'Failed to load models for region: {region}')

            # Process input into model-ready DataFrame
            X_array = self.build_feature_vector(
                date_obj=date_obj, hour=hour, minute=minute,
                temperature=temperature, apparent_temperature=apparent_temperature,
                humidity=humidity, wind_speed=wind_speed,
                precipitation=precipitation, cloud_total=cloud_total,
                cloud_low=cloud_low, cloud_mid=cloud_mid, cloud_high=cloud_high,
            )
            X = pd.DataFrame(X_array, columns=self.ALL_FEATURES)

            m = self.models[region]

            # Individual model predictions
            xgb_pred   = float(m['xgb'].predict(X)[0])
            lgb_pred   = float(m['lgb'].predict(X)[0])
            
            # Ridge requires scaling before prediction
            X_sc       = m['ridge']['scaler'].transform(X)
            ridge_pred = float(m['ridge']['model'].predict(X_sc)[0])

            # Calculate weighted average (Ensemble)
            ensemble = (
                self.W_XGB * xgb_pred +
                self.W_LGB * lgb_pred +
                self.W_RIDGE * ridge_pred
            )

            # Confidence score based on how much XGB and LGB agree
            diff = abs(xgb_pred - lgb_pred)
            confidence = float(min(0.99, max(0.5, 1.0 - diff/(abs(ensemble)+1e-5))))

            return {
                'xgb':        round(xgb_pred, 2),
                'lgb':        round(lgb_pred, 2),
                'ridge':      round(ridge_pred, 2),
                'ensemble':   round(ensemble, 2),
                'confidence': round(confidence, 3),
            }

        except Exception as e:
            print(f'Prediction error [{region}]: {e}')
            # Use simple math-based fallback if ML models fail
            fb = self._fallback(temperature, hour, date_obj.month)
            return {
                'xgb': fb, 'lgb': fb, 'ridge': fb,
                'ensemble': fb, 'confidence': 0.5
            }

    def predict_all_regions(self, date_obj, hour, minute,
                            temperature, apparent_temperature,
                            humidity, wind_speed, **kwargs):
        """Runs the prediction loop for every region in the list."""
        return {
            r: self.predict(
                region=r, date_obj=date_obj,
                hour=hour, minute=minute,
                temperature=temperature,
                apparent_temperature=apparent_temperature,
                humidity=humidity, wind_speed=wind_speed,
                **kwargs
            )['ensemble']
            for r in self.regions
        }

    @staticmethod
    def _fallback(temperature, hour, month):
        """Basic heuristic to estimate demand when ML models are unavailable."""
        base = 2500
        if month in [6,7,8]:   base += 1500 # Summer peak
        elif month in [12,1,2]: base -= 600 # Winter lower
        
        # Adjust based on temperature extremes
        if temperature > 38:   base += int((temperature-38)*80)
        elif temperature < 12: base += int((12-temperature)*50)
        
        # Adjust for night vs day usage patterns
        if 0 <= hour < 5:       base -= 800
        elif 14 <= hour < 21:   base += 500
        
        return max(500, base)