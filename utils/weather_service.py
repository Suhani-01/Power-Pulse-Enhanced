"""
utils/weather_service.py
========================
Fetches hourly weather for Delhi from Open-Meteo API.
Returns all 9 weather fields matching the training feature set.
Falls back to season-aware synthetic data if API is unavailable.
"""

import requests
import pandas as pd
from datetime import datetime


class WeatherService:
    # Coordinates for Delhi (Central)
    API_URL = 'https://api.open-meteo.com/v1/forecast'
    LAT     = 28.6519
    LON     = 77.2315
    TIMEOUT = 10

    # These match the exact feature names required by the MLPredictor
    HOURLY_PARAMS = ','.join([
        'temperature_2m',
        'apparent_temperature',
        'relative_humidity_2m',
        'wind_speed_10m',
        'precipitation',
        'cloud_cover',
        'cloud_cover_low',
        'cloud_cover_mid',
        'cloud_cover_high',
    ])

    def fetch_weather_forecast(self, selected_date):
        """
        Primary method to get 24-hour weather data for a specific date.
        Tries the live API first; uses _fallback if the API fails or times out.
        """
        try:
            # Request configuration for the Open-Meteo API
            params = {
                'latitude':   self.LAT,
                'longitude':  self.LON,
                'hourly':     self.HOURLY_PARAMS,
                'timezone':   'Asia/Kolkata',
                'start_date': selected_date.strftime('%Y-%m-%d'),
                'end_date':   selected_date.strftime('%Y-%m-%d'),
            }

            resp = requests.get(self.API_URL, params=params, timeout=30)

            if resp.status_code == 200:
                h = resp.json()['hourly']
                # Constructing DataFrame from API dictionary response
                df = pd.DataFrame({
                    'datetime':           pd.to_datetime(h['time']),
                    'temperature_2m':       h['temperature_2m'],
                    'apparent_temperature': h['apparent_temperature'],
                    'relative_humidity_2m': h['relative_humidity_2m'],
                    'wind_speed_10m':       h['wind_speed_10m'],
                    'precipitation':        h['precipitation'],
                    'cloud_cover':          h['cloud_cover'],
                    'cloud_cover_low':      h['cloud_cover_low'],
                    'cloud_cover_mid':      h['cloud_cover_mid'],
                    'cloud_cover_high':     h['cloud_cover_high'],
                })
                # Add integer hour column for model feature alignment
                df['hour'] = df['datetime'].dt.hour
                print(f'Weather API success: {len(df)} records for '
                      f'{selected_date.strftime("%Y-%m-%d")}')
                return df

            # Handle non-200 status codes (e.g., 429 rate limit or 500 server error)
            print(f'Weather API error {resp.status_code} — using fallback')
            return self._fallback(selected_date)

        except requests.exceptions.Timeout:
            print('Weather API timeout — using fallback')
            return self._fallback(selected_date)
        except requests.exceptions.ConnectionError:
            print('Weather API connection error — using fallback')
            return self._fallback(selected_date)
        except Exception as e:
            print(f'Weather API exception: {e} — using fallback')
            return self._fallback(selected_date)

    # ──────────────────────────────────────────────
    # Season-aware synthetic fallback (Delhi climate)
    # ──────────────────────────────────────────────

    def _fallback(self, selected_date):
        """
        Generates mathematically simulated weather based on Delhi's historical averages.
        This ensures the application stays functional even without internet or API access.
        """
        m = selected_date.month

        # Define seasonal baseline values (Temp, Apparent Adj, Humidity, Wind)
        if m in [12, 1, 2]:         # Winter: Cold, high humidity (foggy)
            base_t, app_adj, hum, wind = 13, -4, 74, 3
        elif m in [3, 4, 5]:        # Spring / Pre-summer: Hot and dry
            base_t, app_adj, hum, wind = 31, +5, 44, 7
        elif m in [6, 7, 8, 9]:     # Monsoon: Hot, very humid, and windy
            base_t, app_adj, hum, wind = 33, +3, 83, 9
        else:                       # Post-monsoon: Moderate
            base_t, app_adj, hum, wind = 26, +1, 57, 5

        rows = []
        for hr in range(24):
            # Diurnal Temperature Curve: Mimics sun rising and setting
            if hr < 5:
                t = base_t - 6 + hr * 0.3      # Cooling down overnight
            elif hr < 14:
                t = base_t - 4 + (hr - 5) * 2.0 # Warming up toward afternoon
            else:
                t = base_t + 12 - (hr - 14) * 1.5 # Gradual cooling after 2 PM

            # Keep values within realistic physical bounds
            t  = round(max(-5.0, min(48.0, t)), 1)
            at = round(t + app_adj, 1)
            # Humidity drops as temperature rises
            h  = round(max(20.0, min(95.0, hum - (t - base_t) * 1.2)), 1)
            ws = round(max(0.0, wind + (hr / 12.0 - 1) * 2.0), 1)
            cc = round(max(0.0, min(100.0, 35 + hr * 1.5)), 1)

            rows.append({
                'datetime':             selected_date.replace(
                                            hour=hr, minute=0,
                                            second=0, microsecond=0),
                'temperature_2m':       t,
                'apparent_temperature': at,
                'relative_humidity_2m': h,
                'wind_speed_10m':       ws,
                'precipitation':        0.0,
                'cloud_cover':          cc,
                'cloud_cover_low':      round(cc * 0.40, 1),
                'cloud_cover_mid':      round(cc * 0.35, 1),
                'cloud_cover_high':     round(cc * 0.25, 1),
                'hour':                 hr,
            })

        print(f'Using fallback weather for {selected_date.strftime("%Y-%m-%d")}')
        return pd.DataFrame(rows)