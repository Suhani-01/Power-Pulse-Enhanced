"""
app.py
======
PowerPulse — Flask Application Entry Point
XGBoost + LightGBM + Ridge electricity demand forecasting
"""

import os
# Suppress TensorFlow logging to keep the console clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from datetime import datetime

# Import custom utility modules for ML, Weather, and Data Processing
from utils.ml_predictor import MLPredictor
from utils.weather_service import WeatherService
from utils.data_processor import DataProcessor


# ── App initialisation ──────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'powerpulse-secret-2024'
# Disable browser caching for development/testing accuracy
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


# ── Load services ONLY ONCE ─────────────────────────────────────
# We initialize services globally so they persist across requests
print("🚀 Loading models and services...")

ml_predictor    = MLPredictor(model_dir='saved_models')
weather_service = WeatherService()
data_processor  = DataProcessor()

if not ml_predictor.models:
    print("⚠️ Models not loaded — using fallback predictions")

print("✅ All services loaded successfully!")


# List of valid electricity distribution regions in Delhi
REGIONS = ['DELHI', 'BRPL', 'BYPL', 'NDPL', 'NDMC', 'MES']


# ════════════════════════════════════════════════════════════════
# Routes
# ════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Renders the landing page."""
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Simple session-based login logic."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        # Basic hardcoded credentials check
        if username == 'user' and password == 'password':
            session["isLoggedIn"]=True
            return redirect(url_for('forecast'))

        return render_template('login.html',message='Invalid username or password')

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Clears user session and redirects to home."""
    session.clear()
    return redirect(url_for('index'))


@app.route('/map')
def map():
    """Renders the regional map view."""
    return render_template('map.html')


# ── Forecast ─────────────────────────────────────────────────────
@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    """
    Main Forecast Logic:
    1. POST: Captures user selections (regions/date) and stores them in session.
    2. GET: Fetches weather, runs ML predictions for 24 hours, and renders results.
    """
    if not session.get('isLoggedIn'):
       return render_template('login.html',message='You need to log in first')

    print(" Forecasting .....")
    predictions = []
    peak_least_demand_info = []
    plot_filename = None
    selected_regions=[]
    selected_date=None

    # 🔥 STEP 1: POST → Save selections and redirect to prevent double-submit
    if request.method == 'POST':
        selected_regions = request.form.getlist('region')
        selected_date = request.form.get('date')

        print(f" Selected region : {selected_regions} and selected date : {selected_date}\n")

        if not selected_regions:
            flash('Please select at least one region.', 'warning')
            return redirect(url_for('forecast'))

        try:
            date_obj = datetime.strptime(selected_date, '%Y-%m-%d')

            # Fetch weather data for the selected date
            print(f"📡 Fetching weather for {selected_date}...")
            weather_data = weather_service.fetch_weather_forecast(date_obj)

            if weather_data is None:
                flash('Weather fetch failed.', 'error')
                return render_template(
                    'forecast.html',
                    regions=REGIONS,
                    selected_date=selected_date
                )

            hourly_predictions = {r: [] for r in selected_regions}

            # Loop through 24 hours of weather data and predict for each region
            print(f"⚡ Running predictions for {selected_regions}...")
            for _, row in weather_data.iterrows():
                hour = int(row['hour'])

                for region in selected_regions:
                    try:
                        # Call the ML predictor with weather features
                        pred = ml_predictor.predict(
                            region=region,
                            date_obj=date_obj,
                            hour=hour,
                            minute=0,
                            temperature=float(row['temperature_2m']),
                            apparent_temperature=float(row['apparent_temperature']),
                            humidity=float(row['relative_humidity_2m']),
                            wind_speed=float(row['wind_speed_10m']),
                            precipitation=float(row.get('precipitation', 0.0)),
                            cloud_total=float(row.get('cloud_cover', 50.0)),
                            cloud_low=float(row.get('cloud_cover_low', 20.0)),
                            cloud_mid=float(row.get('cloud_cover_mid', 15.0)),
                            cloud_high=float(row.get('cloud_cover_high', 10.0)),
                        )

                        hourly_predictions[region].append({
                            'hour': hour,
                            'predicted_demand': pred['ensemble'],
                            'xgb': pred['xgb'],
                            'lgb': pred['lgb'],
                            'ridge': pred['ridge'],
                            'confidence': pred['confidence'],
                        })

                    except Exception as e:
                        print(f'❌ Error [{region} H{hour}]: {e}')
                        # Fallback row if a specific hour fails
                        hourly_predictions[region].append({
                            'hour': hour, 'predicted_demand': 0, 'confidence': 0.5
                        })

            # Format data for the HTML Table
            for hour in range(24):
                row_data = {'time': f'{hour:02d}:00'}
                for region in selected_regions:
                    row_data[region] = round(
                        hourly_predictions[region][hour]['predicted_demand'], 2
                    )
                predictions.append(row_data)

            # Calculate Highs (Peak) and Lows (Least) for the dashboard summary
            for region in selected_regions:
                items = hourly_predictions[region]
                demands = [x['predicted_demand'] for x in items]

                peak_d = max(demands)
                least_d = min(demands)

                peak_hr = items[demands.index(peak_d)]['hour']
                least_hr = items[demands.index(least_d)]['hour']

                avg_conf = round(
                    sum(x['confidence'] for x in items) / len(items) * 100, 1
                )

                peak_least_demand_info.append({
                    'region': region,
                    'peak_demand': round(peak_d, 2),
                    'least_demand': round(least_d, 2),
                    'peak_hour': f'{peak_hr:02d}:00',
                    'least_hour': f'{least_hr:02d}:00',
                    'confidence': avg_conf,
                })

            # Generate and save a Matplotlib/Seaborn visualization
            plot_filename = data_processor.create_enhanced_plot(
                hourly_predictions, selected_regions, selected_date
            )

            flash('Predictions generated!', 'success')

        except Exception as e:
            print(f'🔥 Forecast error: {e}')
            flash('Something went wrong.', 'error')

        return render_template(
        'forecast.html',
        predictions=predictions,
        peak_least_demand_info=peak_least_demand_info,
        plot_filename=plot_filename,
        selected_date=selected_date,
        selected_regions=selected_regions,
        regions=REGIONS,
        )

    return render_template('forecast.html')


    


# ── API ─────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for single-point external predictions."""
    try:
        data = request.get_json()
        date_obj = datetime.strptime(data.get('date'), '%Y-%m-%d')

        pred = ml_predictor.predict(
            region=data['region'],
            date_obj=date_obj,
            hour=int(data.get('hour', 12)),
            minute=0,
            temperature=float(data['temperature']),
            apparent_temperature=float(data['temperature']),
            humidity=float(data['humidity']),
            wind_speed=float(data['wind_speed']),
        )

        return jsonify({
            'success': True,
            'prediction': pred
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ── Cache Fix ───────────────────────────────────────────────────
@app.after_request
def add_header(response):
    """Force browsers to always fetch the latest results/plots."""
    response.cache_control.no_store = True
    response.cache_control.no_cache = True
    response.cache_control.must_revalidate = True
    response.cache_control.max_age = 0
    return response


# ════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Ensure static directory exists for plot storage
    os.makedirs('static', exist_ok=True)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Re-loader disabled to prevent duplicate model loading
    )