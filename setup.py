"""
setup.py
========
PowerPulse — Project Setup Script
Verifies model files, checks dependencies, prints project structure.
Run once after training: python setup.py
"""

import os
import shutil


def setup_project():
    print('🚀 PowerPulse Setup\n')

    # ── Create required directories ──
    for d in ['static', 'static/plots', 'saved_models', 'utils', 'templates']:
        os.makedirs(d, exist_ok=True)
        print(f'  Directory ready : {d}')

    print()

    # ── Expected model files (from Jupyter notebook output) ──
    REGIONS = ['DELHI', 'BRPL', 'BYPL', 'NDPL', 'NDMC', 'MES']

    # Each region needs these 4 files
    model_files = []
    for region in REGIONS:
        model_files += [
            f'xgb_{region}.pkl',            # XGBoost model
            f'lstm_{region}.keras',          # LSTM model
            f'lstm_scaler_X_{region}.pkl',   # feature scaler
            f'lstm_scaler_y_{region}.pkl',   # target scaler
        ]

    # Source: notebook saves to saved_models/ — copy from there or parent dir
    source_dir = os.path.join('..', 'saved_models')

    print('── Model files ────────────────────────────────────')
    ok = 0
    for fname in model_files:
        dest = os.path.join('saved_models', fname)
        if os.path.exists(dest):
            size = os.path.getsize(dest)
            s    = f'{size/1024:.0f} KB' if size < 1024**2 else f'{size/1024**2:.1f} MB'
            print(f'  ✅  {fname:<45} {s}')
            ok += 1
        else:
            src = os.path.join(source_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dest)
                print(f'  ✅  {fname:<45} (copied from {source_dir})')
                ok += 1
            else:
                print(f'  ⚠️   {fname:<45} NOT FOUND — copy to saved_models/')

    print(f'\n  {ok}/{len(model_files)} model files present')

    # ── Dependency check ──
    print('\n── Python packages ────────────────────────────────')
    packages = {
        'flask':      'flask',
        'pandas':     'pandas',
        'numpy':      'numpy',
        'sklearn':    'scikit-learn',
        'tensorflow': 'tensorflow',
        'xgboost':    'xgboost',
        'matplotlib': 'matplotlib',
        'seaborn':    'seaborn',
        'joblib':     'joblib',
        'requests':   'requests',
        'openpyxl':   'openpyxl',
    }
    missing = []
    for import_name, pip_name in packages.items():
        try:
            __import__(import_name)
            print(f'  ✅  {pip_name}')
        except ImportError:
            print(f'  ❌  {pip_name}  ← MISSING')
            missing.append(pip_name)

    if missing:
        print(f'\n  Install: pip install {" ".join(missing)}')
    else:
        print('\n  All packages installed ✅')

    # ── Project structure ──
    print("""
── Expected project structure ─────────────────────────
  PowerPulse/
  ├── app.py                      ← Flask entry point
  ├── setup.py                    ← this file
  ├── saved_models/
  │   ├── xgb_DELHI.pkl           ← XGBoost (6 regions)
  │   ├── lstm_DELHI.keras        ← LSTM     (6 regions)
  │   ├── lstm_scaler_X_DELHI.pkl ← feature scaler (6)
  │   └── lstm_scaler_y_DELHI.pkl ← target scaler  (6)
  ├── utils/
  │   ├── ml_predictor.py
  │   ├── weather_service.py
  │   └── data_processor.py
  ├── templates/
  │   ├── index.html
  │   ├── forecast.html
  │   ├── login.html
  │   └── map.html
  └── static/
──────────────────────────────────────────────────────
  Run : python app.py
  Open: http://localhost:5000
""")


if __name__ == '__main__':
    setup_project()
