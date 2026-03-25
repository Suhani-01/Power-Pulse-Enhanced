"""
utils/data_processor.py
=======================
Visualization utilities for the PowerPulse Flask app.
Creates enhanced matplotlib plots saved to static/.
"""

import matplotlib
matplotlib.use('Agg')   # non-interactive backend required for Flask
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime


class DataProcessor:

    # Color palette for up to 6 regions
    COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    def __init__(self):
        plt.style.use('seaborn-v0_8')

    # ──────────────────────────────────────────────
    # Main forecast plot
    # ──────────────────────────────────────────────

    def create_enhanced_plot(self, hourly_predictions, selected_regions, selected_date):
        """
        Create an enhanced hourly demand line chart.

        Parameters
        ----------
        hourly_predictions : dict
            { region: [ {'hour': int, 'predicted_demand': float, ...}, ... ] }
        selected_regions   : list of str
        selected_date      : str  'YYYY-MM-DD'

        Returns
        -------
        str  path to saved PNG  ('static/predicted_demand_plot.png')
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 7))

            for i, region in enumerate(selected_regions):
                color   = self.COLORS[i % len(self.COLORS)]
                hours   = list(range(len(hourly_predictions[region])))
                demands = [hourly_predictions[region][h]['predicted_demand']
                           for h in hours]

                # Line + markers
                ax.plot(hours, demands,
                        marker='o', label=region,
                        linewidth=2.5, color=color,
                        markersize=5, markerfacecolor='white',
                        markeredgecolor=color, markeredgewidth=2)

                # Annotate peak value
                pk_val  = max(demands)
                pk_hour = demands.index(pk_val)
                ax.annotate(
                    f'{region}\n{pk_val:.0f} MW',
                    xy=(pk_hour, pk_val),
                    xytext=(pk_hour, pk_val + max(demands) * 0.05),
                    arrowprops=dict(arrowstyle='->', color=color, alpha=0.6),
                    fontsize=8, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor=color, alpha=0.15),
                )

            ax.set_title(
                f'Electricity Demand Forecast — {selected_date}',
                fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel('Hour of Day', fontsize=12)
            ax.set_ylabel('Demand (MW)', fontsize=12)
            ax.set_xticks(range(24))
            ax.set_xticklabels(
                [f'{h:02d}:00' for h in range(24)],
                rotation=45, fontsize=8)
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                      frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_facecolor('#f8f9fa')

            plt.tight_layout()
            os.makedirs('static', exist_ok=True)
            out = 'static/predicted_demand_plot.png'
            plt.savefig(out, dpi=150, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close()
            return out

        except Exception as e:
            print(f'Plot error: {e}')
            plt.close()
            return None

    # ──────────────────────────────────────────────
    # Excel export
    # ──────────────────────────────────────────────

    def export_to_excel(self, predictions_data, selected_date, regions):
        """
        Export hourly predictions to an Excel file with two sheets:
        'Hourly_Predictions' and 'Statistics'.
        """
        try:
            rows = [
                {'Time': f'{h:02d}:00',
                 **{r: round(predictions_data.get(r, {}).get(h, 0), 2)
                    for r in regions}}
                for h in range(24)
            ]
            df = pd.DataFrame(rows)

            ts  = datetime.now().strftime('%H%M%S')
            out = f'static/forecast_{selected_date}_{ts}.xlsx'
            os.makedirs('static', exist_ok=True)

            with pd.ExcelWriter(out, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Hourly_Predictions', index=False)

                stats = [
                    {
                        'Region':    r,
                        'Peak_MW':   max(v for h, v in
                                        predictions_data.get(r, {}).items()),
                        'Min_MW':    min(v for h, v in
                                        predictions_data.get(r, {}).items()),
                        'Avg_MW':    round(np.mean(
                                        list(predictions_data.get(r, {}).values())),
                                        2),
                    }
                    for r in regions
                ]
                pd.DataFrame(stats).to_excel(
                    writer, sheet_name='Statistics', index=False)

            return out

        except Exception as e:
            print(f'Excel export error: {e}')
            return None
