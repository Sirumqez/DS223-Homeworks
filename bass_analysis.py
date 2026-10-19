"""
Marketing Analytics Assignment: Innovation Diffusion Analysis Based on Bass Model
Target Innovation: Lotus Ring (TIME's Best Inventions of 2025)
Data Source: Statista - Worldwide Smart Wearables Shipments Forecast (2016-2026)
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# 1. Ensure required directories exist for submission structure
os.makedirs('img', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('report', exist_ok=True)

# 2. Load and clean dataset
csv_path = 'data/smart_wearable_shipments_forecast_worldwide_from_2016_to_2026_in_millions.csv'

# Fallback check if CSV is in current root directory
if not os.path.exists(csv_path) and os.path.exists('smart_wearable_shipments_forecast_worldwide_from_2016_to_2026_in_millions.csv'):
    import shutil
    shutil.copy('smart_wearable_shipments_forecast_worldwide_from_2016_to_2026_in_millions.csv', csv_path)

df = pd.read_csv(csv_path)
df['Year'] = df['Unnamed: 0'].str.replace('*', '', regex=False).astype(int)
df['Shipments'] = df['Shipments in million units']
df_clean = df[['Year', 'Shipments']].sort_values('Year').reset_index(drop=True)
df_clean['Cumulative_Shipments'] = df_clean['Shipments'].cumsum()

print("=== Cleaned Historical Data ===")
print(df_clean.to_string(index=False))

# 3. Define Bass Model Cumulative Function
t = np.arange(len(df_clean))
N = df_clean['Cumulative_Shipments'].values

def bass_cumulative(t, M, p, q):
    exp_term = np.exp(-(p + q) * t)
    return M * (1 - exp_term) / (1 + (q / p) * exp_term)

# 4. Estimate Bass Model Parameters (M, p, q)
popt, pcov = curve_fit(
    bass_cumulative, t, N,
    p0=[max(N)*3, 0.03, 0.3],
    bounds=([max(N), 0.0001, 0.0001], [max(N)*50, 1.0, 1.0])
)
M_est, p_est, q_est = popt

print("\n=== Estimated Bass Model Parameters ===")
print(f"Market Potential (M): {M_est:.2f} million units")
print(f"Coefficient of Innovation (p): {p_est:.5f}")
print(f"Coefficient of Imitation (q): {q_est:.5f}")

# 5. Predict / Forecast Diffusion Path (2016 - 2030)
years_forecast = np.arange(2016, 2031)
t_forecast = np.arange(len(years_forecast))
N_pred = bass_cumulative(t_forecast, M_est, p_est, q_est)

S_pred = np.zeros_like(N_pred)
S_pred[0] = N_pred[0]
S_pred[1:] = np.diff(N_pred)

forecast_df = pd.DataFrame({
    'Year': years_forecast,
    'Cumulative_Adopters_Millions': N_pred,
    'Annual_Adopters_Millions': S_pred
})

print("\n=== Diffusion Forecast Table (2016 - 2030) ===")
print(forecast_df.to_string(index=False))

# 6. Generate and Save Plot to img/diffusion_plot.png
plt.figure(figsize=(10, 6))
plt.plot(df_clean['Year'], df_clean['Cumulative_Shipments'], 'bo-', label='Historical Smart Wearables (Statista)', linewidth=2, markersize=6)
plt.plot(years_forecast, N_pred, 'r--', label=f'Bass Model Fit (M={M_est:.1f}M)', linewidth=2)
plt.axvline(x=2025, color='gray', linestyle=':', label='Lotus Ring Launch (2025)')
plt.title('Innovation Diffusion Analysis: Bass Model vs Historical Smart Wearables', fontsize=13, fontweight='bold')
plt.xlabel('Year', fontsize=11)
plt.ylabel('Cumulative Shipments / Adopters (Millions)', fontsize=11)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Explicitly save image to the required img directory
output_img_path = 'img/diffusion_plot.png'
plt.savefig(output_img_path, dpi=300)
plt.close()

print(f"\nSuccess! Plot generated and saved to '{output_img_path}'.")