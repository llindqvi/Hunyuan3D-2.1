import pandas as pd

df = pd.read_csv('vram_usage.log')
df['Used_MB'] = pd.to_numeric(df['Used_MB'])
df['Total_MB'] = pd.to_numeric(df['Total_MB'])

print("="*60)
print("VRAM USAGE ANALYSIS")
print("="*60)
print(f"Peak VRAM Usage: {df['Used_MB'].max():.0f} MB ({df['Used_MB'].max()/1024:.2f} GB)")
print(f"Average VRAM Usage: {df['Used_MB'].mean():.0f} MB ({df['Used_MB'].mean()/1024:.2f} GB)")
print(f"Min VRAM Usage: {df['Used_MB'].min():.0f} MB ({df['Used_MB'].min()/1024:.2f} GB)")
print(f"Total GPU Memory: {df['Total_MB'].iloc[0]:.0f} MB ({df['Total_MB'].iloc[0]/1024:.2f} GB)")
print(f"Peak Usage %: {(df['Used_MB'].max() / df['Total_MB'].iloc[0] * 100):.1f}%")
print("="*60)

# Find when peak occurred
peak_idx = df['Used_MB'].idxmax()
print(f"\nPeak occurred at: {df.loc[peak_idx, 'Timestamp']}")
print(f"GPU Utilization at peak: {df.loc[peak_idx, 'Utilization%']:.0f}%")
