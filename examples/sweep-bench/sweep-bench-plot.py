import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file', nargs='+')
args = parser.parse_args()

df = None

#for jsonl_file in args.file:
#    # Read JSONL file into DataFrame
#    df_part = pd.read_json(jsonl_file, lines=True)
#    df_part['label'] = jsonl_file
#    if df is None:
#        df = df_part
#    else:
#        df = pd.concat([df, df_part])
#



for md_file in args.file:
    # Read markdown table file into DataFrame
    df_part = pd.read_csv(md_file, sep=r'\s*\|\s*', engine='python', 
                         header=0, skiprows=[1])
    
    # Clean up columns (remove empty columns from markdown formatting)
    df_part = df_part.iloc[:, 1:-1]
    df_part.columns = [col.strip() for col in df_part.columns]
    
    # Rename columns to match expected names
    df_part = df_part.rename(columns={
        'N_KV': 'n_kv',
        'S_PP t/s': 'speed_pp',
        'S_TG t/s': 'speed_tg'
    })
    
    # Convert to numeric types
    df_part['n_kv'] = pd.to_numeric(df_part['n_kv'])
    df_part['speed_pp'] = pd.to_numeric(df_part['speed_pp'])
    df_part['speed_tg'] = pd.to_numeric(df_part['speed_tg'])
    
    # Add label and append to main DataFrame
    df_part['label'] = md_file
    df = pd.concat([df, df_part]) if df is not None else df_part

# Group by label and n_kv, calculate mean and std for both speed metrics
df_grouped = df.groupby(['label', 'n_kv']).agg({
    'speed_pp': ['mean', 'std'],
    'speed_tg': ['mean', 'std']
}).reset_index()

# Flatten multi-index columns
df_grouped.columns = ['label', 'n_kv', 'speed_pp_mean', 'speed_pp_std',
                      'speed_tg_mean', 'speed_tg_std']

# Replace NaN with 0 (std for a single sample is NaN)
df_grouped['speed_pp_std'] = df_grouped['speed_pp_std'].fillna(0)
df_grouped['speed_tg_std'] = df_grouped['speed_tg_std'].fillna(0)

# Prepare ticks values for X axis (prune for readability)
x_ticks = df['n_kv'].unique()
while len(x_ticks) > 16:
    x_ticks = x_ticks[::2]

# Get unique labels and color map
labels = df_grouped['label'].unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(labels)))

# Create prompt processing plot
plt.figure(figsize=(10, 6))
ax1 = plt.gca()
plt.grid()
ax1.set_xticks(x_ticks)

# Plot each label's data
for label, color in zip(labels, colors):
    label_data = df_grouped[df_grouped['label'] == label].sort_values('n_kv')
    pp = ax1.errorbar(label_data['n_kv'], label_data['speed_pp_mean'],
                     yerr=label_data['speed_pp_std'], color=color,
                     marker='o', linestyle='-', label=label)

# Add labels and title
ax1.set_xlabel('Context Length (tokens)')
ax1.set_ylabel('Prompt Processing Rate (t/s)')
plt.title('Prompt Processing Performance Comparison')
ax1.legend(loc='upper right')

# Adjust layout and save
plt.tight_layout()
plt.savefig('performance_comparison_pp.png', bbox_inches='tight')
plt.close()

# Create token generation plot
plt.figure(figsize=(10, 6))
ax1 = plt.gca()
plt.grid()
ax1.set_xticks(x_ticks)

# Plot each model's data
for label, color in zip(labels, colors):
    label_data = df_grouped[df_grouped['label'] == label].sort_values('n_kv')
    tg = ax1.errorbar(label_data['n_kv'], label_data['speed_tg_mean'],
                     yerr=label_data['speed_tg_std'], color=color,
                     marker='s', linestyle='-', label=label)

# Add labels and title
ax1.set_xlabel('Context Length (n_kv)')
ax1.set_ylabel('Token Generation Rate (t/s)')
plt.title('Token Generation Performance Comparison')
ax1.legend(loc='upper right')

# Adjust layout and save
plt.tight_layout()
plt.savefig('performance_comparison_tg.png', bbox_inches='tight')
plt.close()
