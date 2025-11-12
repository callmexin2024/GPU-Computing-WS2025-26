import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("3.2.3_all_results.csv")

def format_size(x):
    if x < 1024*1024:
        return f"{x/1024:.1f} KB"
    else:
        return f"{x/(1024*1024):.1f} MB"

df['Size_Label'] = df['Size(Bytes)'].apply(format_size)

df_plot = df[df['ThreadsPerBlock'] == 1024]

heatmap_data = df_plot.pivot(index='Size_Label', columns='Blocks', values='Bandwidth(GB/s)')

plt.figure(figsize=(12,8))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='viridis_r')
plt.title("Memory Bandwidth Heatmap (ThreadsPerBlock = 1024)")
plt.xlabel("Blocks")
plt.ylabel("Memory Size")
plt.tight_layout()
plt.show()
