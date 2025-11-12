import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_FILE = "./3.3_stride_results.csv"

df = pd.read_csv(CSV_FILE)

df['Bandwidth(GB/s)'] = df['Bandwidth(GB/s)'].str.replace("GB/s", "").astype(float)

def format_size(x):
    if x < 1024*1024:
        return f"{x//1024} KB"
    else:
        return f"{x//(1024*1024)} MB"

df['SizeLabel'] = df['Size(Bytes)'].apply(format_size)

sizes = sorted(df['Size(Bytes)'].unique())

plt.figure(figsize=(10, 6))

for size in sizes:
    subset = df[df['Size(Bytes)'] == size]
    plt.plot(subset['Stride'], subset['Bandwidth(GB/s)'], marker='o', label=format_size(size))

plt.xlabel("Stride")
plt.ylabel("Bandwidth (GB/s)")
plt.title("Stride vs Global Memory Bandwidth")
plt.grid(True)

plt.xticks(df['Stride'].unique())

y_min, y_max = df['Bandwidth(GB/s)'].min(), df['Bandwidth(GB/s)'].max()
plt.yticks(np.linspace(y_min, y_max, 6))

plt.legend(title="Data Size")
plt.tight_layout()
plt.savefig("3.3_stride.png", dpi=300)