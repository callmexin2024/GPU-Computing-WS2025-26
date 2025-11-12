import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

file_path = "3.4_offset_results.csv"
df = pd.read_csv(file_path)

plt.figure(figsize=(10,6))

for size in sorted(df['Size(Bytes)'].unique()):
    subset = df[df['Size(Bytes)'] == size]
    plt.plot(subset['Offset'], subset['Bandwidth(GB/s)'], marker='o', label=f'Size={size}B')

plt.xlabel("Offset")
plt.ylabel("Bandwidth (GB/s)")
plt.title("Global Memory Offset Benchmark")
plt.legend()
plt.grid(True)

plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
plt.xticks(subset['Offset'])
plt.tight_layout()
plt.savefig("3.4.png", dpi=300)
plt.show()
