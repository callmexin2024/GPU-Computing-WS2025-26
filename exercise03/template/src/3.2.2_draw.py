import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("3.2.2_blocks_results.csv")

data['Size_MB'] = data['Size(Bytes)'] / (1024*1024)

thread_count = 1024
data = data[data['ThreadsPerBlock'] == thread_count]

plt.figure(figsize=(10, 6))

memory_sizes = sorted(data['Size_MB'].unique())
for size in memory_sizes:
    subset = data[data['Size_MB'] == size]
    if size < 1:
        size_label = f'{size*1024:.0f} KB'
    else:
        size_label = f'{size:.0f} MB'
    plt.plot(subset['Blocks'], subset['Bandwidth(GB/s)'], marker='o', label=size_label)

plt.xlabel('Number of Thread Blocks')
plt.ylabel('Bandwidth (GB/s)')
plt.title(f'Global Memory Bandwidth vs Thread Blocks (Threads per Block = {thread_count})')
plt.xticks(range(1, 33))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Memory Size')
plt.tight_layout()
plt.savefig("3.2.2.png", dpi=300)
plt.show()
