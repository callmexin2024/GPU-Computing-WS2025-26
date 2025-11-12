import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('cuda_transfer.csv')

x_data = df['Size_MB']

plt.figure(figsize=(12, 8))
plt.title('CUDA Host-Device Transfer Bandwidth', fontsize=16)
plt.xlabel('Data Size (MB)', fontsize=14)
plt.ylabel('Bandwidth (GB/s)', fontsize=14)

plt.plot(x_data, df['H2D_Pageable'], marker='o', linestyle='-', label='H2D Pageable', color='blue')
plt.plot(x_data, df['H2D_Pinned'], marker='s', linestyle='-', label='H2D Pinned', color='cyan')

plt.plot(x_data, df['D2H_Pageable'], marker='^', linestyle='--', label='D2H Pageable', color='red')
plt.plot(x_data, df['D2H_Pinned'], marker='D', linestyle='--', label='D2H Pinned', color='orange')

plt.plot(x_data, df['D2D'], marker='*', linestyle='-', linewidth=3, label='D2D (Device-to-Device)', color='red')

plt.xscale('log')
ticks = [1/1024, 1/256, 1/64, 1/16, 1/4, 1, 2, 2.25, 2.5, 2.75, 3, 4, 16, 64, 256, 1024]
tick_labels = ['1KB', '4KB', '16KB', '64KB', '256KB', '1MB', '2MB', '', '2.5MB', '', '3MB', '4MB', '16MB', '64MB', '256MB', '1GB']
plt.xticks([t for t in ticks if t in x_data.values or t == 1/1024], 
           [l for t, l in zip(ticks, tick_labels) if t in x_data.values or t == 1/1024], 
           rotation=45, ha='right')

plt.grid(True, which="both", linestyle='--', alpha=0.7)

plt.legend(loc='lower right', frameon=True, fontsize=10)

plt.tight_layout()

plt.savefig('cuda_transfer_bandwidth.png')
print("Plot successfully saved as 'cuda_transfer_bandwidth.png'.")

plt.show()