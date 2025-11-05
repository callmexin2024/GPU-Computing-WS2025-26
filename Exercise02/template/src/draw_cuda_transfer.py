import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cuda_transfer.csv")

plt.figure(figsize=(10,6))

plt.plot(df['Size_MB'], df['H2D_Pageable'], 'o-', label='H2D Pageable', color='blue')
plt.plot(df['Size_MB'], df['H2D_Pinned'],   'o-', label='H2D Pinned',   color='cyan')
plt.plot(df['Size_MB'], df['D2H_Pageable'], 's-', label='D2H Pageable', color='red')
plt.plot(df['Size_MB'], df['D2H_Pinned'],   's-', label='D2H Pinned',   color='orange')

plt.xscale('log')

plt.xlabel("Data Size (MB)")
plt.ylabel("Bandwidth (GB/s)")
plt.title("CUDA Host-Device Transfer Bandwidth")

plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()

plt.savefig("cuda_transfer_bandwidth.png", dpi=300)

plt.show()
