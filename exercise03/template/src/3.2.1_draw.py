import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("3.2.1_coalesced_results.csv")
df = df.dropna()
df["Size(Bytes)"] = df["Size(Bytes)"].astype(int)
df["ThreadsPerBlock"] = df["ThreadsPerBlock"].astype(int)
df["Bandwidth(GB/s)"] = df["Bandwidth(GB/s)"].astype(float)
df["Size(KB)"] = df["Size(Bytes)"] / 1024

sizes = sorted(df["Size(KB)"].unique())
x = range(len(sizes))

plt.figure(figsize=(10, 6))
for tpb in sorted(df["ThreadsPerBlock"].unique()):
    sub = df[df["ThreadsPerBlock"] == tpb]
    plt.plot(x, sub["Bandwidth(GB/s)"], marker='o', label=f"{tpb} threads/block")

plt.xticks(x, [f"{int(s)}" for s in sizes], rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel("Size (KB)")
plt.ylabel("Bandwidth (GB/s)")
plt.title("Global Memory Coalesced Copy Performance")
plt.legend(loc="best", fontsize="small", ncol=2)
plt.tight_layout()
plt.savefig("3.2.1.png", dpi=300)
plt.show()
