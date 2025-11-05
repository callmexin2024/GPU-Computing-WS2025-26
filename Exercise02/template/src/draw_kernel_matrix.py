import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./kernel_timings.csv")

df = df.sort_values(by=['num_blocks', 'threads_per_block'])

num_blocks = df['num_blocks'].nunique()
num_threads = df['threads_per_block'].nunique()

threads = sorted(df['threads_per_block'].unique())
blocks = sorted(df['num_blocks'].unique())

async_times = df['async_time_us'].values.reshape(num_blocks, num_threads)
sync_times  = df['sync_time_us'].values.reshape(num_blocks, num_threads)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

im0 = axs[0].imshow(async_times, origin='lower',
                    extent=[threads[0], threads[-1], blocks[0], blocks[-1]],
                    aspect='auto', cmap='viridis')
axs[0].set_title("Asynchronous launch time (us)")
axs[0].set_xlabel("Threads per block")
axs[0].set_ylabel("Number of blocks")
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(sync_times, origin='lower',
                    extent=[threads[0], threads[-1], blocks[0], blocks[-1]],
                    aspect='auto', cmap='viridis')
axs[1].set_title("Synchronous launch time (us)")
axs[1].set_xlabel("Threads per block")
axs[1].set_ylabel("Number of blocks")
fig.colorbar(im1, ax=axs[1])

plt.tight_layout()

output_path = os.path.join(os.getcwd(), "kernel_timings.png")
plt.savefig(output_path, dpi=300)
print(f"✅ Saved to: {output_path}")

plt.show()
