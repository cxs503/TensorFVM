import numpy as np
r = np.load("/root/tp-agent/work/m21251/outputs/m21251/diag_rows.npy")
print("max |u| over run:", r[:, 3].max(), "at t=", r[np.argmax(r[:, 3]), 1])
print("min dt:", r[:, 2].min(), "clips:", r[-1, 6], "flips:", r[-1, 7])
print("volume drift:", (r[-1, 8] - r[0, 8]) / r[0, 8])
for i in range(0, len(r), 20):
    print("step %4d t=%.4f T=%.2f |u|max=%8.4f dt=%.2e hmin=%.2e"
          % (r[i, 0], r[i, 1], r[i, 1] / 0.70925, r[i, 3], r[i, 2], r[i, 4]))