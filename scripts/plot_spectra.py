import numpy as np
import matplotlib.pyplot as plt

FEMS = ["12", "032", "320", "348"]
dt = np.dtype(np.int32).newbyteorder(">")

def byte2int(fem_id):
    n_path = f"data/fem{fem_id}_north.npz"
    e_path = f"data/fem{fem_id}_east.npz"
    d_north = np.frombuffer(np.load(n_path)["3"], dtype=dt)
    d_east = np.frombuffer(np.load(e_path)["3"], dtype=dt)
    return np.array([d_north, d_east])

data = {}
plt.figure()
for fem_id in FEMS:
    d = byte2int(fem_id)
    data[fem_id] = d
    plt.plot(d[0], label=f"{fem_id}: north")
    plt.plot(d[1], label=f"{fem_id}: east")
plt.legend()
plt.show()
