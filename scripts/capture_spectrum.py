from argparse import ArgumentParser
import datetime
import os
import numpy as np
import redis

from eigsep_corr import io

parser = ArgumentParser()
parser.add_argument("--fname", default=None, help="filename")
args = parser.parse_args()

REDIS_HOST = "10.10.10.10"
REDIS_PORT = 6379
SAVE_DIR = "/media/eigsep/T7/data/gain_calibration"
PAIRS = ["0", "1", "2", "3", "4", "5", "02", "24", "04", "13", "35", "15"]

r = redis.Redis(REDIS_HOST, port=REDIS_PORT)
dt = np.dtype(np.int32).newbyteorder(">")
data = {}
for p in PAIRS:
    data[p] = r.get(f"data:{p}")
cnt = r.get("ACC_CNT")

date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fname = args.fname
if fname is None:
    fname = os.path.join(SAVE_DIR, f"{date}.eig")
np.savez(fname, **data)
#io.write_file(fname, io.DEFAULT_HEADER, data)
