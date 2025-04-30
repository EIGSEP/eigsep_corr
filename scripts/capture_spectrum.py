from argparse import ArgumentParser
import numpy as np
from eigsep_observing import EigsepRedis

parser = ArgumentParser()
parser.add_argument("-fname", help="filename")
args = parser.parse_args()

REDIS_HOST = "10.10.10.10"
REDIS_PORT = 6379
PAIRS = ["0", "1", "2", "3", "4", "5", "02", "24", "04", "13", "35", "15"]

r = EigsepRedis(host=REDIS_HOST, port=REDIS_PORT)
data = {}
for pair in PAIRS:
    data[pair] = r.get_raw(f"data:{pair}")
fname = args.fname
np.savez(fname, **data)
