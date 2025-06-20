from argparse import ArgumentParser
import numpy as np
from eigsep_corr.redis import grab_data

parser = ArgumentParser()
parser.add_argument("-fname", help="filename")
args = parser.parse_args()

# REDIS_HOST = "10.10.10.10"
REDIS_HOST = "192.168.10.83"
REDIS_PORT = 6379
PAIRS = ["0", "1", "2", "3", "4", "5", "02", "24", "04", "13", "35", "15"]

data = grab_data(pairs=PAIRS, host=REDIS_HOST, port=REDIS_PORT)
fname = args.fname
np.savez(fname, **data)
