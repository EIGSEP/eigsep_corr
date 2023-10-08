import redis
from eigsep_corr.plot import plot_live

REDIS_HOST = "192.168.0.116"
REDIS_PORT = 6379

r = redis.Redis(REDIS_HOST, port=REDIS_PORT)
pairs = ["0", "1", "2", "3", "02", "13"]
plot_live(r, pairs=pairs, plot_delay=True)
