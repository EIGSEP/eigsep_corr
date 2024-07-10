import redis
from eigsep_corr.plot import plot_live

REDIS_HOST = "10.10.10.10"
REDIS_PORT = 6379

r = redis.Redis(REDIS_HOST, port=REDIS_PORT)
pairs = ["2", "3"]
plot_live(r, pairs=pairs, plot_delay=False)
