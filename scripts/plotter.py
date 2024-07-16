import redis
from eigsep_corr.plot import plot_live

REDIS_HOST = "10.10.10.10"
REDIS_PORT = 6379
AUTOS = ["0", "1", "2", "3", "4", "5"]
pairs = ["2", "3"]

r = redis.Redis(REDIS_HOST, port=REDIS_PORT)
plot_live(r, pairs=AUTOS, plot_delay=False, log_scale=True)
