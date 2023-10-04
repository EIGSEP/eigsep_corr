import redis
from eigsep_corr.plot import plot_live

REDIS_HOST = "192.168.0.116"
REDIS_PORT = 6379

r = redis.Redis(REDIS_HOST, port=REDIS_PORT)

plot_live(r, plot_delay=True)
