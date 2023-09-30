import redis
from eigsep_corr.plot import plot

REDIS_HOST = "192.168.0.116"
REDIS_PORT = 6379

r = redis.Redis(REDIS_HOST, port=REDIS_PORT)

plot(r, pairs=["0", "2", "02"], ylim_phase=(-5, 5))
