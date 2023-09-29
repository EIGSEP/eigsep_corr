import redis
from eigsep_corr.plot import plot

REDIS_HOST = "localhost"
REDIS_PORT = 6379

r = redis.Redis(REDIS_HOST, port=REDIS_PORT)

plot(r, ylim_mag=(0, 1e5))
