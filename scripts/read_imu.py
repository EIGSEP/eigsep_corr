import redis
import time
from eigsep_corr.redis import grab_imu

MOTOR_HOST = "10.10.10.12"
PORT = 6379
r = redis.Redis(host=MOTOR_HOST, port=PORT)

while True:
    try:
        pos = grab_imu(r)
        print(pos)
        time.sleep(1.)
    except KeyboardInterrupt:
        print("Stopping.")
        break
