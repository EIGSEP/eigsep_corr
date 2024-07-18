import redis

autos = ["0", "1", "2", "3", "4", "5"]
crosses = ["02", "04", "24", "13", "15", "35"]
all_pairs = autos + crosses

HOST = "10.10.10.10"
PORT = 6379

def grab_data(pairs=all_pairs, host=HOST, port=PORT):
    r = redis.Redis(host=host, port=port)
    data = {}
    for p in pairs:
        data[p] = r.get(f"data:{p}")
    return data

def grab_imu(r):
    """
    r : redis.Redis instance
    """
    pos = {"theta": r.get("theta"), "phi": r.get("phi")}
    return pos
