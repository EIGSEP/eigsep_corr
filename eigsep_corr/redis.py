import redis

autos = ["0", "1", "2", "3", "4", "5"]
crosses = ["02", "04", "24", "13", "15", "35"]
all_pairs = autos + crosses


def grab_data(pairs=all_pairs, host="10.10.10.10", port=6379):
    r = redis.Redis(host=host, port=port)
    data = {}
    for p in pairs:
        data[p] = r.get(f"data:{p}")
    return data
