import paramiko
import time
import argparse
from datetime import datetime

# SSH credentials
username = 'eigsep'
password = 'universe'

def sync_time(host):
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        print(f'Connecting to {host}...')
        ssh.connect(host, username=username, password=password)
        print(f'Connected to {host}')
        ssh.exec_command(f'sudo date -s "{current_time}"')
        print(f'Time synced with {host}')
        # Run a command to check the time on the Pi
        stdin, stdout, stderr = ssh.exec_command(
            'python3 -c "import time; from datetime import datetime; now = time.time(); '
            'print(f\'Epoch time: {now}\'); '
            'print(f\'Human-readable time: {datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")}\')"'
        )
        synced_time_output = stdout.read().decode().strip()
        print(f"Current time on {host}:\n{synced_time_output}")
        ssh.close()
    except Exception as e:
        print(f'Failed to sync time with {host}: {e}')

def main(pis_to_connect):
    for ip in pis_to_connect:
        sync_time(ip)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sync time with specified Raspberry Pis')
    parser.add_argument(
        'pis',
        metavar='PI',
        type=str,
        nargs='+',
        help='IP addresses of Raspberry Pis to connect to (e.g., 10.10.10.10 10.10.10.11 10.10.10.12)'
    )
    args = parser.parse_args()
    main(args.pis)
