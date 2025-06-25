import paramiko
import time
import argparse

# SSH credentials
username = "eigsep"
password = "universe"  # Replace with your actual password
connection_timeout = 10  # Timeout for SSH connections in seconds
sync_interval = 60  # Interval to sync time in seconds


def get_current_time(ssh):
    command = (
        'python3 -c "import time; from datetime import datetime; '
        "now = time.time(); "
        "epoch_time = now; "
        "human_time = datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S'); "
        "print('Epoch time:', epoch_time); "
        "print('Human-readable time:', human_time)\""
    )
    stdin, stdout, stderr = ssh.exec_command(command)
    stdout_str = stdout.read().decode().strip()
    stderr_str = stderr.read().decode().strip()

    if stdout_str:
        print(f"Current time on Pi:\n{stdout_str}")
    if stderr_str:
        print(f"Error output from Pi:\n{stderr_str}")


def sync_time(host):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        print(f"Connecting to {host}...")
        ssh.connect(
            host,
            username=username,
            password=password,
            timeout=connection_timeout,
        )
        print(f"Connected to {host}")

        print("Before sync:")
        get_current_time(ssh)

        ssh.exec_command(f'sudo date -s "{current_time}"')
        print(f"Time synced with {host}")

        print("After sync:")
        get_current_time(ssh)

        ssh.close()
    except Exception as e:
        print(f"Failed to sync time with {host}: {e}")
    print("\n")


def main(pis_to_connect):
    try:
        while True:
            for ip in pis_to_connect:
                sync_time(ip)
            time.sleep(sync_interval)
    except KeyboardInterrupt:
        print("\nSyncing script interrupted. Exiting gracefully...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync time with specified Raspberry Pis"
    )
    parser.add_argument(
        "pis",
        metavar="PI",
        type=str,
        nargs="+",
        help=(
            "IP addresses of Raspberry Pis to connect to (e.g., "
            "10.10.10.10 10.10.10.11 10.10.10.12)"
        ),
    )
    args = parser.parse_args()
    main(args.pis)
