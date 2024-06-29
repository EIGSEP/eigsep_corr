import paramiko
import time

# List of Raspberry Pi IPs
raspberry_pi_ips = ['10.10.10.10', '10.10.10.11', '10.10.10.12']

# SSH credentials
username = 'eigsep'
password = 'universe'

def sync_time(host):
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(host, username=username, password=password)
        ssh.exec_command(f'sudo date -s "{current_time}"')
        ssh.close()
        print(f'Time synced with {host}')
    except Exception as e:
        print(f'Failed to sync time with {host}: {e}')

def main():
    for ip in raspberry_pi_ips:
        sync_time(ip)

if __name__ == '__main__':
    main()
