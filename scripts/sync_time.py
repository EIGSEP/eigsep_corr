import paramiko
import time
import argparse
import nmap

# SSH credentials
username = 'eigsep'
password = 'universe'  # Replace with your actual password
connection_timeout = 10  # Timeout for SSH connections in seconds

# List of Raspberry Pi hostnames (if needed for nmap scanning)
raspberry_pi_hostnames = {
    '10.10.10.10': 'pi1',
    '10.10.10.11': 'pi2',
    '10.10.10.12': 'pi3'
}

def scan_network_for_pi(pi_hostname):
    nm = nmap.PortScanner()
    nm.scan(hosts='10.10.10.0/24', arguments='-sn')
    for host in nm.all_hosts():
        if 'hostnames' in nm[host] and nm[host]['hostnames']:
            for hostname in nm[host]['hostnames']:
                if hostname['name'] == pi_hostname:
                    return host
    return None

def sync_time(host):
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        print(f'Connecting to {host}...')
        ssh.connect(host, username=username, password=password, timeout=connection_timeout)
        print(f'Connected to {host}')
        ssh.exec_command(f'sudo date -s "{current_time}"')
        print(f'Time synced with {host}')
        
        # Run a command to check the time on the Pi
        command = (
            'python3 -c "import time; from datetime import datetime; '
            'now = time.time(); '
            'epoch_time = now; '
            'human_time = datetime.fromtimestamp(now).strftime(\'%Y-%m-%d %H:%M:%S\'); '
            'print(\'Epoch time:\', epoch_time); '
            'print(\'Human-readable time:\', human_time)"'
        )
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout_str = stdout.read().decode().strip()
        stderr_str = stderr.read().decode().strip()
        
        if stdout_str:
            print(f"Current time on {host}:\n{stdout_str}")
        if stderr_str:
            print(f"Error output from {host}:\n{stderr_str}")
            
        ssh.close()
    except Exception as e:
        print(f'Failed to sync time with {host}: {e}')
        if host in raspberry_pi_hostnames:
            new_ip = scan_network_for_pi(raspberry_pi_hostnames[host])
            if new_ip:
                print(f'Found {host} under new IP: {new_ip}. Retrying...')
                sync_time(new_ip)
            else:
                print(f'Could not find {host} on the network.')

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