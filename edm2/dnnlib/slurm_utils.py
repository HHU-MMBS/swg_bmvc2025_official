import urllib.request
import urllib.error
import socket
import torch.distributed as dist

def check_internet(url="http://www.google.com", timeout=2):
    """
    Checks for internet connectivity by attempting to reach a URL,
    handling various potential errors like timeouts, connection issues, and DNS failures.

    Args:
        url (str): The URL to try to connect to. Defaults to Google's homepage.
        timeout (int): Timeout in seconds for the request.

    Returns:
        bool: True if internet is likely available, False otherwise.
    """

    # Skip the check if running in distributed mode and not the master process
    if dist.is_initialized() and dist.get_rank() != 0:
        return True
    
    try:
        # Attempt to open the URL with a timeout
        urllib.request.urlopen(url, timeout=timeout)
        print(f"Internet check successful: Connection to {url} successful.")
        return True

    except urllib.error.URLError as e:
        # Catch URL-related errors (e.g., no network, server not found, invalid URL)
        if isinstance(e.reason, socket.gaierror):
            # Handle DNS resolution errors (e.g., no internet, DNS server issues)
            print(f"Internet check failed: DNS resolution error - {e.reason}")
        elif isinstance(e.reason, socket.timeout):
            # Handle timeout errors (e.g., slow network, server not responding)
            print(f"Internet check failed: Timeout error - {e.reason}")
        elif isinstance(e.reason, ConnectionRefusedError):
            # Handle connection refused errors (server actively refused connection)
            print(f"Internet check failed: Connection refused - {e.reason}")
        else:
            # Handle other URLError reasons
            print(f"Internet check failed: URLError - {e.reason}")
        return False

    except socket.timeout:
        # Catch socket timeout errors directly (if urllib.request.urlopen itself times out)
        print("Internet check failed: Socket timeout during connection.")
        return False

    except socket.error as e:
        # Catch general socket errors (e.g., connection reset, network unreachable)
        print(f"Internet check failed: Socket error - {e}")
        return False

    except Exception as e:
        # Catch any other unexpected exceptions during the process
        print(f"Internet check failed: Unexpected error - {e}")
        return False