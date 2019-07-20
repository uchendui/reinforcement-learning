import socket


def find_open_ports(num_ports):
    # https://www.pythonforbeginners.com/code-snippets-source-code/port-scanner-in-python/
    n = 0
    ports = []
    remote_server_ip = socket.gethostbyname('localhost')
    remote_server_ip = 'localhost'

    for port in range(8888, 10000):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((remote_server_ip, port))
            if result == 0:
                print("Port {}:  Open".format(port))
                ports.append(port)
            sock.close()

        except KeyboardInterrupt:
            print("You pressed Ctrl+C")
            break
        except socket.gaierror:
            print('Hostname could not be resolved. Exiting')
            break
        except socket.error:
            print("Couldn't connect to server")

    return ports
