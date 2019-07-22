import socket


def find_open_ports(num_ports):
    sockets = []
    for i in range(num_ports):
        s = socket.socket()
        s.bind(('', 0))
        sockets.append(s)
    ports = [s.getsockname()[1] for s in sockets]
    # Close sockets to allow tensorflow distribute server to connect
    for s in sockets:
        s.close()
    return ports
