import socket


def find_open_ports(num_ports):
    sockets = []
    for i in range(num_ports):
        s = socket.socket()
        s.bind(('', 0))
        sockets.append(s)
    # s.bind(('', 0))  # Bind to a free port provided by the host.

    return [s.getsockname()[1] for s in sockets]  # Return the port number assigned.
