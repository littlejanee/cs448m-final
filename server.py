from websocket_server import WebsocketServer
import json
from plotter import Plotter

server = WebsocketServer(9001)
p = Plotter()

client_width = 500
client_height = 500

target_width = 150
target_height = 150


def recv(client, server, message):
    data = json.loads(message)
    x, y = data['x'], data['y']
    x = x * target_width / client_width
    y = y * target_height / client_height
    p.move(x, y)
        

server.set_fn_message_received(recv)
server.run_forever()
