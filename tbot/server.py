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
    x, y, ty = data['x'], data['y'], data['type']
    x = x * target_width / client_width
    x = target_width - x - target_width / 2
    y = y * target_height / client_height - target_height / 2
    if ty == 'start':
        p.sprint(x, y)
        p.down()
    elif ty == 'move':
        p.move(x, y, feed=10000)
    elif ty == 'end':
        p.up()
        

server.set_fn_message_received(recv)
server.run_forever()
