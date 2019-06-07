from websocket_server import WebsocketServer
import json
# from plotter import Plotter
from plotteraxi import PlotterAxi
from drawing import Drawing

client_width = 1100
client_height = 850

target_width = 11.0 #150
target_height = 8.5 #150

border = .2

DRY_RUN = True

server = WebsocketServer(9001)

if not DRY_RUN:
    # p = Plotter()
    p = PlotterAxi()

drawing = Drawing()
drawing.setstyle(0)
drawing.setclient(client_width, client_height)
drawing.settarget(target_width, target_height, border)


def recv(client, server, message):
    data = json.loads(message)
    x0, y0, ty = data['x'], data['y'], data['type']

    # x = x0 * target_width / client_width
    # y = y0 * target_height / client_height

    # print(x, y)

    # draw things
    print('client x, y: ', x0, y0)
    (x_draw, y_draw) = drawing.computedrawcoordinates(x0, y0)
    print('target x, y: ', x_draw, y_draw)

    x = x_draw
    y = y_draw

    # x_border = x0 * (target_width - border) / client_width + border / 2
    # y_border = y0 * (target_height - border) / client_height + border / 2
    # print('border: ', x_border, y_border)

    if not DRY_RUN:
        if ty == 'start':
            p.sprint(x, y)
            p.down()
        elif ty == 'move':
            p.move(x, y, feed=10000)
        elif ty == 'end':
            p.up()
        

server.set_fn_message_received(recv)
server.run_forever()
