import axi
from axi import Device


class PlotterAxi:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.device = Device()
        self.device.enable_motors()
        initpos = self.device.read_position()
        print(initpos)
        self.device.zero_position()
        print(self.device.read_position())

    def sprint(self, x, y):
        self.device.goto_rel(x - self.x, y - self.y)
        self.x = x
        self.y = y

    def sprint_rel(self, x, y):
        self.x += x
        self.y += y
        self.device.goto_rel(x, y)

    def up(self):
        self.device.pen_up()

    def down(self):
        self.device.pen_down()

    def move(self, x, y, feed=1000):
        self.device.move_rel(x - self.x, y - self.y)
        self.x = x
        self.y = y

    def move_rel(self, x, y, **kwargs):
        self.x += x
        self.y += y
        self.device.move_rel(x, y)
        return self.move(self.x, self.y, **kwargs)
