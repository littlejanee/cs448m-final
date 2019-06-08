import axi
from axi import Device


class PlotterAxi:
    def __init__(self, w, h):
        self.x = 0
        self.y = 0
        self.w = w
        self.h = h
        self.device = Device()
        self.device.enable_motors()
        initpos = self.device.read_position()
        print(initpos)
        self.device.zero_position()
        print(self.device.read_position())

    def inbounds(self, x, y):
        return not (x < 0 or x >= self.w or y < 0 or y >= self.h)

    def sprint(self, x, y):
        if not self.inbounds(x, y):
            print('Warning: sprint to bad position ({}, {})'.format(x, y))
            return False

        self.device.goto_rel(x - self.x, y - self.y)
        self.x = x
        self.y = y
        return True

    def sprint_rel(self, x, y):
        if self.device.goto_rel(x, y):
            self.x += x
            self.y += y

    def up(self):
        self.device.pen_up()

    def down(self):
        self.device.pen_down()

    def path(self, path):
        for x, y in path:
            if not self.inbounds(x, y):
                print('Warning: path to bad position ({}, {})'.format(x, y))
                return False

        self.device.run_path([(self.x, self.y)] + path)
        self.x = path[-1][0]
        self.y = path[-1][1]
        return True

    def move(self, x, y, feed=1000):
        if not self.inbounds(x, y):
            print('Warning: move to bad position ({}, {})'.format(x, y))
            return False

        self.device.move_rel(x - self.x, y - self.y)
        self.x = x
        self.y = y
        return True

    def move_rel(self, x, y, **kwargs):
        self.x += x
        self.y += y
        self.device.move_rel(x, y)
        return self.move(self.x, self.y, **kwargs)
