import serial
import os
import subprocess as sp
import time

import axi
from axi import Device


class PlotterAxi:
    def __init__(self, port=None, baud=115200):
        # if port is None:
        #     s = sp.check_output('ls /dev/tty.* | grep usbmodem', shell=True)
        #     port = s.decode('utf-8')[:-1]
        # self.port = serial.Serial(port, baud)
        # self.write('\r\n\r')
        # self.port.flushInput()
        self.x = 0
        self.y = 0
        self.device = Device()
        self.device.enable_motors()
        initpos = self.device.read_position()
        print(initpos)
        self.device.zero_position()
        print(self.device.read_position())

    # def write(self, s):
    #     self.port.write((s + '\n').encode())
    #     return self.port.readline()

    def sprint(self, x, y):
        self.device.goto_rel(x - self.x, y - self.y)
        self.x = x
        self.y = y
        # return self.write('G0 X{} Y{}'.format(x, y))

    def sprint_rel(self, x, y):
        self.x += x
        self.y += y
        self.device.goto_rel(x, y)
        # return self.sprint(self.x, self.y)

    def up(self):
        self.device.pen_up()
        # return self.write('G0 Z1')

    def down(self):
        self.device.pen_down()
        # return self.write('G0 Z-1')

    def move(self, x, y, feed=1000):
        self.device.move_rel(x - self.x, y - self.y)
        self.x = x
        self.y = y
        # return self.write('G1 X{} Y{} F{}'.format(x, y, feed))

    def move_rel(self, x, y, **kwargs):
        self.x += x
        self.y += y
        self.device.move_rel(x, y)
        return self.move(self.x, self.y, **kwargs)

    def runfile(self, path):
        contents = open(os.path.expanduser(path), 'r').readlines()
        commands = []
        for line in contents:
            if line[0] == 'G' or line[0] == 'M':
                commands.append(line)

        for cmd in commands:
            print(cmd)
            self.write(cmd)
