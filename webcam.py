import cv2
import numpy as np
import sys
from plotteraxi import PlotterAxi
from drawing import Drawing
from scipy import interpolate as interp
import numpy
from pprint import pprint
import atexit
from queue import Queue
from threading import Thread, Lock
import subprocess as sp


DRY_RUN = False
FULL_DEBUG = True

client_width = 1920
client_height = 1080

target_width = 11.0 - 2
target_height = 8.5 - 2

border = .2

fgbg = cv2.createBackgroundSubtractorMOG2()


# returns x, y (1920, 1080) or None if can't find
def find_marker(frame):
    #fgmask = fgbg.apply(frame)
    #masked = cv2.bitwise_and(frame, frame, mask=fgmask)
    masked = frame

    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, sat_bright = cv2.threshold(sat, 200, 255, cv2.THRESH_TOZERO)
    sat_bright = cv2.erode(sat_bright, None, iterations=2)
    sat_bright = cv2.dilate(sat_bright, None, iterations=2)

    cnts, _ = cv2.findContours(sat_bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_srt = sorted([(c, cv2.contourArea(c)) for c in cnts], key=lambda t: t[1])
    point = None
    if len(cnts) > 0:
        (c, area) = cnts_srt[-1]

        if area > 1000:
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            point = cx, cy

    return point, masked, sat_bright


class History:
    def __init__(self, n=10):
        self.pts = []
        self.n = n

    def shift(self, pt):
        self.pts.append(pt)
        if len(self.pts) > self.n:
            del self.pts[0]

    def last(self):
        return self.pts[-1] if len(self.pts) > 0 else None


def main(cam_idx):
    cam = cv2.VideoCapture(cam_idx)
    result, frame = cam.read()
    assert result

    shape = list(frame.shape)
    [frame_height, frame_width] = shape[:2]
    shape[0] *= 2
    shape[1] *= 2
    canvas = np.zeros(shape, dtype=np.uint8)
    history = History()

    drawing = Drawing()
    drawing.setstyle(0)
    drawing.setclient(client_width, client_height)
    drawing.settarget(target_width, target_height, border)

    p = PlotterAxi()

    def point_to_canvas(p):
        (x, y) = p
        return (
            int((x - border / 2) * client_width / (target_width - border)),
            int((y - border / 2) * client_height / (target_height - border))
        )

    def line(p1, p2, color=(255, 0, 0)):
        if p1 is None or p2 is None:
            return

        p1 = point_to_canvas(p1)
        p2 = point_to_canvas(p2)

        cv2.line(canvas, p1, p2, color, 3)

    def circle(p):
        p = point_to_canvas(p)
        cv2.circle(canvas, p, 5, (255, 255, 255), 3)

    path_buffer = Queue()
    device_lock = Lock()

    # initialize plotter
    if not DRY_RUN:
        p = PlotterAxi()

        def disable_device():
            with device_lock:
                p.device.wait()
                p.up()
                p.sprint(0, 0)
                p.device.wait()
                sp.check_call(['axi', 'off'])

        atexit.register(disable_device)

        def send_commands():
            while True:
                path = [path_buffer.get() for _ in range(3)]
                with device_lock:
                    p.path(path)

        Thread(target=send_commands, daemon=True).start()

        def recv_commands():
            while True:
                inp = input().strip()
                with device_lock:
                    if inp == 'i':
                        p.up()
                    elif inp == 'o':
                        p.down()

        Thread(target=recv_commands, daemon=True).start()

    while True:
        result, frame = cam.read()
        assert result
        if FULL_DEBUG:
            canvas[:frame_height, frame_width:, :] = frame

        point, mask_img, sat_img = find_marker(frame)
        if FULL_DEBUG:
            canvas[frame_height:, :frame_width, :] = mask_img
            canvas[frame_height:, frame_width:, :] = np.expand_dims(sat_img, axis=2)

        if point is not None:
            (x, y) = drawing.computedrawcoordinates(point[0], point[1])

            path_buffer.put((x, y))
            line(history.last(), (x, y))

            line(history.last(), (x, y), color=(255, 255, 255))
            circle((x, y))
            history.shift((x, y))

        if FULL_DEBUG:
            cv2.imshow('frame', canvas)
            cv2.waitKey(30)

if __name__ == "__main__":
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(cam_idx)
