import cv2
import numpy as np
import sys
from plotteraxi import PlotterAxi

DEBUG = True

client_width = 1920
client_height = 1080

target_width = 11.0
target_height = 8.5

border = .2

fgbg = cv2.createBackgroundSubtractorMOG2()


# returns x, y (1920, 1080) or None if can't find
def find_marker(frame):
    fgmask = fgbg.apply(frame)
    masked = cv2.bitwise_and(frame, frame, mask=fgmask)

    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, sat_bright = cv2.threshold(sat, 200, 255, cv2.THRESH_TOZERO)
    sat_bright = cv2.erode(sat_bright, None, iterations=2)
    sat_bright = cv2.dilate(sat_bright, None, iterations=2)

    cnts, _ = cv2.findContours(sat_bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_srt = sorted([(c, cv2.contourArea(c)) for c in cnts], key=lambda t: t[1])
    if len(cnts) > 0:
        (c, area) = cnts_srt[-1]

        if area > 1000:
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy

    return None


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


# def draw(x, y):
#     x_border = x0 * (target_width - border) / client_width + border / 2
#     y_border = y0 * (target_height - border) / client_height + border / 2
#     p.move(x_border, y_border)

def main(cam_idx):
    cam = cv2.VideoCapture(cam_idx)
    result, frame = cam.read()
    assert result

    canvas = np.zeros(frame.shape)
    history = History()

    # initialize plotter
    p = PlotterAxi()
    p.down()

    while True:
        result, frame = cam.read()
        assert result

        point = find_marker(frame)
        if point is not None:

            if DEBUG:
                if history.last() is not None:
                    cv2.line(canvas, history.last(), point, (255, 255, 255), 3)

            # draw things
            x_border = point[0] * (target_width - border) / client_width + border / 2
            y_border = point[1] * (target_height - border) / client_height + border / 2
            p.move(x_border, y_border)
            # draw(point.x, point.y)

            history.shift(point)

        if DEBUG:
            cv2.imshow('frame', canvas)
            cv2.waitKey(30)


if __name__ == "__main__":
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(cam_idx)
