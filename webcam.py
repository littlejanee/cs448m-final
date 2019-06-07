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
from threading import Thread

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

def CatmullRomSpline(P0, P1, P2, P3, nPoints=100):
  """
  P0, P1, P2, and P3 should be (x,y) point pairs that define the Catmull-Rom spline.
  nPoints is the number of points to include in this curve segment.
  """
  # Convert the points to numpy so that we can do array multiplication
  P0, P1, P2, P3 = map(numpy.array, [P0, P1, P2, P3])

  # Calculate t0 to t4
  alpha = 0.5
  def tj(ti, Pi, Pj):
    xi, yi = Pi
    xj, yj = Pj
    return ( ( (xj-xi)**2 + (yj-yi)**2 )**0.5 )**alpha + ti

  t0 = 0
  t1 = tj(t0, P0, P1)
  t2 = tj(t1, P1, P2)
  t3 = tj(t2, P2, P3)

  # Only calculate points between P1 and P2
  t = numpy.linspace(t1,t2,nPoints)

  # Reshape so that we can multiply by the points P0 to P3
  # and get a point for each value of t.
  t = t.reshape(len(t),1)
  A1 = (t1-t)/(t1-t0)*P0 + (t-t0)/(t1-t0)*P1
  A2 = (t2-t)/(t2-t1)*P1 + (t-t1)/(t2-t1)*P2
  A3 = (t3-t)/(t3-t2)*P2 + (t-t2)/(t3-t2)*P3
  B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2
  B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3

  C  = (t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2
  return C

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
    p.down()

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

    # initialize plotter
    if not DRY_RUN:
        p = PlotterAxi()

        def disable_device():
            p.device.wait()
            p.up()
            p.sprint(0, 0)

        atexit.register(disable_device)

        def send_commands():
            while True:
                path = [path_buffer.get() for _ in range(5)]
                p.path(path)

        Thread(target=send_commands, daemon=True).start()

        def recv_commands():
            while True:
                inp = input().strip()
                p.device.wait()
                if inp == 'i':
                    p.up()
                elif inp == 'o':
                    p.down()
                elif inp == 'h':
                    p.move(0, 0)

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
            print('client x, y: ', point[0], point[1])
            (x, y) = drawing.computedrawcoordinates(point[0], point[1])
            print('target x, y: ', x, y)

            points = history.pts[-3:] + [(x, y)]
            if len(points) == 4:
                interped = CatmullRomSpline(*points, nPoints=6)
                last_point = history.last()

                interped = [
                    (min(max(x, border), target_width - border),
                     min(max(y, border), target_height - border))
                    for (x, y) in interped
                ]

                if True or np.any(np.isnan(np.array(interped))):
                    if not DRY_RUN:
                        #p.move(x, y)
                        path_buffer.put((x, y))

                    line(history.last(), (x, y), color=(255, 0, 0))
                else:
                    if not DRY_RUN:
                        p.path(interped)

                    for (x_i, y_i) in interped:
                        new_point = (x_i, y_i)
                        line(last_point, new_point)
                        last_point = new_point
            else:
                if not DRY_RUN:
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
