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

client_width = 1350#1920
client_x = int(1920 / 2 - client_width / 2)

client_height = 1000
client_y  = int(1080 / 2 - client_height / 2)

target_width = 11.0
target_height = 8.5

border = 1.5

fgbg = cv2.createBackgroundSubtractorMOG2()

# marker colors in RGB
# yellow = (232, 187, 101)
# green = (57, 117, 97)
# blue = (28, 128, 168)
# red = (171, 21, 36)

# marker colors in HSV
# yellow = (39,56,91)
# green = (160, 51, 46)
# blue = (197, 83, 66)
# red = (354, 88, 67)

sat_low = 60#30
sat_high = 255#100
value_low = 60#35
value_high = 255#100

boundaries = [
    # ([0, sat_low, value_low], [255, sat_high, value_high]), # test
    ([25, sat_low, value_low], [65, sat_high, value_high]), # yellow
    ([95, sat_low, value_low], [120, sat_high, value_high]), # green
    ([130, sat_low, value_low], [180, sat_high, value_high]), # blue
    ([200, sat_low, value_low], [300, sat_high, value_high]), # red
]

# returns x, y (1920, 1080) or None if can't find
def find_marker_for_id(frame, marker_id):
    #fgmask = fgbg.apply(frame)
    #masked = cv2.bitwise_and(frame, frame, mask=fgmask)
    masked = frame

    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)

    # sat = hsv[:, :, 1]
    # _, sat_bright = cv2.threshold(sat, 200, 255, cv2.THRESH_TOZERO)

    # sat_bright = cv2.erode(sat_bright, None, iterations=2)
    # sat_bright = cv2.dilate(sat_bright, None, iterations=2)

    lower = np.array(boundaries[marker_id][0])
    upper = np.array(boundaries[marker_id][1])

    lower[0] = int(lower[0] / 360 * 255.)
    upper[0] = int(upper[0] / 360 * 255.) # maybe ceil/floor and clip?

    print (lower)
    print ('hsv[100][100]')
    print (hsv[100][100])

    hue_color = cv2.inRange(hsv, lower, upper)
    hue_color = cv2.erode(hue_color, None, iterations=2)
    hue_color = cv2.dilate(hue_color, None, iterations=2)
    cv2.imshow('hue_color', hue_color)

    masked_color = cv2.bitwise_and(masked, masked, mask = hue_color)
    cv2.imshow('masked', np.hstack([masked, masked_color]))
    cv2.waitKey(0)

    # find center
    cnts, _ = cv2.findContours(hue_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    frame = cv2.imread("color-webcam.jpg", cv2.IMREAD_COLOR)
    find_marker_for_id(frame, 3)
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()

    cam = cv2.VideoCapture(cam_idx)
    result, frame = cam.read()
    assert result

    shape = list(frame.shape)
    [frame_height, frame_width] = shape[:2]
    shape[0] *= 2
    shape[1] *= 2
    canvas = np.zeros(shape, dtype=np.uint8)
    raw_history = History()
    draw_history = History()

    drawing = Drawing()
    drawing.setstyle(0)
    drawing.setclient(client_width, client_height)
    drawing.settarget(target_width, target_height, border)

    p = PlotterAxi()

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
                path = [path_buffer.get() for _ in range(9)]
                if path_buffer.qsize() > 4:
                    path = path[::3]
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

                try:
                    print(inp)
                    n = int(inp)
                    drawing.setstyle(n)
                    print('Set style to {}'.format(n))
                except ValueError:
                    pass


    Thread(target=recv_commands, daemon=True).start()

    while True:
        result, frame = cam.read()
        assert result
        canvas[:frame_height, frame_width:, :] = frame[::-1, ::-1, :]

        point, _, sat_img = find_marker(frame)
        canvas[frame_height:, frame_width:, :] = np.expand_dims(sat_img, axis=2)[::-1, ::-1, :]

        if point is not None:
            px, py = point
            if px >= client_x and px <= client_x + client_width and \
               py >= client_y and py <= client_y + client_height:

                # Put coordinates within the client box
                px -= client_x
                py -= client_y

                # Mirror x because we're behind the camera
                point = (client_width - px, client_height - py)

                (x, y) = drawing.computedrawcoordinates(*point)

                path_buffer.put((x, y))

                # Draw raw coordinates

                point = (int(point[0] * frame_width / client_width), int(point[1] * frame_height / client_height))
                cv2.line(canvas, raw_history.last(), point, (255, 255, 255), 3)
                cv2.circle(canvas, point, 10, (255, 255, 255), -1)

                # Draw draw coordinates
                draw_canvas = canvas[frame_height:, :frame_width, :]

                def paper_to_pixel(p):
                    return (int(p[0] / target_width * frame_width),
                            int(p[1] / target_height * frame_height))

                if draw_history.last() is not None:
                    cv2.line(
                        draw_canvas,
                        paper_to_pixel(draw_history.last()),
                        paper_to_pixel((x, y)),
                        (255, 255, 0),
                        3)
                cv2.circle(
                    draw_canvas,
                    paper_to_pixel((x, y)),
                    10,
                    (255, 255, 0),
                    -1)

                raw_history.shift(point)
                draw_history.shift((x, y))

        camera_canvas = canvas[:, frame_width:, :]
        cv2.line(camera_canvas, (client_x, 0), (client_x, frame_height), (255, 255, 0), 3)
        cv2.line(camera_canvas, (client_x + client_width, 0), (client_x + client_width, frame_height), (255, 255, 0), 3)
        cv2.line(camera_canvas, (0, client_y), (frame_width, client_y), (255, 255, 0), 3)
        cv2.line(camera_canvas, (0, client_y + client_height), (frame_width, client_y + client_height), (255, 255, 0), 3)

        canvas_resize = cv2.resize(canvas, (1920, 1080))
        cv2.imshow('frame', canvas_resize)
        cv2.waitKey(30)

if __name__ == "__main__":
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(cam_idx)
