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
import pickle


DRY_RUN = False

DEBUG = True

client_width = 1350#1920
client_x = int(1920 / 2 - client_width / 2)

client_height = 1000
client_y  = int(1080 / 2 - client_height / 2)

target_width = 11.0
target_height = 8.5

border = 1.5

intrinsics = pickle.load(open('data/intrinsics.pkl', 'rb'))
camera_height = 13
pen_height = 7

newcameramtx = None

sat_low = 60
sat_high = 255
value_low = 100
value_high = 200

boundaries = [
    ([25, sat_low, value_low], [65, sat_high, value_high]), # yellow
    ([95, sat_low, value_low], [120, sat_high, value_high]), # green
    ([130, sat_low, value_low], [155, sat_high, value_high]), # blue
    ([230, sat_low, value_low], [300, sat_high, value_high]), # red
]

COLOR_YELLOW = 0
COLOR_GREEN = 1
COLOR_BLUE = 2
COLOR_RED = 3

# returns x, y (1920, 1080) or None if can't find
def find_marker_for_id(frame, marker_id):
    frame_fixed = \
        cv2.undistort(frame, intrinsics['mtx'], intrinsics['dist'], newcameramtx)

    hsv = cv2.cvtColor(frame_fixed, cv2.COLOR_BGR2HSV)

    lower = np.array(boundaries[marker_id][0])
    upper = np.array(boundaries[marker_id][1])

    lower[0] = int(lower[0] / 360 * 255.)
    upper[0] = int(upper[0] / 360 * 255.) # maybe ceil/floor and clip?

    # print (lower)
    # print ('hsv[100][100]')
    # print (hsv[100][100])

    hue_color = cv2.inRange(hsv, lower, upper)
    hue_color = cv2.erode(hue_color, None, iterations=2)
    hue_color = cv2.dilate(hue_color, None, iterations=2)

    frame_fixed_color = cv2.bitwise_and(frame_fixed, frame_fixed, mask = hue_color)

    if DEBUG:
        cv2.imshow('hue_color', hue_color)
        cv2.imshow('frame_fixed', np.hstack([frame_fixed, frame_fixed_color]))
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

    return point, frame_fixed, frame_fixed_color


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


def start_thread(f):
    return Thread(target=f, daemon=True).start()

def main(cam_idx):
    if DEBUG:
        frame = cv2.imread("data/color-calib-2.png", cv2.IMREAD_COLOR)
        frame_resize = cv2.resize(frame, (int(1920/2), int(1080/2)))
        find_marker_for_id(frame_resize, 3) # 3 = red, 2 = blue
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()

    cam = cv2.VideoCapture(cam_idx)
    result, frame = cam.read()
    cv2.imwrite('data/sample-frame.png', frame)
    assert result

    shape = list(frame.shape)
    [frame_height, frame_width] = shape[:2]
    shape[0] *= 2
    shape[1] *= 3
    canvas = np.zeros(shape, dtype=np.uint8)

    global newcameramtx
    newcameramtx, _ = \
        cv2.getOptimalNewCameraMatrix(
            intrinsics['mtx'], intrinsics['dist'], (frame_width, frame_height), 1,
            (frame_width, frame_height))

    raw_history = History()
    draw_history = History()
    pen_history = History()

    drawing = Drawing()
    drawing.set_style(0)
    drawing.set_client(client_width, client_height)
    drawing.set_target(target_width, target_height, border)

    path_buffer = Queue()
    device_lock = Lock()

    # initialize plotter
    if not DRY_RUN:
        p = PlotterAxi(w=client_width, h=client_height)

        def disable_device():
            with device_lock:
                p.device.wait()
                p.up()
                #p.move(0, 0)
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

        start_thread(send_commands)

    path_start = None
    recorded_path = []
    recording = False

    def recv_commands():
        nonlocal recording
        nonlocal recorded_path

        while True:
            inp = input().strip()
            with device_lock:
                if inp == 'i':
                    p.up()
                elif inp == 'o':
                    p.down()
                elif inp == 'r':
                    print('Start recording')
                    recording = True
                    recorded_path = []
                    path_start = draw_history.last()
                elif inp == 'f':
                    print('Finish recording')
                    recording = False
                    (lx, ly) = path_start
                    recorded_path = [
                        (x - lx, y - ly)
                        for x, y in recorded_path
                    ]
                elif inp == 'a':
                    path = [
                        (x + p.x, y + p.y)
                        for x, y in recorded_path
                    ]
                    print('Applying path')
                    p.path(path)
                elif inp == 'm':
                    # print('predicted',
                    #       pen_history.last(),
                    #       'actual',
                    #       p.device.read_position())
                    p.up()
                    p.device.disable_motors()
                    print('Disabled motors')
                elif inp == 'n':
                    p.device.enable_motors()
                    p.down()
                    x, y = pen_history.last()
                    p.x = x
                    p.y = y
                    print('Enabled motors at ({}, {})'.format(x, y))
                else:
                    try:
                        print(inp)
                        n = int(inp)
                        drawing.set_style(n)
                        print('Set style to {}'.format(n))
                    except ValueError:
                        pass

    start_thread(recv_commands)

    def draw_debug_info(frame, undistorted_frame, raw_point, draw_point,
                        pen_point):
        frame_canvas = canvas[:frame_height, frame_width:(frame_width*2), :]
        frame_canvas[:,:,:] = frame[::-1, ::-1, :]

        fixed_canvas = canvas[:frame_height, (frame_width*2):, :]
        fixed_canvas[:,:,:] = undistorted_frame[::-1, ::-1, :]

        masked_canvas = canvas[frame_height:, frame_width:(frame_width*2), :]
        masked_canvas[:,:,:] = sat_img[::-1, ::-1, :]

        camera_canvas = canvas[:, frame_width:(frame_width*2), :]
        cv2.line(camera_canvas, (client_x, 0), (client_x, frame_height), (255, 255, 0), 3)
        cv2.line(camera_canvas, (client_x + client_width, 0), (client_x + client_width, frame_height), (255, 255, 0), 3)
        cv2.line(camera_canvas, (0, client_y), (frame_width, client_y), (255, 255, 0), 3)
        cv2.line(camera_canvas, (0, client_y + client_height), (frame_width, client_y + client_height), (255, 255, 0), 3)

        def paper_to_pixel(p):
            return (int(p[0] / target_width * frame_width),
                    int(p[1] / target_height * frame_height))

        if raw_point is not None and draw_point is not None:
            def raw_to_pixel(p):
                return (int(p[0] * frame_width / client_width),
                        int(p[1] * frame_height / client_height))
            # Draw raw coordinates
            raw_point = raw_to_pixel(raw_point)
            if raw_history.last() is not None:
                cv2.line(canvas, raw_to_pixel(raw_history.last()), raw_point, (255, 255, 255), 3)
            cv2.circle(canvas, raw_point, 10, (255, 255, 255), -1)

            # Draw draw coordinates
            draw_canvas = canvas[frame_height:, :frame_width, :]

            draw_color = (0, 255, 0) if recording else (255, 255, 0)
            if draw_history.last() is not None:
                cv2.line(
                    draw_canvas,
                    paper_to_pixel(draw_history.last()),
                    paper_to_pixel(draw_point),
                    draw_color,
                    3)
            cv2.circle(
                draw_canvas,
                paper_to_pixel(draw_point),
                10,
                draw_color,
                -1)

        if pen_point is not None:
            cv2.circle(fixed_canvas, paper_to_pixel(pen_point), 20,
                       (0, 255, 255), -1)

        canvas_resize = cv2.resize(
            canvas, (1920, int(canvas.shape[0] * 1920 / canvas.shape[1])))
        cv2.imshow('frame', canvas_resize)
        cv2.waitKey(30)

    while True:
        result, frame = cam.read()
        assert result

        markers = {
            col: find_marker_for_id(frame.copy(), col)
            for col in [COLOR_BLUE, COLOR_RED]
        }

        def project_raw(point):
            px, py = point

            # Put coordinates within the client box
            px -= client_x
            py -= client_y

            # Mirror x because we're behind the camera
            raw_point = (client_width - px, client_height - py)
            return raw_point

        for k, (pt, _1, _2) in markers.items():
            if pt is not None:
                markers[k] = (project_raw(pt), _1, _2)

        axi_raw_point, undistorted_frame, sat_img = markers[COLOR_BLUE]

        pen_point = None
        if axi_raw_point is not None:
            axi_draw_point = drawing.compute_draw_coordinates(*axi_raw_point)
            camera_pos = np.array([
                target_width / 2, target_height/ 2 , camera_height])
            draw_pos = np.array([axi_draw_point[0], axi_draw_point[1], 0])
            pen_point = \
                draw_pos  + (camera_pos - draw_pos) / camera_height * pen_height
            pen_point = (pen_point[0], pen_point[1])
            pen_history.shift(pen_point)

        raw_point, _, _ = markers[COLOR_RED]
        draw_point = None

        if raw_point is not None:
            px, py = raw_point
            if px >= 0 and px <= client_width and \
               py >= 0 and py <= client_height:
                draw_point = drawing.compute_draw_coordinates(*raw_point)
                styled_point = drawing.apply_style(draw_point)

                if recording:
                    recorded_path.append(draw_point)
                #path_buffer.put(styled_point)

        draw_debug_info(
            frame,
            undistorted_frame,
            raw_point,
            draw_point,
            pen_point)

        if raw_point is not None and draw_point is not None:
            raw_history.shift(raw_point)
            draw_history.shift(draw_point)


if __name__ == "__main__":
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(cam_idx)
