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


DRY_RUN = True

client_width = 1350#1920
client_x = int(1920 / 2 - client_width / 2)

client_height = 1000
client_y  = int(1080 / 2 - client_height / 2)

target_width = 11.0
target_height = 8.5

border = 1.5

intrinsics = pickle.load(open('data/intrinsics.pkl', 'rb'))

# returns x, y (1920, 1080) or None if can't find
def find_marker(frame, newcameramtx):
    undistorted_frame = \
        cv2.undistort(frame, intrinsics['mtx'], intrinsics['dist'], newcameramtx)

    hsv = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)
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

    return point, undistorted_frame, sat_bright


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
    cam = cv2.VideoCapture(cam_idx)
    result, frame = cam.read()
    cv2.imwrite('data/sample-frame.png', frame)
    assert result

    shape = list(frame.shape)
    [frame_height, frame_width] = shape[:2]
    shape[0] *= 2
    shape[1] *= 3
    canvas = np.zeros(shape, dtype=np.uint8)

    newcameramtx, _ = \
        cv2.getOptimalNewCameraMatrix(
            intrinsics['mtx'], intrinsics['dist'], (frame_width, frame_height), 1,
            (frame_width, frame_height))

    raw_history = History()
    draw_history = History()

    drawing = Drawing()
    drawing.set_style(0)
    drawing.set_client(client_width, client_height)
    drawing.set_target(target_width, target_height, border)

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

        start_thread(send_commands)

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
                elif inp == 'f':
                    print('Finish recording')
                    recording = False
                elif inp == 'a':
                    print('Applying')
                    path = [
                        (x - p.x, y - p.y)
                        for x, y in recorded_path
                    ]
                    p.path(path)
                elif inp == 'm':
                    p.up()
                    p.device.disable_motors()
                elif inp == 'n':
                    p.device.enable_motors()
                    p.down()
                    x, y = draw_history.last()
                    p.x = x
                    p.y = y
                else:
                    try:
                        print(inp)
                        n = int(inp)
                        drawing.set_style(n)
                        print('Set style to {}'.format(n))
                    except ValueError:
                        pass

    start_thread(recv_commands)

    def draw_debug_info(frame, undistorted_frame, raw_point, draw_point):
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

            def paper_to_pixel(p):
                return (int(p[0] / target_width * frame_width),
                        int(p[1] / target_height * frame_height))

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

        canvas[:frame_height, frame_width:(frame_width*2), :] = frame[::-1, ::-1, :]
        canvas[:frame_height, (frame_width*2):, :] = undistorted_frame[::-1, ::-1, :]
        canvas[frame_height:, frame_width:(frame_width*2), :] = \
            np.expand_dims(sat_img, axis=2)[::-1, ::-1, :]

        camera_canvas = canvas[:, frame_width:(frame_width*2), :]
        cv2.line(camera_canvas, (client_x, 0), (client_x, frame_height), (255, 255, 0), 3)
        cv2.line(camera_canvas, (client_x + client_width, 0), (client_x + client_width, frame_height), (255, 255, 0), 3)
        cv2.line(camera_canvas, (0, client_y), (frame_width, client_y), (255, 255, 0), 3)
        cv2.line(camera_canvas, (0, client_y + client_height), (frame_width, client_y + client_height), (255, 255, 0), 3)

        canvas_resize = cv2.resize(
            canvas, (1920, int(canvas.shape[0] * 1920 / canvas.shape[1])))
        cv2.imshow('frame', canvas_resize)
        cv2.waitKey(30)

    while True:
        result, frame = cam.read()
        assert result

        raw_point, undistorted_frame, sat_img = find_marker(
            frame.copy(), newcameramtx)
        draw_point = None

        if raw_point is not None:
            px, py = raw_point
            if px >= client_x and px <= client_x + client_width and \
               py >= client_y and py <= client_y + client_height:

                # Put coordinates within the client box
                px -= client_x
                py -= client_y

                # Mirror x because we're behind the camera
                raw_point = (client_width - px, client_height - py)
                draw_point = drawing.compute_draw_coordinates(*raw_point)
                styled_point = drawing.apply_style(draw_point)

                if recording:
                    recorded_path.append(draw_point)
                #path_buffer.put(styled_point)

        draw_debug_info(frame, undistorted_frame, raw_point, draw_point)

        if raw_point is not None and draw_point is not None:
            raw_history.shift(raw_point)
            draw_history.shift(draw_point)


if __name__ == "__main__":
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(cam_idx)
