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
from spline import CatmullRomSpline
from timeit import default_timer as now
from enum import Enum
from websocket_server import WebsocketServer
import json
import math

DRY_RUN = False

DEBUG = False

client_width = 1350#1920
client_x = int(1920 / 2 - client_width / 2)

client_height = 1000
client_y  = int(1080 / 2 - client_height / 2)

target_width = 11.5
target_height = 8.5

border = 0.5

intrinsics = pickle.load(open('data/intrinsics.pkl', 'rb'))
camera_height = 13
pen_height = 7

#camera_sample_rate = 1/24

newcameramtx = None

sat_low = 60
sat_high = 255
value_low = 100
value_high = 255

# frame1
    # red: [177 230 154]
    # skin: [4 96 152]
    # blue: [105 140 178]

# frame3
    # red: [178 220 115] [179 226 113]
    # skin: [4 100 143]
    # blue: [105 169 152]
    # machine blue: [116 188 46]

# frame4
    # yellow: [25 159 220]
    # board: [17 101 104] [15 66 144]

# frame5
    # yellow: [ 21 104 238] [ 30  71 255]
    # board: [ 14  83 151]
    # green: [ 77 115 129]
    # red: [178 180 160]
    # blue: [100 150 195]

boundaries = [
    ([15, 60, 190], [40, sat_high, value_high]), # yellow
    ([50, 80, 90], [85, sat_high, value_high]), # green
    ([90, 100, 120], [130, sat_high, value_high]), # blue
    ([140, 160, 60], [255, sat_high, value_high]), # red
]

server = WebsocketServer(9001)

COLOR_YELLOW = 0
COLOR_GREEN = 1
COLOR_BLUE = 2
COLOR_RED = 3

def point_dist(p1, p2):
    return np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))

def point_add(p1, p2):
    return (p1[0] + p2[0], p1[1] + p2[1])

def path_smooth(path):
    new_path = path[:1]
    for i in range(len(path)-4):
        new_path.extend(CatmullRomSpline(*path[i:i+4], nPoints=8))
    new_path.append(path[-1])
    return new_path

# returns x, y (1920, 1080) or None if can't find
def find_marker_for_id(frame, marker_id):
    frame_fixed = \
        cv2.undistort(frame, intrinsics['mtx'], intrinsics['dist'], newcameramtx)

    hsv = cv2.cvtColor(frame_fixed, cv2.COLOR_BGR2HSV)

    lower = np.array(boundaries[marker_id][0])
    upper = np.array(boundaries[marker_id][1])

    hue_color = cv2.inRange(hsv, lower, upper)
    hue_color = cv2.erode(hue_color, None, iterations=2)
    hue_color = cv2.dilate(hue_color, None, iterations=2)

    frame_fixed_color = cv2.bitwise_and(frame_fixed, frame_fixed, mask = hue_color)

    if DEBUG:
        # cv2.imshow('hue_color', hue_color)
        frame_fixed_resize = cv2.resize(frame_fixed, (int(1920/4), int(1080/4)))
        frame_fixed_color_resize = cv2.resize(frame_fixed_color, (int(1920/4), int(1080/4)))
        cv2.imshow('frame_fixed', np.hstack([frame_fixed_resize, frame_fixed_color_resize]))
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
        frame1 = cv2.imread("data/color-calib-1.png", cv2.IMREAD_COLOR)
        frame2 = cv2.imread("data/color-calib-2.png", cv2.IMREAD_COLOR)
        frame3 = cv2.imread("data/color-calib-3.png", cv2.IMREAD_COLOR)
        frame4 = cv2.imread("data/color-calib-4.png", cv2.IMREAD_COLOR) # red, blue, yellow
        frame5 = cv2.imread("data/color-calib-5.png", cv2.IMREAD_COLOR) # red, blue, yellow, green
        frame6 = cv2.imread("data/color-calib-6.png", cv2.IMREAD_COLOR) # all + pens
        # find_marker_for_id(frame1, COLOR_BLUE)
        # find_marker_for_id(frame1, COLOR_RED)
        # find_marker_for_id(frame2, COLOR_BLUE)
        # find_marker_for_id(frame2, COLOR_RED)
        # find_marker_for_id(frame3, COLOR_BLUE)
        # find_marker_for_id(frame3, COLOR_RED)
        # find_marker_for_id(frame4, COLOR_BLUE)
        # find_marker_for_id(frame4, COLOR_RED)
        # find_marker_for_id(frame4, COLOR_YELLOW)
        # find_marker_for_id(frame5, COLOR_BLUE)
        # find_marker_for_id(frame5, COLOR_RED)
        # find_marker_for_id(frame5, COLOR_YELLOW)
        # find_marker_for_id(frame5, COLOR_GREEN)
        find_marker_for_id(frame6, COLOR_BLUE)
        find_marker_for_id(frame6, COLOR_RED)
        find_marker_for_id(frame6, COLOR_YELLOW)
        find_marker_for_id(frame6, COLOR_GREEN)
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
    styled_history = History()
    pen_history = History()

    drawing = Drawing()
    drawing.set_style(1)
    drawing.set_client(client_width, client_height)
    drawing.set_target(target_width, target_height, border)

    path_buffer = Queue()
    device_lock = Lock()

    def project_draw_to_bot(draw_point):
        return (
            draw_point[0] - (target_width - draw_point[0]) * 0.02,
            draw_point[1] - (target_height - draw_point[1]) * .1
        )

    # initialize plotter
    if not DRY_RUN:
        p = PlotterAxi(w=client_width-0.5, h=client_height-0.5, border=border)

        def disable_device():
            with device_lock:
                p.device.wait()
                p.up()
                if not actions['motors'].device_reset:
                    p.move(0, 0)
                p.device.wait()
                sp.check_call(['axi', 'off'])

        atexit.register(disable_device)

        def send_commands():
            first_command = True
            while True:
                if path_buffer.qsize() > 48:
                    print('Queue is too large: {}'.format(path_buffer.qsize()))
                    path = [path_buffer.get() for _ in range(48)][::3]
                else:
                    path = [path_buffer.get() for _ in range(12)]

                path = [project_draw_to_bot(p) for p in path]

                with device_lock:
                    if first_command:
                        p.move(*path[0])
                        p.down()
                        first_command = False
                    p.path(path)

        start_thread(send_commands)


    class RecordTemplateAction:
        def __init__(self):
            self.template = []
            self.start_point = None
            self.recording = False

        def trigger(self):
            if not self.recording:
                print('Recording template')
                self.template = []
                self.start_point = draw_history.last()
                self.recording = True
            else:
                print('Finish template')
                self.recording = False
                (lx, ly) = self.start_point
                self.template = [
                    (x - lx, y - ly)
                    for x, y in self.template
                ]
                return self.template

        def on_draw(self, point):
            if self.recording:
                self.template.append(point)

    class ApplyTemplateAction:
        def __init__(self):
            self.path = []
            self.recording = False
            self.rotation = 0
            self.scale = 1

        def apply_template(self, p, translate, rotate, scale):
            transformed_path = drawing.apply_transform(
                actions['record_template'].template,
                translate, rotate, scale)
            template_path = path_smooth(transformed_path)
            template_path = actions['record_template'].template

            print('Applying template')
            origin = (p.x, p.y)
            print(origin)
            with device_lock:
                p.move(*point_add(template_path[0], origin))
                p.down()
                p.path([
                    point_add(p, origin)
                    for p in template_path
                ])
                p.up()

                # update template with transform
                actions['record_template'].template = transformed_path
                transformed_path = drawing.apply_transform(
                    actions['record_template'].template,
                    translate, rotate, scale)
                template_path = path_smooth(transformed_path)

        def trigger(self):
            translate = (0, 0)#(p.x, p.y)
            rotate = actions['apply_template'].rotation#math.pi/3#0
            scale = actions['apply_template'].scale#.8
            self.apply_template(p, translate, rotate, scale)

        def on_draw(self, point):
            self.path.append(point)

    class ApplyTemplatePathAction:
        def __init__(self):
            self.path = []
            self.recording = False
            self.rotation = 0
            self.scale = 1

        def apply_template(self, p, translate, rotate, scale):
            transformed_path = drawing.apply_transform(
                actions['record_template'].template,
                translate, rotate, scale)
            template_path = path_smooth(transformed_path)

            diameter = max([
                point_dist(p1, p2)
                for p1 in template_path
                for p2 in template_path
            ])

            draw_points = self.path[:1]
            i = 1
            while i < len(self.path):
                if point_dist(self.path[i], draw_points[-1]) >= diameter/2:
                    draw_points.append(self.path[i])
                i += 1

            print('Applying template path')
            for origin in draw_points:
                with device_lock:
                    p.move(*point_add(template_path[0], origin))
                    p.down()
                    p.path([
                        point_add(p, origin)
                        for p in template_path
                    ])
                    p.up()

                    # update template with transform
                    actions['record_template'].template = transformed_path
                    transformed_path = drawing.apply_transform(
                        actions['record_template'].template,
                        translate, rotate, scale)
                    template_path = path_smooth(transformed_path)

        def trigger(self):
            if not self.recording:
                print('Recording application path')
                self.path = []
                self.recording = True
            else:
                print('Finish application path')
                translate = (0, 0)#(p.x, p.y)
                rotate = actions['apply_template_path'].rotation#math.pi/3#0
                scale = actions['apply_template_path'].scale#.8
                self.apply_template(p, translate, rotate, scale)

        def on_draw(self, point):
            self.path.append(point)


    class MotorAction:
        def __init__(self):
            self.device_reset = False
            self.disabled = False

        def trigger(self):
            if not self.disabled:
                # print('predicted',
                #       pen_history.last(),
                #       'actual',
                #       p.device.read_position())
                with device_lock:
                    p.up()
                    p.device.disable_motors()
                    print('Disabled motors')
                    self.disabled = True
            else:
                with device_lock:
                    p.device.enable_motors()
                    p.down()
                    x, y = pen_history.last()
                    p.x = x
                    p.y = y
                    print('Enabled motors at ({}, {})'.format(x, y))
                    self.device_reset = True
                    self.disabled = False

        def on_draw(self, point):
            pass

    actions = {
        'record_template': RecordTemplateAction(),
        'apply_template': ApplyTemplateAction(),
        'apply_template_path': ApplyTemplatePathAction(),
        'motors': MotorAction(),
    }

    def recv_keyboard():
        while True:
            inp = input().strip()
            if inp == 'i':
                with device_lock:
                    p.up()
            elif inp == 'o':
                with device_lock:
                    p.down()
            else:
                try:
                    print(inp)
                    n = int(inp)
                    drawing.set_style(n)
                    print('Set style to {}'.format(n))
                except ValueError:
                    pass

                for a in actions.values():
                    a.on_keypress(inp)

    start_thread(recv_keyboard)

    def recv_websocket(client, server, message):
        data = json.loads(message)
        if 'rotation' in data:
            rotation = int(data['rotation'])
            print(rotation)
            rotation = math.radians(rotation)
            # print(rotation)
            actions['apply_template'].rotation = rotation
            actions['apply_template_path'].rotation = rotation
        if 'scale' in data:
            scale = data['scale']
            print(scale)
            actions['apply_template'].scale = scale
            actions['apply_template_path'].scale = scale
        if 'type' in data:
            server.send_message(
                client,
                json.dumps(actions[data['type']].trigger()))

    def websocket_server():
        server.set_fn_message_received(recv_websocket)
        server.run_forever()

    start_thread(websocket_server)

    def draw_debug_info(frame, undistorted_frame, sat_img,
                        raw_point, draw_point, pen_point):
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
            return (int(p[0] / target_width * client_width + client_x),
                    int(p[1] / target_height * client_height + client_y))

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

            draw_color = (0, 255, 0) if actions['record_template'].recording \
                else (255, 255, 0)
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
            px, py = paper_to_pixel(pen_point)
            cv2.circle(fixed_canvas, (px, py), 20,
                       (0, 255, 255), -1)
            cv2.putText(fixed_canvas, "{:.2f}, {:.2f}".format(*pen_point),
                        (px - 80, py+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 255, 255), 5)

            bot_point = project_draw_to_bot(pen_point)
            px, py = paper_to_pixel(bot_point)
            cv2.circle(fixed_canvas, (px, py), 20,
                       (0, 255, 0), -1)
            cv2.putText(fixed_canvas, "{:.2f}, {:.2f}".format(*bot_point),
                        (px - 80, py-50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 255, 0), 5)

        canvas_resize = cv2.resize(
            canvas, (1920, int(canvas.shape[0] * 1920 / canvas.shape[1])))
        cv2.imshow('frame', canvas_resize)
        cv2.waitKey(30)

    last_sample = now()
    while True:
        # print('Total: {:.4f}'.format(now() - last_sample))
        last_sample = now()
        # if now() - last_sample < camera_sample_rate:
        #     continue

        result, frame = cam.read()
        assert result

        start = now()
        markers = {
            col: find_marker_for_id(frame.copy(), col)
            for col in [COLOR_BLUE, COLOR_RED]
        }
        # print('Finding markers: {:.4f}'.format(now() - start))

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

        axi_raw_point, undistorted_frame, _ = markers[COLOR_BLUE]

        sat_img = np.zeros(frame.shape)
        for _1, _2, sat_img_col in markers.values():
            sat_img[:, :, :] += sat_img_col

        pen_point = None
        if axi_raw_point is not None:
            axi_draw_point = drawing.compute_draw_coordinates(*axi_raw_point)
            cx = target_width / 2
            cy = target_height / 2
            camera_pos = np.array([cx, cy, camera_height])
            draw_pos = np.array([axi_draw_point[0], axi_draw_point[1], 0])
            pen_point = \
                draw_pos  + (camera_pos - draw_pos) / camera_height * pen_height

            # # Push x closer to middle
            pen_point[0] = cx + (pen_point[0] - cx) * 1.15

            # # Increase y
            pen_point[1] += 0.2

            pen_point = (pen_point[0], pen_point[1])
            pen_history.shift(pen_point)

        raw_point, _, _ = markers[COLOR_RED]
        draw_point = None
        styled_point = None

        if raw_point is not None:
            px, py = raw_point
            if px >= 0 and px <= client_width and \
               py >= 0 and py <= client_height:
                draw_point = drawing.compute_draw_coordinates(*raw_point)
                styled_point = drawing.apply_style(draw_point)

                for a in actions.values():
                    a.on_draw(draw_point)

                # if len(draw_history.pts) < 4:
                #     path_buffer.put(styled_point)
                #     pass
                # else:
                #     interped = CatmullRomSpline(
                #         *(styled_history.pts[-3:] + [styled_point]),
                #         nPoints=4)
                #     if np.any(np.isnan(interped)):
                #         path_buffer.put(styled_history.pts[-1])
                #     else:
                #         for pt in interped:
                #             path_buffer.put(pt)

        start = now()
        draw_debug_info(
            frame,
            undistorted_frame,
            sat_img,
            raw_point,
            draw_point,
            pen_point)
        # print('Drawing debug info: {:.3f}'.format(now() - start))

        if raw_point is not None and draw_point is not None:
            raw_history.shift(raw_point)
            draw_history.shift(draw_point)
            styled_history.shift(styled_point)

if __name__ == "__main__":
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(cam_idx)
