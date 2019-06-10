import numpy as np

class Style:
    def __init__(self, drawing):
        self.drawing = drawing

class DefaultStyle(Style):
    def transform(self, point, vec):
        return point

class MirrorStyle(Style):
    def transform(self, point, vec):
        (x, y) = point
        x_distance = self.drawing.target_width / 2 - x
        x_mod = self.drawing.target_width / 2 + x_distance
        y_mod = y
        return (x_mod, y_mod)

class HeatmapStyle(Style):
    def __init__(self, drawing):
        Style.__init__(self, drawing)
        self.pos = np.array([0, 0])

    def transform(self, point, vec):
        poi = np.array([self.drawing.target_width / 2, self.drawing.target_height / 2])
        vec = np.array(vec)
        target_point = self.pos + vec

        dist = np.sqrt(np.sum(np.square(target_point - poi)))
        vec = dist / 100 * (poi - target_point)
        final_point = target_point  + vec

        print('Current: {}, vector: {}, final: {}'.format(
            self.pos, vec, final_point))

        self.pos = final_point

        return (final_point[0], final_point[1])

class Drawing:
    def __init__(self, p, pen_history, style=0):
        self.p = p
        self.pen_history = pen_history
        self.style = 0
        self.client_width = 0
        self.client_height = 0
        self.target_width = 0
        self.target_height = 0
        self.border = 0
        self.styles = [
            DefaultStyle(self),
            MirrorStyle(self),
            HeatmapStyle(self)
        ]

    def set_style(self, style):
        self.style = style

    def set_client(self, client_width, client_height):
        self.client_width = client_width
        self.client_height = client_height

    def set_target(self, target_width, target_height, border):
        self.target_width = target_width
        self.target_height = target_height
        self.border = border

    def apply_style(self, point, vec):
        return self.styles[self.style].transform(point, vec)

    def compute_draw_coordinates(self, x, y): # x,y is in client coordinates
        x_target = x * self.target_width / self.client_width
        y_target = y * self.target_height / self.client_height
        return (x_target, y_target)

    def compute_draw_coordinates_clipped(self, x, y):
        (x_target, y_target) = self.compute_draw_coordinates(x, y)
        x_target = np.clip(x_target, self.border, self.target_width - self.border)
        y_target = np.clip(y_target, self.border, self.target_height - self.border)

        return (x_target, y_target) # x_target, y_target is in target (axi) coordinates

    def apply_transform(self, path, translate, rotate, scale):
        transform_path = path

        # scale
        ref_point = transform_path[0]
        transform_path = [ 
            (float(x - ref_point[0]) * scale + ref_point[0], 
             float(y - ref_point[1]) * scale + ref_point[1])
            for x, y in transform_path
        ]
        # print('scale')
        # print(transform_path)

        # rotate (in radians)
        c, s = np.cos(rotate), np.sin(rotate)
        rotation_matrix = np.matrix([[c, s], [-s, c]])
        transform_m = [
            np.dot(rotation_matrix, [x, y])
            for x, y in transform_path
        ]
        transform_path = [
            (m.item(0), m.item(1))
            for m in transform_m
        ]
        # print('rotate')
        # # print(transform_m)
        # print(transform_path)

        # apply style (?)
        # print('apply style')
        # print(transform_path)

        # translate such that first point is (0,0)
        ref_point = transform_path[0]
        transform_path = [
            (x - ref_point[0],
             y - ref_point[1])
            for x, y in transform_path
        ]
        # print('translate to (0,0)')
        # print(transform_path)

        return transform_path
