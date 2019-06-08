import numpy as np

class Style:
    def __init__(self, drawing):
        self.drawing = drawing

class DefaultStyle(Style):
    def transform(self, point):
        return point

class MirrorStyle(Style):
    def transform(self, point):
        (x, y) = point
        x_distance = self.drawing.client_width / 2 - x
        x_mod = self.drawing.client_width / 2 + x_distance
        y_mod = y
        return (x_mod, y_mod)

class Drawing:
    def __init__(self, style=0):
        self.styles = [
            DefaultStyle(self),
            MirrorStyle(self)
        ]
        self.style = 0
        self.client_width = 0
        self.client_height = 0
        self.target_width = 0
        self.target_height = 0
        self.border = 0

    def set_style(self, style):
        self.style = style

    def set_client(self, client_width, client_height):
        self.client_width = client_width
        self.client_height = client_height

    def set_target(self, target_width, target_height, border):
        self.target_width = target_width
        self.target_height = target_height
        self.border = border

    def apply_style(self, point):
        return self.styles[self.style].transform(point)

    def compute_draw_coordinates(self, x, y): # x,y is in client coordinates
        x_target = x * self.target_width / self.client_width
        y_target = y * self.target_height / self.client_height
        return (x_target, y_target)

    def compute_draw_coordinates_clipped(self, x, y):
        (x_target, y_target) = self.compute_draw_coordinates(x, y)
        x_target = np.clip(x_target, self.border, self.target_width - self.border)
        y_target = np.clip(y_target, self.border, self.target_height - self.border)

        return (x_target, y_target) # x_target, y_target is in target (axi) coordinates
