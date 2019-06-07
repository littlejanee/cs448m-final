class Drawing:
    def __init__(self, style=0):
        self.style = 0
        self.client_width = 0
        self.client_height = 0
        self.target_width = 0
        self.target_height = 0
        self.border = 0
        
    def setstyle(self, style):
        self.style = style

    def setclient(self, client_width, client_height):
        self.client_width = client_width
        self.client_height = client_height

    def settarget(self, target_width, target_height, border):
        self.target_width = target_width
        self.target_height = target_height
        self.border = border

    def computedrawcoordinates(self, x, y): # x,y is in client coordinates
        x_mod = x
        y_mod = y

        # 0 is direct mapping

        if (self.style == 1): # mirror
            x_distance = self.client_width / 2 - x
            x_mod = self.client_width / 2 + x_distance
            y_mod = y
        elif (self.style == 2): # other things...
            x_mod = x
            y_mod = y

        x_target = x_mod * (self.target_width - self.border) / self.client_width + self.border / 2
        y_target = y_mod * (self.target_height - self.border) / self.client_height + self.border / 2

            
        return (x_target, y_target) # x_target, y_target is in target (axi) coordinates

