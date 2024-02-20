#parametri
class MarkerPose:
    def __init__(self, x, y, theta, quality, order = None):
        #Koordinati markera
        self.x = x
        self.y = y
        self.theta = theta
        self.quality = quality #kachestvo
        self.order = order #kol-vo stupeni

    def scale_position(self, scale_factor):
        self.x = self.x * scale_factor
        self.y = self.y * scale_factor


