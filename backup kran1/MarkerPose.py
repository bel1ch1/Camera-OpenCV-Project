# Класс установки позиции

#parametri ?
class MarkerPose:
    def __init__(self, x, y, theta, quality, order = None):
        # Координаты маркера ?
        self.x = x
        self.y = y
        self.theta = theta
        self.quality = quality #качество чего-то ?
        self.order = order #количество каких-то ступеней ?

    # скорее всего перенос позиции с учетом множителя ?
    def scale_position(self, scale_factor):
        self.x = self.x * scale_factor
        self.y = self.y * scale_factor
