#The main file

# python imports
import cv2
import math # необходимо убрать
import numpy as np
from pyModbusTCP.server import ModbusServer, DataBank # подключение к серверу
from time import sleep

# добавле ние финкций
from PerspectiveTransform import PerspectiveCorrecter
from MarkerPose import MarkerPose
from MarkerTracker import MarkerTracker

# parametr ?
show_image = True
list_of_markers_to_find = [4]  # Список проверяемых ступеней ?
get_images_to_flush_cam_buffer = 5
server = ModbusServer("10.131.115.144", 1234, no_block=True) # подключение к серверу


class CameraDriver:
    def __init__(self, marker_orders=[4], default_kernel_size=21, scaling_parameter=2500, downscale_factor=1):

        if show_image is True:
            cv2.namedWindow('filterdemo', cv2.WINDOW_AUTOSIZE)

        # Выбор камеры не обходимо вынести это в отдельныое пространство
        self.camera = cv2.VideoCapture(0)   # выбор первой камеры
        self.set_camera_resolution()        # устанавливает разрешение камеры

        # Промежуточные переменные
        self.current_frame = None   # текущий кадр
        self.processed_frame = None # обработанныей кадр
        self.running = True         # выполнение ?
        self.downscale_factor = downscale_factor # понижение чего-то

        # Treker
        self.trackers = []      # переменная накопления для значений temp
        self.old_locations = [] # Прошлая позиция ?

###############################################################################################################
        for marker_order in marker_orders:
            temp = MarkerTracker(marker_order, default_kernel_size, scaling_parameter)
            temp.track_marker_with_missing_black_leg = False
            self.trackers.append(temp)
            self.old_locations.append(MarkerPose(None, None, None, None, None))
################################################################################################################

    def set_camera_resolution(self):
	    #zadaetsa razreshenie
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def get_image(self):
        # Poluchenie izobrazenia
        for k in range(get_images_to_flush_cam_buffer):
            self.current_frame = self.camera.read()[1]

    # обработка кадра
    def process_frame(self):
        self.processed_frame = self.current_frame
        # Poisk markera
        frame_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2GRAY) #perehod v seroe
        # меняем размер
        reduced_image = cv2.resize(frame_gray, (0, 0), fx=1.0/self.downscale_factor, fy=1.0 / self.downscale_factor)
        for k in range(len(self.trackers)):
            # Previous marker location is unknown, search in the entire image.
            self.current_frame = self.trackers[k].locate_marker(reduced_image)
            self.old_locations[k] = self.trackers[k].pose
            self.old_locations[k].scale_position(self.downscale_factor)

    # Рисуем контуры
    def draw_detected_markers(self):
	    #oboznachit marker
        for k in range(len(self.trackers)):
            xm = self.old_locations[k].x
            ym = self.old_locations[k].y
            orientation = self.old_locations[k].theta
            if self.old_locations[k].quality > 0.3:
            #    cv2.circle(self.processed_frame, (xm, ym), 4, (55, 55, 255), 1)
            #else:
                cv2.circle(self.processed_frame, (xm, ym), 4, (55, 55, 255), 3)

                xm2 = int(xm + 50 * math.cos(orientation))
                ym2 = int(ym + 50 * math.sin(orientation))
                cv2.line(self.processed_frame, (xm, ym), (xm2, ym2), (255, 0, 0), 2)

    def show_processed_frame(self):
	    #vivod izobrazenia
        if show_image is True:
            cv2.imshow('filterdemo', self.processed_frame)

    def reset_all_locations(self):
        #poisk lokacii markera
        for k in range(len(self.trackers)):
            self.old_locations[k] = MarkerPose(None, None, None, None, None)

    def handle_keyboard_events(self):
        if show_image is True:
            # Prerivanie Esc
            key = cv2.waitKey(100)
            key = key & 0xff
            if key == 27:  # Esc
                self.running = False

    def return_positions(self):
        return self.old_locations

def main():

    cd = CameraDriver(list_of_markers_to_find, default_kernel_size=55, scaling_parameter=1000, downscale_factor=1)  # Best in robolab.
    # cd = ImageDriver(list_of_markers_to_find, defaultKernelSize = 21)

    # Kalibrovka
    reference_point_locations_in_image = [[1328, 340], [874, 346], [856, 756], [1300, 762]]
    reference_point_locations_in_world_coordinates = [[0, 0], [300, 0], [300, 250], [0, 250]]
    perspective_corrector = PerspectiveCorrecter(reference_point_locations_in_image,
                                                 reference_point_locations_in_world_coordinates)

    print("Start server...")
    #server.start()
    print("Server is online")
    state = [0]

    while cd.running:
        cd.get_image()
        cd.process_frame()
        cd.draw_detected_markers()
        cd.show_processed_frame()
        cd.handle_keyboard_events()
        y = cd.return_positions()


        for k in range(len(y)):
            try:
                # pose_corrected = perspective_corrector.convertPose(y[k])
                pose_corrected = y[k]
		#DataBank.set_words(1, [int(pose_corrected.x)])
		#DataBank.set_words(2, [int(pose_corrected.y)])
		        #pechat
                print("%8.3f %8.3f %8.3f %8.3f %s" % (pose_corrected.x,
                                                        pose_corrected.y,
                                                        pose_corrected.theta,
                                                        pose_corrected.quality,
                                                        pose_corrected.order))
            except Exception as e:
                print("%s" % e)

    print("Stopping")


main()
