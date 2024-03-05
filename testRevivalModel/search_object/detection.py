# python imports
import cv2
import math
import numpy as np

show_image = True
list_of_markers_to_find = [4]
get_images_to_flush_cam_buffer = 5



class CameraDriver:
    def __init__(self, marker_orders=[4], default_kernel_size=21, scaling_parameter=2500, downscale_factor=1):

        if show_image:
            cv2.namedWindow('filterdemo', cv2.WINDOW_AUTOSIZE)

        # Выбор камеры не обходимо вынести это в отдельныое пространство
        self.camera = cv2.VideoCapture(0)   # выбор первой камеры
        # Размер зафвата камеры
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Промежуточные переменные
        self.current_frame = None   # текущий кадр
        self.processed_frame = None # обработанныей кадр
        self.running = True         # выполнение ?
        self.downscale_factor = downscale_factor # понижение чего-то

        # Treker
        self.trackers = []      # переменная накопления для значений temp
        self.old_locations = [] # Прошлая позиция

        for marker_order in marker_orders: # for i in range([4])
            temp = MarkerTracker(marker_order, default_kernel_size, scaling_parameter)
            temp.track_marker_with_missing_black_leg = False
            self.trackers.append(temp)
            self.old_locations.append(MarkerPose(None, None, None, None, None))


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
        if show_image:
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

### дальше идет магия (не ходить)
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


class MarkerTracker:
    def __init__(self, order, kernel_size, scale_factor):
        self.kernel_size = kernel_size
        (kernel_real, kernel_imag) = self.generate_symmetry_detector_kernel(order, kernel_size)

        self.order = order
        self.mat_real = kernel_real / scale_factor
        self.mat_imag = kernel_imag / scale_factor

        self.frame_real = None
        self.frame_imag = None
        self.last_marker_location = None
        self.orientation = None
        self.track_marker_with_missing_black_leg = True

        # Sozdanie yadra
        (kernel_remove_arm_real, kernel_remove_arm_imag) = self.generate_symmetry_detector_kernel(1, self.kernel_size)
        self.kernelComplex = np.array(kernel_real + 1j*kernel_imag, dtype=complex)
        self.KernelRemoveArmComplex = np.array(kernel_remove_arm_real + 1j*kernel_remove_arm_imag, dtype=complex)

        # Izmerenie kachestva
        absolute = np.absolute(self.kernelComplex)
        self.threshold = 0.4*absolute.max()
        self.quality = None
        self.y1 = int(math.floor(float(self.kernel_size)/2))
        self.y2 = int(math.ceil(float(self.kernel_size)/2))
        self.x1 = int(math.floor(float(self.kernel_size)/2))
        self.x2 = int(math.ceil(float(self.kernel_size)/2))

        # Info o locacii markera
        self.pose = None


    # Simmetrichni detektor
    @staticmethod
    def generate_symmetry_detector_kernel(order, kernel_size):

        value_range = np.linspace(-1, 1, kernel_size)
        temp1 = np.meshgrid(value_range, value_range)
        kernel = temp1[0] + 1j * temp1[1]

        magnitude = abs(kernel)
        kernel = np.power(kernel, order)
        kernel = kernel * np.exp(-8 * magnitude ** 2)

        return np.real(kernel), np.imag(kernel)


    # Lokalizacia markera
    def locate_marker(self, frame):
        assert len(frame.shape) == 2, "Input image is not a single channel image."
        self.frame_real = frame.copy()
        self.frame_imag = frame.copy()

        # Rasschet svertki
        self.frame_real = cv2.filter2D(self.frame_real, cv2.CV_32F, self.mat_real)
        self.frame_imag = cv2.filter2D(self.frame_imag, cv2.CV_32F, self.mat_imag)
        frame_real_squared = cv2.multiply(self.frame_real, self.frame_real, dtype=cv2.CV_32F)
        frame_imag_squared = cv2.multiply(self.frame_imag, self.frame_imag, dtype=cv2.CV_32F)
        self.frame_sum_squared = cv2.add(frame_real_squared, frame_imag_squared, dtype=cv2.CV_32F)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(self.frame_sum_squared)
        self.last_marker_location = max_loc
        self.determine_marker_orientation(frame)
        self.determine_marker_quality(frame)

        self.pose = MarkerPose(max_loc[0], max_loc[1], self.orientation, self.quality, self.order)
        return self.pose


    # Raspoznavanie
    def determine_marker_orientation(self, frame):
        (xm, ym) = self.last_marker_location
        real_value = self.frame_real[ym, xm]
        imag_value = self.frame_imag[ym, xm]
        self.orientation = (math.atan2(-real_value, imag_value) - math.pi / 2) / self.order

        max_value = 0
        max_orientation = 0
        search_distance = self.kernel_size / 3
        for k in range(self.order):
            orient = self.orientation + 2 * k * math.pi / self.order
            xm2 = int(xm + search_distance * math.cos(orient))
            ym2 = int(ym + search_distance * math.sin(orient))
            try:
                intensity = frame[ym2, xm2]
                if intensity > max_value:
                    max_value = intensity
                    max_orientation = orient
            except Exception as e:
                print("determineMarkerOrientation: error: %d %d %d %d" % (ym2, xm2, frame.shape[1], frame.shape[0]))
                print(e)
                pass

        self.orientation = self.limit_angle_to_range(max_orientation)


    @staticmethod
    def limit_angle_to_range(angle):
        while angle < math.pi:
            angle += 2 * math.pi
        while angle > math.pi:
            angle -= 2 * math.pi
        return angle


    # Izmerenie kachestva
    def determine_marker_quality(self, frame):
        (bright_regions, dark_regions) = self.generate_template_for_quality_estimator()
        # cv2.imshow("bright_regions", 255*bright_regions)
        # cv2.imshow("dark_regions", 255*dark_regions)

        try:
            frame_img = self.extract_window_around_maker_location(frame)
            (bright_mean, bright_std) = cv2.meanStdDev(frame_img, mask=bright_regions)
            (dark_mean, dark_std) = cv2.meanStdDev(frame_img, mask=dark_regions)

            mean_difference = bright_mean - dark_mean
            normalised_mean_difference = mean_difference / (0.5*bright_std + 0.5*dark_std)
            # perevod the normalised_mean_differences k [0, 1]
            temp_value_for_quality = 1 - 1/(1 + math.exp(0.75*(-7+normalised_mean_difference)))
            self.quality = temp_value_for_quality
        except Exception as e:
            print("error")
            print(e)
            self.quality = 0.0
            return


    # Virezanie markera
    def extract_window_around_maker_location(self, frame):
        (xm, ym) = self.last_marker_location
        frame_tmp = np.array(frame[ym - self.y1:ym + self.y2, xm - self.x1:xm + self.x2])
        frame_img = frame_tmp.astype(np.uint8)
        return frame_img


    # Sozdanie shablona dla kachestva
    def generate_template_for_quality_estimator(self):
        phase = np.exp((self.limit_angle_to_range(-self.orientation)) * 1j)
        angle_threshold = 3.14 / (2 * self.order)
        t3 = np.angle(self.KernelRemoveArmComplex * phase) < angle_threshold
        t4 = np.angle(self.KernelRemoveArmComplex * phase) > -angle_threshold

        signed_mask = 1 - 2 * (t3 & t4)
        adjusted_kernel = self.kernelComplex * np.power(phase, self.order)
        if self.track_marker_with_missing_black_leg:
            adjusted_kernel *= signed_mask
        bright_regions = (adjusted_kernel.real < -self.threshold).astype(np.uint8)
        dark_regions = (adjusted_kernel.real > self.threshold).astype(np.uint8)

        return bright_regions, dark_regions

if __name__ == "__main__":
    cd = CameraDriver(list_of_markers_to_find, default_kernel_size=55, scaling_parameter=1000, downscale_factor=1)  # Best in robolab.
    # cd = ImageDriver(list_of_markers_to_find, defaultKernelSize = 21)

    # Kalibrovka
    reference_point_locations_in_image = [[1328, 340], [874, 346], [856, 756], [1300, 762]]
    reference_point_locations_in_world_coordinates = [[0, 0], [300, 0], [300, 250], [0, 250]]
    #perspective_corrector = PerspectiveCorrecter(reference_point_locations_in_image,
                                                 #reference_point_locations_in_world_coordinates)

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


        # for k in range(len(y)):
        #     try:
        #         # pose_corrected = perspective_corrector.convertPose(y[k])
        #         pose_corrected = y[k]
		# DataBank.set_words(1, [int(pose_corrected.x)])
		# DataBank.set_words(2, [int(pose_corrected.y)])
		#         #pechat
        #         print("%8.3f %8.3f %8.3f %8.3f %s" % (pose_corrected.x,
        #                                                 pose_corrected.y,
        #                                                 pose_corrected.theta,
        #                                                 pose_corrected.quality,
        #                                                 pose_corrected.order))
        #     except Exception as e:
        #         print("%s" % e)

    print("Stopping")
