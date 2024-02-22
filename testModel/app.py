# объявление библиотек
import cv2 as cv
import numba as nb
import numpy as np

# путь до img
path_to_test = "test.jpg"

img = cv.imread(path_to_test)
cv.imshow("test_leable", img)

key = cv.waitKey(100)
if key == key == 27: # Esc
    cv.destroyAllWindows()
