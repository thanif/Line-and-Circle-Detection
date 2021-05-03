import cv2
from imutils import build_montages

i1 = cv2.imread("circles_lines.png")
i2 = cv2.imread("output_lines.png")

images = []

images.append(i1)
images.append(i2)

montages = build_montages(images, (128, 196), (2, 1))

cv2.imwrite("lines_montage.png", montages[0])
