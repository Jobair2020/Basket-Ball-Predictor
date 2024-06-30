import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import cvzone
from cvzone.ColorModule import ColorFinder


def find_contours_scratch(binary_image):
    # Ensure binary_image is binary
    binary_image = binary_image.astype(np.uint8)
    contours = []
    height, width = binary_image.shape

    def get_neighbors(x, y):
        neighbors = [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x + 1, y), (x + 1, y + 1), (x, y + 1), (x - 1, y + 1),
                     (x - 1, y)]
        return [(nx, ny) for nx, ny in neighbors if 0 <= nx < width and 0 <= ny < height]

    def trace_contour(start):
        contour = []
        current = start
        while True:
            contour.append(current)
            binary_image[current[1], current[0]] = 0  # Mark as visited
            neighbors = get_neighbors(*current)
            for neighbor in neighbors:
                if binary_image[neighbor[1], neighbor[0]] > 0:
                    current = neighbor
                    break
            else:
                break
        return contour

    for y in range(height):
        for x in range(width):
            if binary_image[y, x] > 0:
                contours.append(trace_contour((x, y)))

    return contours


def draw_contours_scratch(image, contours):
    output_image = image.copy()
    for contour in contours:
        for point in contour:
            cv2.circle(output_image, point, 1, (0, 255, 0), -1)
    return output_image


def find_contours(img, mask, minArea=500):
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # mask = cv2.Canny(mask, 100, 200)
    contours = find_contours_scratch(mask)
    # Filter contours by area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(np.array(cnt)) > minArea]
    # Prepare details for output
    contourDetails = []
    for cnt in filtered_contours:
        cnt_array = np.array(cnt)
        area = cv2.contourArea(cnt_array)
        x, y, w, h = cv2.boundingRect(cnt_array)
        center = (int(x + w / 2), int(y + h / 2))
        contourDetails.append({'area': area, 'bbox': (x, y, w, h), 'center': center})
    # Draw contours from scratch
    contours_ = [np.array(contour) for contour in contours]
    imgContours = cv2.drawContours(img.copy(), contours_, -1, (0, 255, 0), 3)
    return imgContours, contourDetails, contours


def resize(img):
    img1 = cv2.resize(img, (0, 0), None, 0.7, 0.6)
    return img1


def find_color(img, hsvVals):
    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create mask
    lower = np.array([hsvVals['hmin'], hsvVals['smin'], hsvVals['vmin']])
    upper = np.array([hsvVals['hmax'], hsvVals['smax'], hsvVals['vmax']])
    mask = cv2.inRange(hsv, lower, upper)

    # Apply the mask on the original image
    imgColor = cv2.bitwise_and(img, img, mask=mask)

    return imgColor, mask


# hsvVals = {'hmin': 0, 'smin': 115, 'vmin': 0, 'hmax': 15, 'smax': 255, 'vmax': 255}
hsvVals = {'hmin': 0, 'smin': 130, 'vmin': 0, 'hmax': 15, 'smax': 255, 'vmax': 255}

img = cv2.imread("cropped_with_ball.png")
imgColor, mask = find_color(img, hsvVals)

# cv2.imshow('mask output', resize(mask))
# cv2.imshow('ball detected Color', resize(imgColor))

imgContours, contours, contour_points = find_contours(img, mask, minArea=500)
print(contour_points)

imgContours_gray = cv2.cvtColor(imgContours, cv2.COLOR_BGRA2GRAY)
cv2.imshow('contour image', resize(imgContours))
# print(contours[0]["center"])


# for contour in contour_points:
#     # Draw the contour lines
#     plt.plot(contour[:, 0, 0], contour[:, 0, 1], color='blue')
# Draw the contour points
# for point in contour:
# plt.scatter(point[0][0], point[0][1], color='red')
# plt.gca().invert_yaxis()  # Invert y-axis to match OpenCV coordinate system
# plt.show()


for contour in contour_points:
    contour = np.array(contour)
    plt.scatter(contour[:, 0], contour[:, 1], color='green', linewidth=2)
plt.gca().invert_yaxis()  # Invert y-axis to match OpenCV coordinate system
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
