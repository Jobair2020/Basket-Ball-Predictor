# not done for video 6
import cv2
import cvzone
import numpy as np
import math

# Initialize the Video
capture = cv2.VideoCapture('Videos/vid (6).mp4')
hsvVals = {'hmin': 0, 'smin': 115, 'vmin': 0, 'hmax': 15, 'smax': 255, 'vmax': 255}
# hsvVals = {'hmin': 0, 'smin': 130, 'vmin': 0, 'hmax': 15, 'smax': 255, 'vmax': 255}
# Variables
positionList = []
xList = [item for item in range(1, 1300, 2)]


# User-defined function for finding color
def find_color(img, hsvVals):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvVals['hmin'], hsvVals['smin'], hsvVals['vmin']])
    upper = np.array([hsvVals['hmax'], hsvVals['smax'], hsvVals['vmax']])
    mask = cv2.inRange(hsv, lower, upper)
    imgColor = cv2.bitwise_and(img, img, mask=mask)
    return imgColor, mask

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


def find_contours(img, mask, minArea=500):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > minArea]
    contourDetails = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        center = (int(x + w / 2), int(y + h / 2))
        contourDetails.append({'area': area, 'bbox': (x, y, w, h), 'center': center})
    imgContours = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)
    return imgContours, contourDetails

def find_contours2(img, mask, minArea=500):
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = cv2.Canny(mask, 100, 200)
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
    return imgContours, contourDetails


# Main loop
while True:
    success, img = capture.read()
    if not success:
        break
    img = img[0:950, :]

    # Find the color Ball using the user-defined function
    imgColor, mask = find_color(img, hsvVals)

    # Find location of the ball using the user-defined function
    # imgContours, contours = find_contours(img, mask, minArea=500)
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)

    if contours:
        positionList.append(contours[0]['center'])
    prediction = 0
    if positionList:
        # Polynomial Regression y = Ax^2 + Bx + C
        X = [pos[0] for pos in positionList]
        Y = [pos[1] for pos in positionList]
        A, B, C = np.polyfit(X, Y, 2)

        for x in xList:
            y = int(A * x * x + B * x + C)
            cv2.circle(imgContours, (x, y), 2, (255, 0, 255), cv2.FILLED)

        if len(positionList) < 10:

            for i, pos in enumerate(positionList):
                cv2.circle(imgContours, pos, 8, (0, 255, 0), cv2.FILLED)
                if i != 0:
                    cv2.line(imgContours, pos, positionList[i - 1], (0, 255, 0), 5)

            # Prediction
            a = A
            b = B
            c = C - 590
            x = int((-b - math.sqrt(b * b - (4 * a * c))) / (2 * a))
            prediction = 330 < x < 415


        if prediction:
            cv2.putText(imgContours, "Basket", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 0), 5)
        else:
            cv2.putText(imgContours, "No Basket", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 200), 5)

    # Display
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.6)
    mask = cv2.resize(mask, (0, 0), None, 0.7, 0.6)
    imgColor = cv2.resize(imgColor, (0, 0), None, 0.7, 0.6)
    cv2.imshow("mask", mask)
    # cv2.imshow("color found",imgColor)
    cv2.imshow("imageContours", imgContours)

    cv2.waitKey(100)
