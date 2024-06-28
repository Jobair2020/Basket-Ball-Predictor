"""
A project to detect the path of basketball projection weather it will make up to the basket or not.
"""

import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import math
import numpy as np

# Initialize the Video
capture = cv2.VideoCapture('Videos/vid (1).mp4')
hsvVals = {'hmin': 0, 'smin': 115, 'vmin': 0, 'hmax': 15, 'smax': 255, 'vmax': 255}

# Variables
positionList = []
xList = [item for item in range(1, 1300, 2)]

# Create the color Finder object
myColorFinder = ColorFinder(False)  # trying to find the color from the image
prediction = False
# per frame
while True:
    # input image
    success, img = capture.read()
    # img = cv2.imread("Ball.png")
    img = img[0:950, :]

    # Find the color Ball
    # todo
    imgColor, mask = myColorFinder.update(img, hsvVals)

    # Find location of the ball
    # todo
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)

    if contours:
        positionList.append(contours[0]['center'])

    if positionList:
        # Polynomial Regression y = Ax^2 + Bx + C
        X = [pos[0] for pos in positionList]
        Y = [pos[1] for pos in positionList]
        A, B, C = np.polyfit(X, Y, 2)

        for i, pos in enumerate(positionList):
            # todo
            cv2.circle(imgContours, pos, 8, (0, 255, 0), cv2.FILLED)
            if i != 0:
                # todo
                cv2.line(imgContours, pos, positionList[i - 1], (0, 255, 0), 5)

        for x in xList:
            y = int(A * x * x + B * x + C)
            cv2.circle(imgContours, (x, y), 2, (255, 0, 255), cv2.FILLED)

        if len(positionList) < 10:
            # Prediction
            # x values 325 to 425 and y value 590 ax^2+bx+c = 0
            a = A
            b = B
            c = C - 590
            x = int((- b - math.sqrt(b * b - (4 * a * c))) / (2 * a))
            prediction = 330 < x < 415

        if prediction:
            cvzone.putTextRect(imgContours, "Basket", (50, 100), scale=5, thickness=5, colorR=(0, 200, 0),
                               offset=20)
        else:
            cvzone.putTextRect(imgContours, "No Basket", (50, 100), scale=5, thickness=5, colorR=(0, 0, 200),
                               offset=20)

    # Display
    # todo
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.6)
    mask2 = cv2.resize(mask, (0, 0), None, 0.7, 0.6)
    cv2.imshow("mask", mask2)
    cv2.imshow("imageContors", imgContours)
    # cv2.imshow("ballDetection",imgColor)

    cv2.waitKey(100)

# if __name__ == '__main__':
#     basket()
