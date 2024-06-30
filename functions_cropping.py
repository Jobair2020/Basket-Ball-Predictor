import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import math
import numpy as np

# Initialize the Video
capture = cv2.VideoCapture('Videos/vid (6).mp4')
hsvVals = {'hmin': 0, 'smin': 115, 'vmin': 0, 'hmax': 15, 'smax': 255, 'vmax': 255}

# Load the template image of the basket
template = cv2.imread('cropped.jpg', 0)  # Load in grayscale
w, h = template.shape[::-1]

# Variables
positionList = []
xList = [item for item in range(1, 1300, 2)]

# Create the color Finder object
myColorFinder = ColorFinder(False)  # trying to find the color from the image
prediction = False


def crop_to_template_size(img, max_loc, template_w, template_h):
    img_height, img_width = img.shape[:2]

    # Calculate the top-left corner of the cropping box
    start_x = max(0, max_loc[0])
    start_y = max(0, max_loc[1])

    # Ensure the cropping box is within image boundaries
    end_x = min(img_width, start_x + template_w)
    end_y = min(img_height, start_y + template_h)

    # Crop the image to the template size
    cropped_img = img[start_y:end_y, start_x:end_x]

    return cropped_img


# Per frame
while True:
    # Input image
    success, img = capture.read()
    if not success:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Template matching to find the basket
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Crop the image based on the matched region and template size
    img = crop_to_template_size(img, max_loc, w, h)

    # Find the color Ball
    imgColor, mask = myColorFinder.update(img, hsvVals)

    # Find location of the ball
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)

    if contours:
        positionList.append(contours[0]['center'])

    if positionList:
        # Polynomial Regression y = Ax^2 + Bx + C
        X = [pos[0] for pos in positionList]
        Y = [pos[1] for pos in positionList]
        A, B, C = np.polyfit(X, Y, 2)

        for i, pos in enumerate(positionList):
            cv2.circle(imgContours, pos, 8, (0, 255, 0), cv2.FILLED)
            if i != 0:
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
            x = int((-b - math.sqrt(b * b - (4 * a * c))) / (2 * a))
            prediction = 330 < x < 415

        if prediction:
            cvzone.putTextRect(imgContours, "Basket", (50, 100), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
        else:
            cvzone.putTextRect(imgContours, "No Basket", (50, 100), scale=5, thickness=5, colorR=(0, 0, 200), offset=20)

    # Display
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.6)
    mask2 = cv2.resize(mask, (0, 0), None, 0.7, 0.6)
    cv2.imshow("mask", mask2)
    cv2.imshow("imageContours", imgContours)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
