"""
A project to detect the path of basketball projection weather it will make up to the basket or not.
"""
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import math
import numpy as np

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

def find_color(img, hsvVals):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvVals['hmin'], hsvVals['smin'], hsvVals['vmin']])
    upper = np.array([hsvVals['hmax'], hsvVals['smax'], hsvVals['vmax']])

    mask = np.all(hsv >= lower, axis=2) & np.all(hsv <= upper, axis=2)
    mask = (mask * 255).astype(np.uint8)

    # mask = cv2.inRange(hsv, lower, upper)
    imgColor = cv2.bitwise_and(img, img, mask=mask)
    return imgColor, mask

def find_contours(img, mask, minArea=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > minArea]
    contourDetails = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        center = (int(x + w / 2), int(y + h / 2))
        contourDetails.append({'area': area, 'bbox': (x, y, w, h), 'center': center})
    imgContours = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)
    return imgContours, contourDetails


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


# Main code
capture = cv2.VideoCapture('Videos/vid (4).mp4')
hsvVals = {'hmin': 0, 'smin': 115, 'vmin': 0, 'hmax': 15, 'smax': 255, 'vmax': 255}

template = cv2.imread('cropped.jpg', 0)  # Load in grayscale
w, h = template.shape[::-1]
# print(f"width:{w} and height:{h}")

# Variables
positionList = []
xList = [item for item in range(1, w, 2)]

# Create the color Finder object
myColorFinder = ColorFinder(False)  # trying to find the color from the image
prediction = False

# per frame
while True:
    # input image
    success, img = capture.read()
    # img = cv2.imread("Ball.png")
    # img = img[0:950, :]

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    img = crop_to_template_size(img, max_loc, w, h)

    # Find the color Ball
    imgColor, mask = find_color(img, hsvVals)

    # Find location of the ball
    # imgContours, contours = cvzone.findContours(img, mask, minArea=500)
    imgContours, contours = find_contours(img, mask, minArea=500)
    # imgContours, contours = find_contours2(img, mask, minArea=500)

    if contours:
        positionList.append(contours[0]['center'])

    if positionList:
        # Polynomial Regression y = Ax^2 + Bx + C
        X = [pos[0] for pos in positionList]
        Y = [pos[1] for pos in positionList]
        A, B, C = np.polyfit(X, Y, 2)

        #for showing the regression line.
        for x in xList:
            y = int(A * x * x + B * x + C)
            cv2.circle(imgContours, (x, y), 2, (255, 0, 255), cv2.FILLED)

        # will check only 9 center positions to predict the basket.
        if len(positionList) < 10:
            for i, pos in enumerate(positionList):
                cv2.circle(imgContours, pos, 8, (0, 255, 0), cv2.FILLED) #green
                if i != 0:
                    cv2.line(imgContours, pos, positionList[i - 1], (0, 255, 0), 5)
            # Prediction
            # x values 325 to 425 and y value 590 ax^2+bx+c = 0
            a = A
            b = B
            c = C - 590
            x = int((- b - math.sqrt(b * b - (4 * a * c))) / (2 * a))
            prediction = 330 < x < 415
            print(f"x : {x} , prediction: {prediction}")


        if prediction:
            cvzone.putTextRect(imgContours, "Basket", (50, 100), scale=5, thickness=5, colorR=(0, 200, 0),
                               offset=20)
        else:
            cvzone.putTextRect(imgContours, "No Basket", (50, 100), scale=5, thickness=5, colorR=(0, 0, 200),
                               offset=20)
        # if len(positionList) == 9:
        #     cv2.imwrite("with_not_basket.png",imgContours)



    # Display
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.6)
    mask = cv2.resize(mask, (0, 0), None, 0.7, 0.6)
    imgColor = cv2.resize(imgColor, (0, 0), None, 0.7, 0.6)

    cv2.imshow("mask", mask)
    cv2.imshow("ballDetection",imgColor)
    cv2.imshow("imageContors", imgContours)


    cv2.waitKey(100)



capture.release()
cv2.destroyAllWindows()
# if __name__ == '__main__':
#     basket()
