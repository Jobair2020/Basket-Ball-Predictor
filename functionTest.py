import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

SPREAD_VAL = 40


def spread(image, sp_x, sp_y, to_replace, replace_with):
    h, w = image.shape

    parent_map = {}
    length = 0
    last = None

    stack = [(sp_x, sp_y, 0)]
    parent_map[(sp_x, sp_y)] = None

    while stack:
        x, y, it = stack.pop()
        if image[x, y] != to_replace:
            continue

        image[x, y] = replace_with

        it += 1
        if it > length:
            length = it
            last = (x, y)

        tmp = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        indices = []

        for pt in tmp:
            indices.append(pt)

        # for pt in tmp:
        #     pt2 = (pt[0]*2, pt[1]*2)
        #     pt3 = (pt[0]*3, pt[1]*3)
        #     indices.append(pt2)
        #     indices.append(pt3)

        for dx, dy in indices:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and image[nx, ny] == to_replace:
                if (nx, ny) not in parent_map:  # Check if not already visited
                    parent_map[(nx, ny)] = (x, y)
                    stack.append((nx, ny, it))

    # Get the longest path
    points = []
    while last is not None:
        points.append(last)
        last = parent_map[last]
    points.reverse()

    return points


def get_contours(image):
    image = image.copy()
    h, w = image.shape

    pad = 10
    image = image[pad:h - pad, pad:w - pad]
    h, w = image.shape

    contours = []

    it = 1
    visited = {}

    def find_nearest_white_pixel(sx, sy):
        nonlocal visited, it
        to_it = (sx, sy)

        while to_it != None:
            queue = deque()
            queue.append(to_it)

            to_it = None
            count = 0
            while queue:
                x, y = queue.popleft()
                if visited.get((x, y)) == True:
                    continue
                count += 1

                image[x, y] = SPREAD_VAL

                indices = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
                for dx, dy in indices:
                    nx, ny = x + dx, y + dy
                    if nx < 0 or nx >= h or ny < 0 or ny >= w or image[
                        nx, ny] == SPREAD_VAL:  # SPREAD_VAL = visited already
                        continue

                    if image[nx, ny] == 255:
                        to_it = (nx, ny)
                        queue.clear()
                        break
                    if visited.get(to_it) == None:
                        queue.append((nx, ny))
                visited[(x, y)] = True

            if to_it == None:
                break

            print(f"Spreading no: {it}")
            it += 1

            points = spread(image, to_it[0], to_it[1], to_replace=255, replace_with=2 * SPREAD_VAL)
            last_pt = points[len(points) - 1]

            points = spread(image, last_pt[0], last_pt[1], to_replace=2 * SPREAD_VAL, replace_with=SPREAD_VAL)
            if (len(points) > 20):
                contours.append(points)

            # show_image("Spread result", image=image)
            to_it = points[len(points) - 1]

    for x in range(h):
        for y in range(w):
            if visited.get((x, y)) == None:
                find_nearest_white_pixel(x, y)

    return contours, image


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
    print(imgColor)

    return imgColor, mask


hsvVals = {'hmin': 0, 'smin': 115, 'vmin': 0, 'hmax': 15, 'smax': 255, 'vmax': 255}

img = cv2.imread("Ball.png")
imgColor, mask = find_color(img, hsvVals)
cv2.imshow('ball detected Color', resize(imgColor))
cv2.imshow('inRange output', resize(mask))

imgContours, contours = find_contours(img, mask, minArea=500)
imgContours_gray = cv2.cvtColor(imgContours,cv2.COLOR_BGRA2GRAY)
cv2.imshow('contour image', resize(imgContours_gray))
print(contours[0]["center"])

cv2.waitKey(0)
cv2.destroyAllWindows()
