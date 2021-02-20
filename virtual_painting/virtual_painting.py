import numpy as np
import cv2
from collections import deque

# upper and lower for blue pen
blue_lower = np.array([110, 70, 70])
blue_upper = np.array([130, 255, 255])

# 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

# colors in arrays
blue_points = [deque(maxlen=512)]
green_points = [deque(maxlen=512)]
red_points = [deque(maxlen=512)]
yellow_points = [deque(maxlen=512)]
black_points = [deque(maxlen=512)]


# index for each colors
blu_index = green_index = red_index = yellow_index = black_index = 0

# blue, green, red and yellow
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (0, 0, 0)]
color_index = 0

#  paint interface
paint_window = np.zeros((500, 680, 3)) + 255
paint_window = cv2.rectangle(paint_window, (20, 1), (110, 50), (0, 0, 0), 2)
paint_window = cv2.rectangle(paint_window, (130, 1), (220, 50), colors[0], -1)
paint_window = cv2.rectangle(paint_window, (250, 1), (340, 50), colors[1], -1)
paint_window = cv2.rectangle(paint_window, (360, 1), (450, 50), colors[2], -1)
paint_window = cv2.rectangle(paint_window, (470, 1), (560, 50), colors[3], -1)
paint_window = cv2.rectangle(paint_window, (580, 1), (670, 50), colors[4], -1)
cv2.putText(paint_window, "CLEAR ALL", (22, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paint_window, "BLUE", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paint_window, "GREEN", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paint_window, "RED", (390, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paint_window, "YELLOW", (485, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)
cv2.putText(paint_window, "BLACK", (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 250, 250), 2, cv2.LINE_AA)

cv2.namedWindow("Paint", cv2.WINDOW_AUTOSIZE)

# load the video
cam = cv2.VideoCapture(0)

# lightness
cam.set(10, 100)

# capturing webcam
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (680, 500), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # coloring options to the frame
    frame = cv2.rectangle(frame, (20, 1), (110, 50), (122, 122, 122), -1)
    frame = cv2.rectangle(frame, (130, 1), (220, 50), colors[0], -1)
    frame = cv2.rectangle(frame, (250, 1), (340, 50), colors[1], -1)
    frame = cv2.rectangle(frame, (360, 1), (450, 50), colors[2], -1)
    frame = cv2.rectangle(frame, (470, 1), (560, 50), colors[3], -1)
    frame = cv2.rectangle(frame, (580, 1), (670, 50), colors[4], -1)

    cv2.putText(frame, "CLEAR ALL", (22, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (390, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (485, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLACK", (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 250, 250), 2, cv2.LINE_AA)

    if not ret:
        break

    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)             # threshold
    blue_mask = cv2.erode(blue_mask, kernel, iterations=2)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel, iterations=1)

    # contours in the image
    contours, _ = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # check if contours were found
    if len(contours) > 0:
        # when it detects the color create around it a bounding circle
        cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        # radius of the circle
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)

        # circle around the contour
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

        #  moments to calculate the center of the contour
        moments = cv2.moments(cnt)
        center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))

        # tracking coordinates of each and every point
        if center[1] <= 65:
            if 20 <= center[0] <= 110:  # Clear All
                blue_points = [deque(maxlen=512)]
                green_points = [deque(maxlen=512)]
                red_points = [deque(maxlen=512)]
                yellow_points = [deque(maxlen=512)]
                black_points = [deque(maxlen=512)]

                blu_index = green_index = red_index = yellow_index = black_index = 0

                paint_window[50:, :, :] = 255
            elif 130 <= center[0] <= 220:
                color_index = 0  # blue
            elif 250 <= center[0] <= 340:
                color_index = 1  # green
            elif 360 <= center[0] <= 450:
                color_index = 2  # ged
            elif 470 <= center[0] <= 560:
                color_index = 3  # yellow
            elif 580 <= center[0] <= 670:
                color_index = 4  # black
        else:
            if color_index == 0:
                blue_points[blu_index].appendleft(center)
            elif color_index == 1:
                green_points[green_index].appendleft(center)
            elif color_index == 2:
                red_points[red_index].appendleft(center)
            elif color_index == 3:
                yellow_points[yellow_index].appendleft(center)
            elif color_index == 4:
                black_points[black_index].appendleft(center)

    # apppend the next deque when no contours are detected
    else:
        blue_points.append(deque(maxlen=512))
        blu_index += 1
        green_points.append(deque(maxlen=512))
        green_index += 1
        red_points.append(deque(maxlen=512))
        red_index += 1
        yellow_points.append(deque(maxlen=512))
        yellow_index += 1
        black_points.append(deque(maxlen=512))
        black_index += 1

    # lines of all the colors
    points = [blue_points, green_points, red_points, yellow_points, black_points]
    for b in range(len(points)):
        for g in range(len(points[b])):
            for r in range(1, len(points[b][g])):
                if points[b][g][r - 1] is None or points[b][g][r] is None:
                    continue
                cv2.line(frame, points[b][g][r - 1], points[b][g][r], colors[b], 2)
                cv2.line(paint_window, points[b][g][r - 1], points[b][g][r], colors[b], 2)

    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paint_window)

    # stop loop if "q" is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cam.release()
cv2.destroyAllWindows()
