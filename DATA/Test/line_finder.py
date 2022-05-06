import pickle
import cv2 as cv
import numpy as np
import math

def get_frame(filename):
    cap=cv.VideoCapture(filename)
    if not cap.isOpened():
        print("Error opening video  file")
    i=0
    while cap.isOpened():
        ret, frame = cap.read()
        if i==0 :
            cv.imwrite("Parking_line.png",frame)
        cv.imshow('Frame', frame)
        i=i+1

        if cv.waitKey(25) & 0xFF == ord('q'):
            break

#get_frame("C:\\Users\\NAVEEN\\PycharmProjects\\ML\\DATA\\test.webm")


cap=cv.VideoCapture("C:\\Users\\NAVEEN\\PycharmProjects\\ML\\DATA\\test.webm")
if not cap.isOpened():
    print("Error opening video  file")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    img_Blur=cv.GaussianBlur(gray,(5,5),1)
    img_Threshold=cv.adaptiveThreshold(img_Blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,19,16)
    img_median_Threshold=cv.medianBlur(img_Threshold,5)
    k=np.ones((3,3),np.uint8)
    img_dilated=cv.dilate(img_median_Threshold,k,iterations=1)
    edges=cv.Canny(img_Blur,50, 200, None, 3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 150, None, 0, 0)
    for i in range(0, len(lines)):
        rho_l = lines[i][0][0]
        theta_l = lines[i][0][1]
        a_l = math.cos(theta_l)
        b_l = math.sin(theta_l)
        x0_l = a_l * rho_l
        y0_l = b_l * rho_l
        pt1_l = (int(x0_l + 1000 * (-b_l)), int(y0_l + 1000 * (a_l)))
        pt2_l = (int(x0_l - 1000 * (-b_l)), int(y0_l - 1000 * (a_l)))
        cv.line(frame, pt1_l, pt2_l, (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow('Frame', edges)
    cv.imshow("BLUR",img_Blur)
    cv.imshow("Threshold",img_Threshold)
    cv.imshow("Median Threshold", img_median_Threshold)
    cv.imshow("Dilate", img_dilated)
    cv.waitKey(100)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break