import cv2
import cv2 as cv
import  numpy as np
import pickle

W=40
H=94

try :
    with open("parking_space.conf",'rb') as f:
        POS_LIST=pickle.load(f)
        print(POS_LIST)
except :
    POS_LIST = []
    if len(POS_LIST)==0 :
        print("Run Parking_Space.py file first")
        exit(0)

#Read video
cap=cv.VideoCapture("car1.webm")
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
    index=0
    free_slots=0
    for i in POS_LIST:
        # w=i[0] and h=w[1]
        car_spaces = img_dilated[i[1]:i[1] + H, i[0]:i[0] + W]
        #center of rectangles
        R_X=int(((i[0]*2)+W)/2)
        R_Y=int(((i[1]*2)+H)/2)
        #cv.imshow(f"car_spaces {index}", car_spaces)
        number_of_white_pix = cv.countNonZero(car_spaces)
        if number_of_white_pix<220 or (index<12 and number_of_white_pix<420) or( index==1 and number_of_white_pix<520) :
            color=(0,255,255)
            t=3
            free_slots=free_slots+1
            text_color=(255,255,255)
        else :
            color=(0,0,255)
            t=2
            text_color=(255,255,0)
        cv2.putText(frame,f"{number_of_white_pix}",(i[0],i[1]+H-1),cv2.FONT_HERSHEY_SIMPLEX,0.3, (0, 255, 0),1)
        cv.rectangle(frame, i, (i[0] + W, i[1] + H), color, t)
        cv2.putText(frame, f"{index}", (R_X, R_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color,2)
        index=index+1

    cv2.putText(frame, f"Free Slots : {free_slots}/{24}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv.imshow('Frame', frame)
    cv.imshow("BLUR",img_Blur)
    cv.imshow("Threshold",img_Threshold)
    cv.imshow("Median Threshold", img_median_Threshold)
    cv.imshow("Dilate", img_dilated)
    cv2.waitKey(100)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break