import cv2
import cv2 as cv
import numpy as np

img =cv.imread("C:\\Users\\NAVEEN\\Downloads\\CarParkProject\\carParkImg.png")
#_,img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
gray=cv.cvtColor(img,code=cv.COLOR_BGR2GRAY)
blur_gray = cv.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

edges = cv.Canny(image=blur_gray, threshold1=50, threshold2=150, apertureSize=3)
lines=cv.HoughLinesP(edges,1,np.pi/180,50)#,maxLineGap=30
print(lines)
i=0
'''
for line in lines :
    x1,y1,x2,y2=lin
    e[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)
    #cv2.putText(img,f"{i+1}",(x1,y1),fontFace=cv2.FONT_HERSHEY_SIMPLEX,color=(0,0,255),fontScale=1)
    i=i+1
cv.imshow("PARKING LOT ",edges)
cv2.imshow("LINES ",img)'''

# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# draw contours on the original image
image_copy = img.copy()
i=1
for cnt in contours:
    approx = cv2.approxPolyDP( cnt, 0.01 * cv2.arcLength(cnt, True), True)

    M = cv2.moments(cnt)
    if M['m00'] != 0.0:
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
        print(len(approx))
    if len(approx)<=10:
        cv2.drawContours(image_copy, cnt,-1, (0, 0, 255), 5)
        cv2.putText(img,f"{i}" , (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 80)
        i=i+1

#cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                 #lineType=cv2.LINE_AA)

# see the results
cv2.imshow('None approximation', image_copy)
cv2.waitKey(0)
cv2.imwrite('contours_none_image1.jpg', image_copy)
cv2.destroyAllWindows()
cv.waitKey(0)
cv.destroyAllWindows()