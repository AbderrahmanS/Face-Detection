import cv2 as cv

img = cv.imread('./img/group 1.jpg')
# cv.imshow('group' , img)

haarCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)

face_det = haarCascade.detectMultiScale(gray , 1.1 , 1)

# draw a rectangle on the detected faces 
for (x,y,w,h) in face_det :
    cv.rectangle(img , (x,y) , ( x+w ,y+h ) , (0,255,0))

cv.imshow('detected faces : ' , img)




cv.waitKey(0)