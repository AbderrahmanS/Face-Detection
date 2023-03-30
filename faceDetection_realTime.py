import cv2 as cv

capture = cv.VideoCapture(0)
haarCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

while True :
    #read the frame
    _,img = capture.read()

    # img ==> to grayscale image
    gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)

    face_det = haarCascade.detectMultiScale(gray , 1.1 , 4)

    for (x,y,w,h) in face_det :
        cv.rectangle(img , (x,y) , (x+w , y+h) , (0,255,0))

    #display
    cv.imshow('img' , img)

    # Stop if escape key is pressed
    k = cv.waitKey(30) & 0xff
    if k==27:
        break

# Release the VideoCapture object
capture.release()    