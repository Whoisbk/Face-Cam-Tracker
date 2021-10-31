import numpy as np 
import cv2



cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')


while True:
    ret ,frame = cap.read()#returns the image/video

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#convert the frame on my webcam and convert it to gray scale
    faces = face_cascade.detectMultiScale(gray,1.3, 5)#returns the location of detected faces
   
    for (x,y,w,h) in faces:#takes the x,y ,width,height of my face and uses them to draw the rectangle around it
         
        cv2.rectangle(frame,(x,y),(x+w ,y+h),(0,255,0),2)
        roi_gray = gray[y:y+w, x:x+w] #this is the location of my face
        roi_color = frame[y:y+h, x:x+w] # this is a reference to the original frame

        eyes = eye_cascade.detectMultiScale(roi_gray,1.3,5)#returns the location of my eyes

        for (ex,ey,ew,eh) in eyes:#takes the x,y ,width,height of my eyes and uses them to draw the rectangle around them
            cv2.rectangle(roi_color,(ex,ey),(ex + ew ,ey + eh),(0,0,255),2)
        
        smile = smile_cascade.detectMultiScale(roi_gray,1.9,20)

        for (sx,sy,sw,sh) in smile:#takes the x,y ,width,height of my smile and uses them to draw the rectangle around it
            cv2.rectangle(roi_color,(sx,sy),(sx + sw ,sy + sh),(255,0,0),2)



    cv2.imshow('frame',frame)#creates the window to display the vid/image

    if cv2.waitKey(1) == ord('q'):#if the asci key is of q then we breka out of the loop
        break

 
cap.release()
cv2.destroyAllWindows() 


#scalefactor - how much we must shrink the image at each iteration(1.05 :higher accuracy but lower performing algorithm , 1.3: lower accuracy but fast performing algorithm))
#minNeighbors - how many ractangels do i need before the algorithm detects if there's really a face
#minSize - minimum size of the object
#maxSize - maximum possible object size