from __future__ import print_function
import cv2
import numpy as np  
import os


cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

skip = 0
face_data = []
path = os.path.dirname(os.path.abspath('__file__')) + '/data'
nameoffile = input('enter the name : ')
while True:
	ret , frame = cap.read()
	if ret == False:
		continue


	faces = face_cascade.detectMultiScale(frame , 1.3 , 5)

	faces = sorted(faces , key= lambda f:f[2]*f[3])
	x,y,w,h = faces[-1]

	cv2.rectangle(frame , (x,y) , (x+w,y+h) , (0,255,0) , 2)

	offset = 10
	face_section = frame[y:y+h+offset , x:x+w+offset]
	face_section = cv2.resize(face_section , (100,100))

	skip+=1
	if(skip%10 == 0):
		face_data.append(face_section)
		print(len(face_data))

	cv2.imshow('frame' , frame)
	cv2.imshow('face_section' , face_section)

	key_pressed = cv2.waitKey(1)&0xFF
	if key_pressed == ord('q'):
		break

#convert facelist to numpy
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0] , -1))
np.save(path+'/'+nameoffile+'.npy' , face_data)
print('data saved successfully')

cap.release()
cv2.destroyAllWindows()