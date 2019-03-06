import cv2
import os
import numpy as np 

subjects = ['' , 'me' , 'other']
def detect_face(image):
	gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
	face_cascade = cv2.CascadeClassifier('/home/nilesh/Desktop/MY FILES/MY_codes/CV codes/lbpcascade_frontalface.xml')
	faces = face_cascade.detectMultiScale(gray , scaleFactor = 1.2 , minNeighbors = 5)
	if len(faces) == 0:
		return None , None
	else:
		x,y,w,h = faces[0]
		return gray[y:y+w , x:x+h] , faces[0]

def prepare_data(folderpath , label):
	dirs = os.listdir(folderpath)
	faces , labels = [] , []
	for dir_name in dirs:
		sub_path = folderpath + '/' + dir_name
		image = cv2.imread(sub_path)
		face , rect = detect_face(image)
		if face is not None:
			faces.append(face)
			labels.append(label)
	return faces , labels

person , labels = [] , []
person1 , label1 = prepare_data('/home/nilesh/Desktop/MY FILES/MY_codes/CV codes/person1' , 1) 
person2 , label2 = prepare_data('/home/nilesh/Desktop/MY FILES/MY_codes/CV codes/person2' , 2)
person += person1
person += person2
labels += label1
labels += label2

print('total faces' , len(person))
print('total labels' , len(labels)) 
person = np.array(person)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(person , np.array(labels))

def draw_rect(img , rect):
	(x,y,w,h) = rect
	cv2.rectangle(img , (x,y) , (x+w , y+h) , (0,255,0) , 2)
def draw_text(img , text , x , y):
	cv2.putText(img , text , (x,y) , cv2.FONT_HERSHEY_PLAIN , 1.5 , (0 , 255 , 0) , 2)

def test(image):
	img = image.copy()
	face , rect = detect_face(img)
	label = face_recognizer.predict(face)
	label_text = subjects[label[0]]
	draw_rect(img , rect)
	draw_text(img , label_text , rect[0] , rect[1] - 5)
	return img , label_text

test_img = cv2.imread('/home/nilesh/Desktop/MY FILES/MY_codes/CV codes/testset/newim.jpg')
pred_img , text = test(test_img)
cv2.imshow(text , pred_img)
cv2.imwrite('saved.jpg' , pred_img)
cv2.waitKey(0)


