import numpy as np
import facemodel
from fr_utils import *
import cv2
from keras import backend as K
K.set_image_data_format('channels_first')
from facemodel import *
import os


# _MODEL_OUTPUT_SHAPE = (1, 128)
cap = cv2.VideoCapture(0)
FRmodel = faceRecoModel(input_shape=(3,96,96))
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

# update database
build_database(FRmodel)

# load known people
known_people = load_known_people()

PADDING = 0
# define a face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
err_count = 0
pass_count = 0
while cap.isOpened():
	ret, frame = cap.read()

	# convert frames to gray scale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_detector.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=3
		)
	for (x,y,w,h) in faces:
		PADDING = int(w*0.1)
		x1 = x + PADDING
		y1 = y
		x2 = x + w - PADDING
		y2 = y + h
		cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 2)
		face = frame[y1:y2, x1:x2]
		person = identify(face, FRmodel, known_people)
		if person == -1: 
			# if recorded face fails test more than 10 consecutive times, reject this person
			pass_count = 0
			err_count += 1
			if err_count > 10:
				err_count = 0
				print "Sorry, we can't recognize you"
			else:
				continue
		else:
			# if recorded face passes test more than 5 consecutive times, confirm this person
			err_count = 0
			pass_count += 1
			if pass_count > 5:
				pass_count = 0
				print "Welcome home {}".format(person)
			else:
				continue


	cv2.imshow('frame',frame)
	if cv2.waitKey(1) and 0xFF==ord('q'):
		break


cap.release()
cv2.destroyAllWindows()
