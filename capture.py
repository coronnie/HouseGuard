import cv2
import os
import sys
import time

if len(sys.argv)!=2:
	exit("capture.py takes exactly 1 input, exiting...")
name = sys.argv[1]
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

PADDINGX = 0
while cap.isOpened():
	ret, frame = cap.read()
	cv2.imshow('live stream', frame)
	# testing face classifier
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_detector.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=3
		)
	largest = 0
	X=0
	Y=0
	W=0
	H=0
	for (x,y,w,h) in faces:
		if w*h > largest:
			largest = w*h
			(X, Y, W, H) = (x, y, w, h)
			PADDINGX = int(W*0.1)
	if W<50 or H<50:
		# if detected area is too small, reject because it may not be a face or it's too small, wait for person to come closer
		print "If you are standing too far, please come closer"
		time.sleep(1)
		continue
	else:
		x1 = X + PADDINGX
		y1 = Y
		x2 = X + W - PADDINGX
		y2 = Y + H

		cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 2)

		# make sure the calculation of PADDINGX is the same as in the camera main program, so as to ensure that the captured images and the faces recorded in real time are cropped in the same manner
		face = frame[y1:y2, x1:x2]
		file = os.path.join('./faces', name+'.png')
		cv2.imwrite(file, face)
		break

cap.release()
cv2.destroyAllWindows()