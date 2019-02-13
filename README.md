# HouseGuard
This is a face recognition system for personal house use.

Requirements:
  - Python.2.7.x
  - Keras
  - TensorFlow
  - opencv

Usage:

Step 1. capture.py - is the program to add known people to the database. To use: python capture.py "person_name".

Step 2. camera.py - is the main program, it will open the detected camera and recognize the faces in the camera.

Voila, that's it.


Notes:
When adding people to database, be sure to use the camera that's used to detect people's faces, to ensure that the faces in the database and the faces being detected in the camera have the same image source.

Reference:
FaceNet paper: https://arxiv.org/abs/1503.03832
