###set "KERAS_BACKEND=tensorflow"
###python python/facial-expression-recognition-from-stream.py

import numpy as np
import cv2
from keras.preprocessing import image
import h5py
from datetime import datetime
import os
import tkinter as tk #GUI package
#import facial_expression_recognition as fer #from facial-expression-recognition.py script
import matplotlib.pyplot as plt

#-----------------------------
#opencv initialization
face_cascade = cv2.CascadeClassifier('C:/Users/User/Anaconda3/envs/tensorflow/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0) #real-time streaming using webcam

#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json 
model = model_from_json(open("model/facial_expression_model_structure.json", "r").read())	#json format for keras is just the architecture strucutre of the model 
model.load_weights('model/facial_expression_model_weights.h5') #load weights
#HDF5 or h5py is the file type that contains a model/weights in keras
#-----------------------------

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

def createFolder(directory):
	try:
		if not os.path.exists(directory):
			os.mkdir(directory)
			print("%s created in %s" % (new_folder, new_folder_dir))
	except OSError:
		print ("Error creating directory for "+directory)

cwd = os.getcwd()
time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
new_folder = "frames_captured_{0}".format(time_now)
new_folder_dir = "{0}\{1}\{2}".format(cwd, 'python', new_folder)
createFolder(new_folder_dir)

#--------------------------
#function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()
#------------------------------


#counter for filename later for faces captured in each frame 
k = 0 

while(True):
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	#locations of detected faces

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
		
		detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
		detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
		detected_face_mini = cv2.resize(detected_face, (48, 48)) #resize to 48x48
		
		img_pixels = image.img_to_array(detected_face_mini)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
		
    #store probabilities of 7 expressions
		predictions = model.predict(img_pixels)

		#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
		max_index = np.argmax(predictions[0])

		#""""
		#window show probabilities
		#--------------------------
		index = 0
		for i in predictions:
			#if index < 30 and index >= 20:
				#print(i) #predicted scores
				#print(y_test[index]) #actual scores
				
				#testing_img = np.array(x_test[index], 'float32')
				#testing_img = testing_img.reshape([48, 48]);
				
				#plt.gray()
				#plt.imshow(testing_img)
			#plt.show()
			print(i)			
			#emotion_analysis(i)
			#print("----------------------------------------------")
			index = index + 1
		#""""
		emotion_captured = emotions[max_index]
		
		#write emotion text above rectangle
		cv2.putText(img, emotion_captured, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	
		#saving of frames captured into frames_captured folder
		k+=1
		FaceFileName = "./python/{0}/frame_{1}_{2}.jpg".format(new_folder, k, emotion_captured)
		cv2.imwrite(FaceFileName, detected_face)
		#process on detected face end
		#-------------------------

    

	cv2.imshow('Real Time Facial Expression Recognition',img)

	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break

#kill open cv things		
cap.release()
cv2.destroyAllWindows()