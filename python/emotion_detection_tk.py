###set "KERAS_BACKEND=tensorflow"
import numpy as np
import cv2
from keras.preprocessing import image as kerasimage
from keras.models import model_from_json 
import h5py
from datetime import datetime
import os
import tkinter as tk #GUI package
import tkinter.filedialog
from tkinter import *
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageTk

emotions_probList = [None]*7
#-----------------------------
#initializations
face_cascade = cv2.CascadeClassifier('./dependencies/haarcascade_frontalface_default.xml')
ip_cam_url = "http://192.168.43.1:8080/shot.jpg"	#for instructions on IP CAM--> https://www.youtube.com/watch?v=2xcUzXataIk&t=561s
model = model_from_json(open("./dependencies/emotion_detection_model.json", "r").read())	#json format for keras is just the architecture strucutre of the model 
model.load_weights('./dependencies/emotion_detection_weights.h5') #load weights
#HDF5 or h5py is the file type that contains a model/weights in keras
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
#-----------------------------


#-----------------------------
#For static Image for Emotion Detection
def select_image():
	num_faces = 0
	#grab a reference to an image panel
	global panelA
	global angry_prob, disgust_prob, fear_prob, happy_prob, sad_prob, surprise_prob, neutral_prob
	panelA = None
	#open a file chooser dialog for user to select an input image
	path = filedialog.askopenfilename()

	#ensure a file path was selected
	if(len(path) > 0):
		image = cv2.imread(path)
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
		faces_array = np.array(faces)	#cast the tuple type to array type to get the shape later on 
		num_faces = faces_array.shape[0]
		print('Number of faces: ',num_faces)
		#locations of detected faces
		for (x,y,w,h) in faces:
			cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0),2) #draw rectangle to main image
			detected_face = image[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
			detected_face_mini = cv2.resize(detected_face, (48, 48)) #resize to 48x48
			img_pixels = kerasimage.img_to_array(detected_face_mini)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

			#store probabilities of 7 expressions
			predictions = model.predict(img_pixels)

			#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
			max_index = np.argmax(predictions[0])	#returns the index of the max value
			max_prob = np.max(predictions[0])

			#probabilities of each expression
			angry_prob = predictions[0][0]
			disgust_prob = predictions[0][1]
			fear_prob = predictions[0][2]
			happy_prob = predictions[0][3]
			sad_prob = predictions[0][4]
			surprise_prob = predictions[0][5]
			neutral_prob = predictions[0][6]
			i=0
			for i in range(7):
				emotions_probList[i] = predictions[0][i]			
			set_label(emotions_probList, num_faces)

			#this is the emotion distinguished by the model
			#emotion_captured = emotions[max_index] + ' '+ str(max_prob)
			emotion_captured = emotions[max_index]
			emotion_prob = str(max_prob)

			#write emotion text above rectangle
			#WHITE TEXT (255,255,255)
			#final_image = cv2.putText(image, emotion_captured, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
			final_image = cv2.putText(image, emotion_captured, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,255,0), 2)	#green text at the top 
			final_image = cv2.putText(image, emotion_prob, (int(x), int(y+h+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255,0), 2)	#green text at the bottom
			#final_image = cv2.putText(image, emotion_captured, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
			cv2.imshow('Static Image Emotion Detection', final_image)

			#In order to display our images in the Tkinter GUI, we first need to change the formatting. 
			#To start, OpenCV represents images in BGR order; however PIL/Pillow represents images in RGB order, so we need to reverse the ordering of the channels
			#convert image to PIL format
			final_image = Image.fromarray(final_image)
			#then convert to ImageTk format
			final_image = ImageTk.PhotoImage(final_image)

		"""if panelA is None:
			panelA = Label(image=final_image)
			panelA.image = final_image
			panelA.grid(column=0, row=6)
			#panelA.pack(side="top", padx=10, pady=10) #ERROR: do not use PACK and GRID in the same window!

		# otherwise, update the image panels
		else:
			# update the pannels
			panelA.configure(image=final_image)
			panelA.image = final_image"""
#-----------------------------


#-----------------------------
#For VideoStream/Live Stream or Emotion Detection
def select_vs():
	num_faces = 0
	cap = cv2.VideoCapture(0) #real-time streaming using webcam
	#cap = cv2.VideoCapture(ip_cam_url)
	cwd = os.getcwd()
	time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
	new_folder = "frames_captured_{0}".format(time_now)
	new_folder_dir = "{0}\{1}\{2}".format(cwd, 'python', new_folder)
	createFolder(new_folder, new_folder_dir)

	#counter for filename later for faces captured in each frame 
	j = 0 
	while(True):
		#cap = cv2.VideoCapture(ip_cam_url)
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		faces_array = np.array(faces)
		num_faces = faces_array.shape[0]

		#locations of detected faces
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image	
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
			detected_face_mini = cv2.resize(detected_face, (48, 48)) #resize to 48x48		
			img_pixels = kerasimage.img_to_array(detected_face_mini)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
			
			#predictions is a 2-d array! predictions[0] contains probabilities of 7 emotions respectively
			predictions = model.predict(img_pixels) #store probabilities of 7 expressions
			
			#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
			max_index = np.argmax(predictions[0])	#returns the index of the max value
			max_prob = np.max(predictions[0])

			#probabilities of each expression
			angry_prob = predictions[0][0]
			disgust_prob = predictions[0][1]
			fear_prob = predictions[0][2]
			happy_prob = predictions[0][3]
			sad_prob = predictions[0][4]
			surprise_prob = predictions[0][5]
			neutral_prob = predictions[0][6]
			i=0
			for i in range(7):
				emotions_probList[i] = predictions[0][i]
			set_label(emotions_probList, num_faces)
			#set_label(emotions_probList)
			print(emotions_probList)			
			#emotion_captured = emotions[max_index] + ' ' + str(max_prob)
			emotion_captured = emotions[max_index]
			emotion_prob = str(max_prob)
			
			#write emotion text above rectangle
			#cv2.putText(img, emotion_captured, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
			cv2.putText(img, emotion_captured, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,255,0), 2)	#green text at the top 
			cv2.putText(img, emotion_prob, (int(x), int(y+h+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255,0), 2)	#green text at the bottom
		
			#saving of frames captured into frames_captured folder
			j+=1
			FaceFileName = "./python/{0}/frame_{1}_{2}_{3}.jpg".format(new_folder, j, emotion_captured, emotion_prob)
			cv2.imwrite(FaceFileName, detected_face)
			#process on detected face end
			#-------------------------

		cv2.imshow('Real-Time Emotion Detection',img)

		if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
			break

	#kill open cv things		
	cap.release()
	cv2.destroyAllWindows()
#-----------------------------


#-----------------------------
#Clear values of the labels
def clearValues():
	labelAngry['text'] = "Angry: \t"+ "0.00"
	labelDisgust['text'] = "Disgust: \t"+ "0.00"
	labelFear['text'] = "Fear: \t"+ "0.00"
	labelHappy['text'] = "Happy: \t"+ "0.00"
	labelSad['text'] = "Sad: \t"+ "0.00"
	labelSurprise['text'] = "Surprise: "+ "0.00"
	labelNeutral['text'] = "Neutral: \t"+ "0.00"
	labelNumFaces['text'] = "No. of Faces: \t"+ "0"
#-----------------------------


#-----------------------------
#Create a folder function
def createFolder(new_folder, new_folder_dir):
	try:
		if not os.path.exists(new_folder_dir):
			os.mkdir(new_folder_dir)
			print("%s created in %s" % (new_folder, new_folder_dir))
	except OSError:
		print ("Error creating directory for "+new_folder_dir)
#-----------------------------


#-----------------------------
#dynamically change label texts
def set_label(emotions_probList, num_faces):
	labelAngry['text'] = "Angry: \t"+ str(emotions_probList[0])
	labelDisgust['text'] = "Disgust: \t"+ str(emotions_probList[1])
	labelFear['text'] = "Fear: \t"+ str(emotions_probList[2])
	labelHappy['text'] = "Happy: \t"+str(emotions_probList[3])
	labelSad['text'] = "Sad: \t"+ str(emotions_probList[4])
	labelSurprise['text'] = "Surprise:"+ str(emotions_probList[5])
	labelNeutral['text'] = "Neutral: \t"+ str(emotions_probList[6])
	labelNumFaces['text'] = "No. of Faces: "+ str(num_faces)
	window.after(1)
#-----------------------------


#----------------------------- GUIDE:https://www.youtube.com/watch?v=JrWHyqonGj8
#Tkinter GUI main window
window = tk.Tk()
window.geometry("500x300")	#width x height
window.title("Melvyn FYP Emotion Detection")
#window.geometry("800x800")

#LABEL
title = tk.Label(text="Choose one of the following:")
title.grid(column=0, row=0)

#BUTTON 1
button1 = tk.Button(window, text="Upload an Image/Video File", command=select_image)
button1.grid(column=0, row=1)

#BUTTON 2
button2 = tk.Button(window, text="Run Webcam", command=select_vs)
button2.grid(column=1, row=1)

#BUTTON 3
button3 = tk.Button(window, text="Clear Values", command=clearValues)
button3.grid(column=0, row=2)


#0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
#EXPRESSIONS PROBABILITIES (1st column)
labelAngry = tk.Label(window, text="Angry: ",font="Helvetica 10 bold")
labelAngry.grid(column=0, row=3, sticky=W)

labelDisgust = tk.Label(window, text="Disgust: ",font="Helvetica 10 bold")
labelDisgust.grid(column=0, row=4, sticky=W)

labelFear = tk.Label(window, text="Fear: ",font="Helvetica 10 bold")
labelFear.grid(column=0, row=5, sticky=W)

labelHappy = tk.Label(window, text="Happy: ",font="Helvetica 10 bold")
labelHappy.grid(column=0, row=6, sticky=W)

#EXPRESSIONS PROBABILITIES (2nd column)
labelSad = tk.Label(window, text="Sad: ",font="Helvetica 10 bold")
labelSad.grid(column=1, row=3, sticky=W)

labelSurprise = tk.Label(window, text="Surprise: ",font="Helvetica 10 bold")
labelSurprise.grid(column=1, row=4, sticky=W)

labelNeutral = tk.Label(window, text="Neutral: ",font="Helvetica 10 bold")
labelNeutral.grid(column=1, row=5, sticky=W)

#NUMBER OF FACES DETECTED
labelNumFaces = tk.Label(window, text="No. of Faces: ", font="Helvetica 10 bold")
labelNumFaces.grid(column=1, row=6, sticky=W)

#set_label(emotions_probList)
window.mainloop() #mainloop() must always be at the bottom