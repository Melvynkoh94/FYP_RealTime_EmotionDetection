###set "KERAS_BACKEND=tensorflow"
###python python/facial-expression-recognition-from-stream.py

import numpy as np
import cv2
from keras.preprocessing import image
import h5py
from datetime import datetime
import os
from imutils.video import VideoStream
import imutils

#-----------------------------
#OpenCV2 DNN with Caffe as the model 
prototxt = r"C:\Users\User\Documents\GitHub\FYP_Emotion_Detection\Project2\python\caffe_models\deploy.prototxt.txt"
caffe_model = r"C:\Users\User\Documents\GitHub\FYP_Emotion_Detection\Project2\python\caffe_models\res10_300x300_ssd_iter_140000.caffemodel"
print ("[INFO] loading caffe model...")
caffe_net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)
#-----------------------------
#opencv initialization

face_cascade = cv2.CascadeClassifier('C:/Users/User/Anaconda3/envs/tensorflow/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
#cap = cv2.VideoCapture(0) #real-time streaming using webcam
cap = VideoStream(src=0).start()

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
      #print("%s created in %s" % (new_folder, new_folder_dir))
	except OSError:
		print ("Error creating directory for "+directory)

cwd = os.getcwd()
time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
new_folder = "frames_captured_{0}".format(time_now)
new_folder_dir = "{0}\{1}\{2}".format(cwd, 'python', new_folder)
createFolder(new_folder_dir)


#-----------------------------
while True:
  frame = cap.read()
  frame = imutils.resize(frame, width=400)
  (h, w) = frame.shape[:2]
  #blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
  #REFER: https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
  blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0, 177.0, 123.0))

  caffe_net.setInput(blob)
  detections = caffe_net.forward()
  
  for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
    confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence

    if confidence < 0.5:
      continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
    confidence_text = "{:.2f}%".format(confidence * 100)	#confidenceText
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.rectangle(frame, (startX, startY), (endX, endY),
      (0, 0, 255), 2)
		
		#saves the rectangle box in the frame as test.jpg
    sub_face = frame[startY:endY, startX:endX]
    sub_face = cv2.resize(sub_face, (48,48))
    sub_face = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY) #transform it to grayscale

    FaceFileName = "test2.jpg"
    cv2.imwrite(FaceFileName, sub_face)

		# Getting the Result from the label_image file, i.e., Classification Result
    #expression_text = label_image.main(FaceFileName)
    #final_text = expression_text + " " + confidence_text

    cv2.putText(frame, confidence_text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    

  # show the output frame   
  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
  if key == ord("q"):
    break
# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()



#------------------------------------
"""
#counter for filename later for faces captured in each frame 
i = 0 

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
		
		predictions = model.predict(img_pixels) #store probabilities of 7 expressions
		
		#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
		max_index = np.argmax(predictions[0])
		
		emotion_captured = emotions[max_index]
		
		#write emotion text above rectangle
		cv2.putText(img, emotion_captured, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	
		#saving of frames captured into frames_captured folder
		i+=1
		FaceFileName = "./python/{0}/frame_{1}_{2}.jpg".format(new_folder, i, emotion_captured)
		cv2.imwrite(FaceFileName, detected_face)
		#process on detected face end
		#-------------------------

	cv2.imshow('Real Time Facial Expression Recognition',img)

	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break

#kill open cv things		
cap.release()
cv2.destroyAllWindows()
"""