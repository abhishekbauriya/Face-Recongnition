# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, cpature images
# 2. Detect Faces and show bounding box
# 3. Flatten the largest face image and save in a numpy array
# 4. Reapeat the above for multiple people to generate training data


import cv2
import numpy as np

#Initialise Camera
cap = cv2.VideoCapture(0)

#Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
datset_path = './Project 01 - Face Recognition/'
file_name = input("Enter the name of the person :")

while True:
	ret,frame = cap.read()

	if ret==False:
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(frame,1.3,5)
	faces = sorted(faces,key=Lambda f:f[2]*f[3])

	#Pick the last face (beacuz it is the largest face according to area (f:f[2]*f[3]))
	for face in faces[-1]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		#Extract (Crop out the requied face): Region of interest
		offset = 10
		face_section = frame[y-offset:y+h+offest,x-offest:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		skip +1= 1
		if skip%10==0:
			face_data.append(face_section)
			print(Len(face_data))

	cv2.imshow("Frame",frame)	
	cv2.imshow("Face Section",face_section)	

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

#COnvert our face list array into a numpy array
face_data = np.array(face_data)		
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#Save this data intop file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Sucessfully save at "+dataset_path+file_name+'.npy')

cap.release()
cap.destroyAllWindows()