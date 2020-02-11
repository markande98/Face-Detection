import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []

# Dataset_path = "./Face_recognition/"

filename = input("Enter the name of the person : ")

while True:

	ret, frame = cap.read()

	if ret == False:
		continue

	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray_frame, 1.3, 4)

	if(len(faces) == 0):
		continue

	faces = sorted(faces, key=lambda f:f[2]*f[3])

	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(gray_frame, (x,y), (x+w, y+h), (167,100,50), 3)

		# Extract the face from the frame : Region of interest
		offset = 10

		face_section = gray_frame[y-offset: y + h + offset, x - offset: x + w + offset]

		face_section = cv2.resize(face_section, (100, 100))

		face_data.append(face_section)
		print(len(face_section))

	cv2.imshow("gray_frame", gray_frame)

	key_pressed = cv2.waitKey(1) & 0xFF

	if key_pressed == ord('q'):
		break;

# Convert Face data list into numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape(face_data.shape[0], -1)
print(face_data.shape)

# Save the face data into file system
np.save(filename+'.npy', face_data)
print("saved succesfully")

cap.release()
cv2.destroyAllWindows()