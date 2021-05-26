## 1. Read and show video stream, capture images
## 2. Detect Faces and show bounding box (haarcascade)
## 3. Flatten the largest face image and save in a numpy array
## 4. Repeat the above for multiple people to generate training data

#Import Libraries
import cv2
import numpy as np

# Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
face_data = []

# Input the name of the person
file_name = input('Enter the name of the person: ')

while True:
    ret, frame = cap.read()
    
    # Check if image is detected or not
    if ret == False:
        continue
        
    # Convert image to grayscale
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Detect and store faces in the array
    faces = face_cascade.detectMultiScale(frame,1.3,5)

    # Sort the faces acc to their sizes (f[2]*f[3])
    faces = sorted(faces, key=lambda f : f[2]*f[3])

    # Draw a bounding box around the largest face (last face is largest)
    for (x,y,w,h) in faces[-1:]:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        # Extract (Crop out the required face) : Region of Interest
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100,100))

        # Store every 10th image
        skip+=1
        if(skip%10==0):
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow('Frame', frame)
    #cv2.imshow('Frame Section', face_section)
    
    # Wait for user input - q, then loop will stop
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    
# Convert face list to numpyb array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))

# Save this data into file system
np.save('data/'+file_name+'.npy', face_data)
print('Dataset Saved succesfully at '+'data/'+file_name+'.npy')

# Release all system resources used
cap.release()
cv2.destroyAllWindows()