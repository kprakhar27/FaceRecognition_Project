## 1. load the training data (x- stored in numpy arrays, y- values need to be assigned for each person)
## 2. Read a video stream using opencv
## 3. extract faces out of it
## 4. use knn to find prediction of face (int)
## 5. map th predicted id to the name of the user
## 6. display teh predictions on the screen - bounding box and name

# Import Libraries
import cv2
import numpy as np
import os

################# KNN - CODE #################
def distance(v1,v2):
    return np.sqrt(sum((v1-v2)**2))

def knn(train, test, k=5):
    dist = []
        
    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i,:-1]
        iy = train[i,-1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    
    # Sort based on distance and get top k
    dk = sorted(dist,key = lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:,-1]

    # Get frequencies of each label
    output = np.unique(labels,return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]
##############################################

# Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

face_data = []
labels = []

class_id = 0 # labels for the given file
names = {} # Mapping between id:name

# Data Preparation
for fx in os.listdir('data/'):
    if fx.endswith('.npy'):
        # Create a mapping between id:name
        names[class_id] = fx[:-4]
        print('loaded '+fx)
        data_item = np.load('data/'+fx)
        face_data.append(data_item)

        # Create labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

# Concetenate the face data to create a single array for knn
face_dataset = np.concatenate(face_data,axis=0)
# Concatenate the labels to create a single array of labels
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

# Concetenate the data to create single training dataset
trainset = np.concatenate((face_dataset,face_labels),axis=1)

# Testing
while True:
    ret, frame = cap.read()
    
    # Check if image is detected or not
    if ret == False:
        continue
        
    # Convert image to grayscale
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Detect and store faces in the array
    faces = face_cascade.detectMultiScale(frame,1.3,5)

    # Draw a bounding box around the largest face (last face is largest)
    for (x,y,w,h) in faces[-1:]:
               
        # Get the face : Region of Interest
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100,100))

        # Predicted label (out)
        out = knn(trainset,face_section.flatten())
        # Predicted name
        pred_name = names[int(out)]

        # Display name and draw rectangle around it
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # Display the video output
    cv2.imshow('Frame', frame)
    
    # Wait for user input - q, then loop will stop
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Release all system resources used
cap.release()
cv2.destroyAllWindows()