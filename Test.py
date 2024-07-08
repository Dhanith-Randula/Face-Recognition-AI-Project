import cv2
import pickle
import numpy as np
import os
import time
from datetime import datetime
from win32com.client import Dispatch
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score



# Load face detection model
video = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
model_load = load_model('data/face_recognition_cnn_model.h5')
img_Background = cv2.imread("background.png")
# Load data
with open('data/names_data.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)
print('Shape of Faces matrix --> ', FACES.shape)
# Preprocess data for CNN
FACES = FACES.reshape(FACES.shape[0], 50, 50, 3)
encoder = LabelEncoder()
LABELS = encoder.fit_transform(LABELS)
LABELS = to_categorical(LABELS)
name_counts = {}
count = 0
run = True;
while run:
    
    if count >= 100:
            run = False
            break
    
        
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).reshape(1, 50, 50, 3)
        prediction = model_load.predict(resized_img)
             
        
        out_put = np.argmax(prediction, axis=1)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        time_stamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist_path = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        
        class_name = encoder.inverse_transform(out_put)[0]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, class_name, (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        # cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        attendance = [class_name, str(time_stamp), date]
        
        
        
        if attendance[0] in name_counts:
            name_counts[attendance[0]] += 1
            count +=1
        else:
            name_counts[attendance[0]] = 1
            count +=1
            
        # print(name_counts)
        # print(name_counts[attendance[0]] +name_counts[attendance[0]])
        
        

    img_Background[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", img_Background)
    k = cv2.waitKey(1)
    
    print(count)
    
    
  
video.release()
cv2.destroyAllWindows()

for attendance in name_counts:   
    print("Name : " + attendance)
    print("------------------")
    print("Correct Precentage : "+ str(name_counts[attendance]) + "%")
    
    y_test = [1]*100  
    y_pred = [1]*name_counts[attendance] + [0]*(100-name_counts[attendance])

# Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

# Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)

# Calculate Precision
    precision = precision_score(y_test, y_pred)

# Calculate Recall
    recall = recall_score(y_test, y_pred)

# Calculate F1 score
    f1 = f1_score(y_test, y_pred)

    print(f'Confusion Matrix: \n{cm}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print("======================================================")