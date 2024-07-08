import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from keras.models import load_model



def speak(str1):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(str1)

# Load face detection model
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')




model = load_model('data/face_recognition_cnn_model.h5')

img_Background = cv2.imread("background.png")

COL_NAMES = ['Name', 'Time', 'Date']


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


while True:
            
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_dec = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces_dec:
        crop_imag = frame[y:y+h, x:x+w, :]
        resized_imag = cv2.resize(crop_imag, (50, 50)).reshape(1, 50, 50, 3)
        prediction = model.predict(resized_imag)
            
        
        out_put = np.argmax(prediction, axis=1)
        t_s = time.time()
        date = datetime.fromtimestamp(t_s).strftime("%d-%m-%Y")
        time_stamp = datetime.fromtimestamp(t_s).strftime("%H:%M-%S")
        exist_path = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        
        class_name = encoder.inverse_transform(out_put)[0]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, class_name, (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        # cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        attendance = [class_name, str(time_stamp), date]
    img_Background[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", img_Background)
    k = cv2.waitKey(1)
    
    if k == ord('o'):
        speak("Attendance Taken .."+class_name)
        time.sleep(5)
        if exist_path:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()