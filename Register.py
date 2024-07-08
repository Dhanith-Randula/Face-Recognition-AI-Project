import cv2
import pickle
import numpy as np
import os
import tkinter as tk
from tkinter import simpledialog

video=cv2.VideoCapture(0)
face_detect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data=[]

i=0

root = tk.Tk()
root.geometry("400x100") 
root.configure(bg='blue') 
root.withdraw()
img_Background = cv2.imread("background.png")

Enter_name = simpledialog.askstring("Input", "Enter Your Name:", parent=root)

# name=input("Enter Your Name: ")

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50))
        if len(faces_data)<=100 and i%10==0:
            faces_data.append(resized_img)
        i=i+1
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        img_Background[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame",img_Background)

    k=cv2.waitKey(1)

    if k==ord('q') or len(faces_data)==100:
        break
video.release()
cv2.destroyAllWindows()

faces_data=np.asarray(faces_data)
faces_data=faces_data.reshape(100, -1)


if 'names_data.pkl' not in os.listdir('data/'):
    names=[Enter_name]*100
    with open('data/names_data.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names_data.pkl', 'rb') as f:
        names=pickle.load(f)
    names=names+[Enter_name]*100
    with open('data/names_data.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces=pickle.load(f)
    faces=np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)