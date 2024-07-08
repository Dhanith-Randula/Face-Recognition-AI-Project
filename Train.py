
import pickle
from win32com.client import Dispatch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


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

# Build CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(LABELS.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(FACES, LABELS, epochs=10, batch_size=32, validation_split=0.1)

# Save the trained model
model.save('data/face_recognition_cnn_model.h5')