import cv2
import numpy as np
from keras.models import load_model

model = load_model('model_file.h5')


video = cv2.VideoCapture('Cropped Reaction video.mp4')

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

raw_test_results = []

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized/255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        raw_test_results.append(label)
        print(label)
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        # cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        # cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
    # cv2.imshow("Frame", frame)
    # k = cv2.waitKey(1)
    # if k == ord('q'):
    #     break

print(raw_test_results)
video.release()
cv2.destroyAllWindows()
