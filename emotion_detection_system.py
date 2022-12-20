from keras.models import load_model
import numpy as np
import cv2

model = load_model('model\\6_class_emotion_detector_V2.h5')
cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('Haarcascades\\haarcascade_frontalface_default.xml')

int2emotions = {0:'angry', 1:'fear', 2:'happy', 3:'neutral', 4:'sad', 5:'surprise'}
emotions2int = {'angry':0, 'fear':1, 'happy':2, 'neutral':3, 'sad':4, 'surprise':5}

def detect_face(frame):
    faces=classifier.detectMultiScale(frame,1.3,4)
    if faces==():
        return frame
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(172,42,251),2)
        face = frame[y:y+h,x:x+w]
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face,(48,48))
        face = face.reshape(1,48,48,1)
        cv2.putText(frame,text=int2emotions[np.argmax(model.predict(face))],
                    org=(x,y-15),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(106,40,243),thickness=2)
    return frame

# Live Predictor
while True:
    ret,frame= cap.read()
    if ret==True:
        cv2.imshow('emotion_detection_system',detect_face(frame))
        if cv2.waitKey(1)== ord("q"):
            break
cap.release()
cv2.destroyAllWindows()