import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

face_capture = cv2.CascadeClassifier(
    r'C:\Users\MARTAND MISHRA\AppData\Local\Programs\Python\Python310\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
)

age_proto = r"C:\Users\MARTAND MISHRA\.vscode\python projects\models\age_deploy.prototxt"
age_model = r"C:\Users\MARTAND MISHRA\.vscode\python projects\models\age_net.caffemodel"
gender_proto = r"C:\Users\MARTAND MISHRA\.vscode\python projects\models\gender_deploy.prototxt"
gender_model = r"C:\Users\MARTAND MISHRA\.vscode\python projects\models\gender_net.caffemodel"

age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

emotion_model_path = r"C:\Users\MARTAND MISHRA\.vscode\python projects\models\FER_static_ResNet50_AffectNet.onnx"
emotion_net = cv2.dnn.readNetFromONNX(emotion_model_path)

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
EMOTION_LIST = ['Neutral', 'Sad', 'Surprise', 'Fear', 'Disgust',  'Contempt', 'Happy']

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def detect_faces_and_predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_capture.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]

        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_LIST[age_preds[0].argmax()]

        face_resized = cv2.resize(face_img, (224, 224))
        face_blob = cv2.dnn.blobFromImage(face_resized, scalefactor=1.0, size=(224,224), mean=(0,0,0), swapRB=True, crop=False)

        
        emotion_net.setInput(face_blob)
        emotion_preds = emotion_net.forward()
        emotion_idx = np.argmax(emotion_preds)
        emotion = EMOTION_LIST[emotion_idx]

        label = f"{gender}, {age}, {emotion}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    return frame

choice = input("Run on (1) Video or (2) Image? ")

if choice == '1':
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        output_frame = detect_faces_and_predict(frame)
        cv2.imshow("Genis", output_frame)
        if cv2.waitKey(10)  == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

elif choice == '2':
    root = Tk()
    root.withdraw()
    path = askopenfilename(title="Select Image", filetypes=[("Images", "*.jpg *.jpeg *.png")])
    root.destroy()

    if path:
        img = cv2.imread(path)
        img = cv2.resize(img, (800, 650))
        if img is None:
            print("Failed to load image. Check the file format or path.")
        else:
            output_img = detect_faces_and_predict(img)
            cv2.imshow("Genius", output_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("No image selected.")

else:
    print("Invalid input.")
