import os
import pickle
import cv2
import numpy as np
import glob

training_filename = "trainner.pkl"
persons = []



def load_training_data():
    faces = []
    labels = []
    

    subdirs = glob.glob("dataset/*/")
    for subdir in subdirs:
        persons.append(subdir.replace("dataset","").replace("/",""))
        
        for file in os.listdir(subdir):
            if file.endswith(".jpg"):
                path = os.path.join(subdir, file)
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                faces.append(image)
                labels.append(os.path.basename(subdir[:-1]))

    return faces, labels

        


def save_training_data(faces, labels, ):
    with open(training_filename, 'wb') as file:
        pickle.dump((faces, labels), file)


def load_training_data_from_file():
    with open(training_filename, 'rb') as file:
        faces, labels = pickle.load(file)
    return faces, labels


def recognize_face(img, recognizer):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades  +'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        label, confidence = recognizer.predict(roi_gray)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print("ID лица:", persons[label])
        print("Уверенность:", confidence)
        
        
        
        

    if len(faces) == 0:
        print("Лицо не найдено")

    cv2.imshow('giga', img)


def main():
    cap = cv2.VideoCapture(0)

    if os.path.exists(training_filename):
        faces, labels = load_training_data_from_file()
    else:
        faces, labels = load_training_data()
        # save_training_data(faces, labels)

    
    unique_labels = list(set(labels))
    label_dict = {label: i for i, label in enumerate(unique_labels)}
    labels = [label_dict[label] for label in labels]

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    
    
    while True:
        ret, frame = cap.read()

        recognize_face(frame, recognizer)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()