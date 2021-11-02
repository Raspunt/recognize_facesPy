import cv2 
import os
from PIL import Image
import numpy as np
import pickle


class camera ():


    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')


    cap = cv2.VideoCapture(0)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read("trainner.yml")

    labels = {"person_name":1}
    with open("label.pickle","rb") as f:
        og_labels = pickle.load(f)
        labels =  {v:k for  k,v in og_labels.items()}

    # recognizer.read("trainner.yml")


    def detect_faces(self):

        err,frame = self.cap.read()
        gray = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            crop_gray = gray[y:y+h,x:x+w]
            crop_color = frame[y:y+h,x:x+w]
            # cv2.imwrite(f"images/crop_collor.png",crop_color)
            cv2.imwrite(f"images/crop_gray.png",crop_gray)
            
            id_,conf = self.recognizer.predict(crop_gray)

            if  conf >= 45 :
                print(id_)
                print(self.labels[id_])
                text = self.labels[id_]
                cv2.putText(frame,text,(x+10,y),cv2.FONT_HERSHEY_COMPLEX,1, (255,255,255),2 ,cv2.LINE_AA)

        cv2.imshow("frame",frame)
        

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []    
    
    def get_imgages(self):

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_DIR,"images")


        for root ,dirs , files in os.walk(image_dir):

            for file in files:
                if file.endswith("png") or file.endswith("jpg") or file.endswith("webp"):
                    path = os.path.join(root,file)
                    label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
                    
                    if not label in self.label_ids:
                        self.label_ids[label] = self.current_id
                        self.current_id += 1

                    id_ = self.label_ids[label]
                    print(self.label_ids)

                    pil_image = Image.open(path).convert("L")
                    image_arr = np.array(pil_image,"uint8")
                    
                    faces = self.face_cascade.detectMultiScale(image_arr, 1.5, 4)


                    for (x,y,w,h) in faces:
                        roi =  image_arr[y:y+h , x:x+w]
                        self.x_train.append(roi)
                        self.y_labels.append(id_)
                    

    def write_pickle(self):

        with open("label.pickle",'wb') as f:
            pickle.dump(self.label_ids ,f)


    def CameraTrainer(self):
        self.recognizer.train(self.x_train,np.array(self.y_labels))
        self.recognizer.save("trainner.yml")
