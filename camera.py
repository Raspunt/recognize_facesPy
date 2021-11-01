import cv2 
import os
from PIL import Image
import numpy as np
import pickle


class camera ():


    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    cap = cv2.VideoCapture(0)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # recognizer.read("trainner.yml")


    def detect_faces(self):

        err,frame = self.cap.read()
        gray = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.5, 4)


        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            crop_gray = gray[y:y+h,x:x+w]
            crop_color = frame[y:y+h,x:x+w]
            # cv2.imwrite(f"images/crop_collor.png",crop_color)
            # cv2.imwrite(f"images/crop_gray.png",crop_gray)


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
