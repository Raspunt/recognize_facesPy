import pickle
import face_recognition as fr
import cv2
import os
import numpy as np

from face_recognition.api import face_encodings
import know_faces_reader as kfr
know_faces_dir = "know_faces/"



cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
crop_name = "crop_img.png"

kfr.face_encodings_list()

known_face_names = kfr.get_face_names()

with open("encoded_faces.pickle","rb") as  f:
    known_face_encodings = pickle.load(f)



# /////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////



def recognize_faces(gray,frame):
    faces = face_cascade.detectMultiScale(gray, 1.5, 4)
    try:
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            crop_gray = gray[y:y+h,x:x+w]
            crop_frame = frame[y:y+h,x:x+w]
            cv2.imwrite(crop_name,crop_gray)
        
            if  os.path.exists(crop_name) and os.path.exists("encoded_faces.pickle"):
                
                for kf in known_face_encodings:
                    frame_encode = fr.face_encodings(cv2.imread(crop_name))[0]
                    
                    res = fr.compare_faces([kf["encode_img"]],frame_encode)
            
                    if res[0] :
                        print(f"face found !!! it is {kf['name']}")
                        cv2.putText(frame,kf['name'],(x+10,y),cv2.FONT_HERSHEY_COMPLEX,1, (255,255,255),2 ,cv2.LINE_AA)
        
        return frame


    except IndexError as e:
        print(e)






while True:
    err,frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    # frame = recognize_faces(gray,frame)

    faces_locations = fr.face_locations(rgb_small_frame)

    face_encode = fr.face_encodings(rgb_small_frame,faces_locations)



    face_names = []
    for face_encoding in face_encode:
        matches = fr.compare_faces(known_face_encodings, face_encoding)

        face_distances = fr.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            face_names.append(name)





    for (top, right, bottom, left), name in zip(faces_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break






