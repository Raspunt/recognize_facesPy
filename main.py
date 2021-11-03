import pickle
import face_recognition as fr
import cv2
import os
import know_faces_reader as kfr
know_faces_dir = "know_faces/"


cap = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
crop_name = "crop_img.png"

kfr.write_encoded_Files()

with open("encoded_faces.pickle","rb") as  f:
    data = pickle.load(f)



while True:
    err,frame = cap.read()

    gray = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.5, 4)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        crop_gray = gray[y:y+h,x:x+w]
        crop_frame = frame[y:y+h,x:x+w]
        cv2.imwrite(crop_name,crop_gray)
    
    try:
        if  os.path.exists(crop_name) and os.path.exists("encoded_faces.pickle"):
            
            for kf in data:
                frame_encode = fr.face_encodings(cv2.imread(crop_name))[0]
                
                res = fr.compare_faces([kf["encode_img"]],frame_encode)
        
                if res[0] :
                    print(f"face found !!! it is {kf['name']}")


    except IndexError as e:
        print(e)




    cv2.imshow("fr",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# if os.path.exists(crop_gray_name):
#         frame_encode = fr.face_encodings(cv2.imread(crop_gray_name))[0]
#         res = fr.compare_faces([unknow_encoding],frame_encode)

#         if res[0]:
#             print("YES")
#         else:
#             print("NO")
