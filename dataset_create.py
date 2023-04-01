import os
import cv2
import glob

import numpy as np





def create_dataset():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    saved_faces = []

    dirName = input("What would you like to name the directory?: ")

    if not os.path.isdir(f'dataset/{dirName}'):
        os.mkdir(f'dataset/{dirName}')

    while True: 
        ret, frame = cap.read()
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            crop = frame[y+10:y+h+10, x+10:x+w+10]
            
            # Resize the crop to match the saved face images' size
            resized_crop = cv2.resize(crop, (100, 100))
            
            # Convert the image to grayscale
            resized_crop_gray = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2GRAY)
            
            # Check if the detected face matches any of the saved face images
            match_found = False
            for saved_face in saved_faces:
                diff = cv2.absdiff(resized_crop_gray, saved_face)
                mean_diff = np.mean(diff)
                if mean_diff < 10:  # Threshold for matching faces
                    match_found = True
                    break
            
            # If no match is found, save the face image
            if not match_found:
                cv2.imwrite(f"dataset/{dirName}/{len(saved_faces)}.jpg", resized_crop)
                saved_faces.append(resized_crop_gray)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2) 
            
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def create_face_grid():
    
    persons = []
    
    subdirs = glob.glob("dataset/*/")
    for subdir in subdirs:
        images = []
        for file in os.listdir(subdir):
            if file.endswith(".jpg"):
                path = os.path.join(subdir, file)
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                images.append(image)
                
            
        name = subdir.replace("dataset","").replace("/","")
        persons.append((name,images))
    
    
    for person in persons:
        imgs = person[1]
        img = cv2.hconcat(imgs)
        
        cv2.imwrite(f"{person[0]}.jpg",img)
            
    
    
        
 
            
    
# create_face_grid()
create_dataset()