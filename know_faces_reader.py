import os
import json
import face_recognition as fr
import pickle

from face_recognition.api import face_encodings


know_faces_dir = "know_faces"
know_faces = []


def write_encoded_Files():


    person_arr = []
    for filename in os.listdir(know_faces_dir):

        unknow_image = fr.load_image_file(f"{know_faces_dir}/{filename}")
        unknow_encoding = fr.face_encodings(unknow_image)[0]

        person ={
            "name":filename.replace(".jpg",""),
            "encode_img":unknow_encoding,
        }

        person_arr.append(person)


        with open("encoded_faces.pickle","wb") as f:
            pickle.dump(person_arr,f)




def face_encodings_list():

    person_arr = []
    for filename in os.listdir(know_faces_dir):

        unknow_image = fr.load_image_file(f"{know_faces_dir}/{filename}")
        unknow_encoding = fr.face_encodings(unknow_image)[0]


        person_arr.append(unknow_encoding)


        with open("encoded_faces.pickle","wb") as f:
            pickle.dump(person_arr,f)

def get_face_names():

    name_arr = []
    for filename in os.listdir(know_faces_dir):

        name_arr.append(filename.replace(".jpg",""))

    return name_arr
