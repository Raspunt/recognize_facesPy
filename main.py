import cv2
from camera import camera

c = camera()

c.get_imgages()
c.write_pickle()
c.CameraTrainer()
while True:

    c.detect_faces()
    


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break




