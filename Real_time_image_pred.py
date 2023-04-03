import cv2
import cvlib as cv
import numpy as np
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model

import pickle

# model=pickle.load(open(r"C:\Users\Rajkumar Maity\Documents\University Projects\CNN Models\DA_AI_stu_data\VGG16\vggv2_n_model_save",'rb'))
model=load_model(r"C:\Users\Rajkumar Maity\Documents\UNIVERSITY PROJECTS\DRONE BASED ATTENDENCE SYSTEM\Trained_Model\set0123_vgg_model_v10_e2",compile=False)
l_path=r"C:\Users\Rajkumar Maity\Documents\UNIVERSITY PROJECTS\DRONE BASED ATTENDENCE SYSTEM\Label\stu_names.npz"
stu_names=np.load(l_path)
stu_names=stu_names.f.arr_0
total_face=[]
webcam=cv2.VideoCapture(r"C:\Users\Rajkumar Maity\Documents\UNIVERSITY PROJECTS\DRONE BASED ATTENDENCE SYSTEM\DataSet_V1\Da_Al_stu_capture_vedio\Sudipta Saha.MOV")
while webcam.isOpened():
    status,frame=webcam.read()
    face, confidence=cv.detect_face(frame)
    for idx,f in enumerate(face):
        (startx,starty)=f[0],f[1]
        (endx,endy)=f[2],f[3]
        cv2.rectangle(frame,(startx,starty),(endx,endy),(0,255,0),2)
        crop_face=np.copy(frame[starty:endy,startx:endx])
        if(crop_face.shape[0])<10 or (crop_face.shape[1])<10:
            continue
        crop_face = cv2.resize(crop_face, (160, 160)).astype("float") / 255.0
        crop_face=img_to_array(crop_face)
        crop_face = np.expand_dims(crop_face, axis=0)
        #prediction one th face image
        out=model.predict(crop_face)[0]
        acc=np.max(out)
        stu_name=stu_names[np.argmax(out)]
        acc=round(acc*100,2)
        Y=starty-10 if starty-10>10 else starty+10
        if(acc>80):
            cv2.putText(frame,"Name: "+stu_name,(startx,Y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,250,250),2)
            cv2.putText(frame,"Accuracy: "+str(acc),(startx,Y-25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            if(stu_name not in total_face):
                total_face.append(stu_name)
                ex_stu=stu_name
        else:
            cv2.putText(frame,"Detecting.....",(startx,Y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(251,255,0),2)
    cv2.imshow("FACE RECOGNIZATION SYSTEM",frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()
print("\nAttendence List: ",total_face)