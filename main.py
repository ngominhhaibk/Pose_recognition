import cv2
import time
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from pygame import mixer
mixer.init()
sound = mixer.Sound('alarm2.wav')

from tools.ActionsEstLoader import TSSTG
from yolov7.pose import yolov7_pose
model = yolov7_pose('E:\AI_ML_DL\Human-Falling-Detect-YOLOv7\yolov7\weight\yolov7-w6-pose.pt')

source = 0


def YL2XYS(kpts,steps=3):
    result = np.zeros([17,3])
    num_kpts = len(kpts) // steps
    for kid in range(num_kpts):
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                result[kid,0] = x_coord
                result[kid,1] = y_coord
                result[kid,2] = conf
    return result

if __name__ == '__main__':
    device = 'cuda'
    # Actions Estimate.
    action_model = TSSTG()
    
    cap = cv2.VideoCapture(source)
    miss = 0
    keypoints_list = []
    codec = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter('test.avi', codec, 30, (640, 640))
    count_falldown = 0
    thr = 15
    while True:
        ret,frame = cap.read()
        if ret:
            t_start = time.time()
            # Detect humans bbox in the frame with detector model.
            image, detected = model.predict(frame)
            if detected.shape[0]>0:
                miss = 0
                keypoints = YL2XYS(detected)
                keypoints_list.append(keypoints)
            else:
                miss +=1
            
            if miss >= 5:
                keypoints_list = []
            action_name = "pending..."
            color = (0, 255, 0)
            if len(keypoints_list) == 30:
                pts = np.array(keypoints_list.copy(), dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                count = 0
                keypoints_list = keypoints_list[1:]
            print(action_name)
            if action_name =="Fall Down":
                try:
                    sound.play()
                except:  # isplaying = False
                    pass
                count_falldown+=1
                if count_falldown < thr:
                    action_name = buff_name
                else:
                    color = (0,0,255)
            else:
                count_falldown = 0
                buff_name = action_name
            
            
            fps = 1/(time.time() - t_start)
            cv2.putText(image, str(fps),(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color , 1)
            cv2.putText(image, str(action_name),(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow('frame', image)
            writer.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
