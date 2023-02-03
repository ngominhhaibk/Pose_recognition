import cv2
import time
import numpy as np
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from tools.ActionsEstLoader import TSSTG
from yolov7.pose import yolov7_pose
from Track.Tracker import Detection, Tracker

model = yolov7_pose('./yolov7/weight/yolov7-w6-pose.pt')
source = 0

class_names = {'Standing':(255,255,255),
               'Stand up':(255,0,0),
               'Sitting':(255,0,255),
               'Sit down':(0,255,255),
               'Lying Down':(255,255,0),
               'Walking':(0,255,0),
               'Fall Down':(0,0,255),
               'pending..':(0,0,0)
               }



if __name__ == '__main__':

    # Actions Estimate.
    action_model = TSSTG('./weights/tsstg-model.pth')
    cap = cv2.VideoCapture(source)
    
    #write video
    codec = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter('333.avi', codec, 30, (1600, 1000))
  
    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    while True:
        ret,frame = cap.read()
        if ret:
            t_start = time.time()
            # Detect humans bbox in the frame with detector model.
            nimg,detected,keypoints = model.predict2(frame)
            
            tracker.predict()

            for track in tracker.tracks:
                det = torch.tensor([track.to_tlbr().tolist() + [0.5]], dtype=torch.float32).numpy()
                detected = np.concatenate((detected, det), axis=0) if detected is not None else det

            detections = []
            if detected is not None:
                for idx,kps in enumerate(keypoints):
                    detections.append(Detection(detected[idx,0:4],keypoints[idx],keypoints[idx][:,2].mean()))
            tracker.update(detections)

            for i, track in enumerate(tracker.tracks):
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)

                action = 'pending..'
                clr = (0, 0, 255)
                # Use 30 frames time-steps to prediction.
                if len(track.keypoints_list) == 30:
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    out = action_model.predict(pts, frame.shape[:2])
                    action_name = action_model.class_names[out[0].argmax()]
                    action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                    cv2.putText(nimg,action,(bbox[0],bbox[1]),cv2.FONT_HERSHEY_COMPLEX,0.5,class_names[action_name],1)
                else:
                    cv2.putText(nimg,action,(bbox[0],bbox[1]),cv2.FONT_HERSHEY_COMPLEX,0.5,class_names[action],1)
            nimg = cv2.resize(nimg,[1600,1000])

            cv2.imshow('pose',nimg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            writer.write(nimg)
        else:
            break
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
