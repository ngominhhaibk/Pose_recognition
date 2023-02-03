import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import time
import torch.backends.cudnn as cudnn
import sys
sys.path.append('./yolov7')

from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from models.experimental import attempt_load


class yolov7_pose:
    def __init__(self,weights):
        self.weights = weights
        self.conf_thres = 0.01
        self.iou_thres = 0.45
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
    
    def load_model(self):
        model = torch.load(self.weights)['model']
        model = model.half().to(self.device)
        _ = model.eval()
        return model
    
    
    def preprocess_image(self,image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = letterbox(img.copy(), stride=64, auto=False)[0]
        img = transforms.ToTensor()(img)
        img = torch.tensor(np.array([img.numpy()]))
        img = img.to(self.device)
        img = img.half()
        return img
  

    def predict(self,image):
        img = self.preprocess_image(image)
        with torch.no_grad():
            output, _ = self.model(img)

        output = non_max_suppression_kpt(output, self.conf_thres, self.iou_thres, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)
        output = output_to_keypoint(output)
        nimg = img[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

        
        if output.shape[0] >0:
            idx = np.argmax(output,axis=0)[6] # bb co score lon nhat
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
            xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
            xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
            cv2.rectangle(nimg,(int(xmin), int(ymin)),(int(xmax), int(ymax)),color=(255, 0, 0),thickness=1,lineType=cv2.LINE_AA)
            return nimg,output[idx,7:]
        else:
            return nimg,np.array([])
    def predict1(self,image):
        img = self.preprocess_image(image)
        with torch.no_grad():
            output, _ = self.model(img)
        
        output = non_max_suppression_kpt(output, self.conf_thres, self.iou_thres, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)
        output = output_to_keypoint(output)
        nimg = img[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
            xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
            xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
            cv2.rectangle(nimg,(int(xmin), int(ymin)),(int(xmax), int(ymax)),color=(255, 0, 0),thickness=1,lineType=cv2.LINE_AA)

        if output.shape[0] >0:
            output = output[:,1:]
        else:
            output = np.array([])
        return nimg,output
    
    def YL2XYS(self,kpts,steps=3):
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
    
    def predict2(self,image):
        img = self.preprocess_image(image)
        with torch.no_grad():
            output, _ = self.model(img)

        output = non_max_suppression_kpt(output, self.conf_thres, self.iou_thres, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)
        output = output_to_keypoint(output)
        nimg = img[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

        kpoints =[]
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
            kpoints.append(self.YL2XYS(output[idx,7:]))
        if output.shape[0] >0:
            results = np.zeros([output.shape[0],5])
            results[:,0] = output[:, 2]-output[:, 4]/2
            results[:,1] = output[:, 3]-output[:, 5]/2
            results[:,2] = output[:, 2]+output[:, 4]/2
            results[:,3] = output[:, 3]+output[:, 5]/2
            results[:,4] = output[:, -1]

    
            return nimg,results,kpoints
        else: 
            return nimg,None,None

if __name__ == '__main__':
    model = yolov7_pose('./weight/yolov7-w6-pose.pt')
    image = cv2.imread('img.jpg')
    image,keypoints = model.predict(image)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()