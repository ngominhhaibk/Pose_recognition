# Pose_recognition
**Detect anh recognize pose**

## Nhận diện hành động:
+ Standing
+ Stand up
+ Sitting
+ Sit down
+ Lying Down
+ Walking
+ Fall Down

Nếu ngã (Fall Down) thì phát chuông cảnh báo



## Data:
Link dataset: https://www.kaggle.com/datasets/ngoduy/dataset-video-for-human-action-recognition


**Create data:**

.data/create_dataset_1.py : Từ video, gán nhãn thủ công 0-7 cho mỗi frame trong mỗi video

.data/create_dataset_2.py : Sử dụng yolov7_pose để tìm ra những keypoints tương ứng với label đã gán

.data/create_dataset_3.py : Giữ lại những điểm quan trọng main_parts và lưu vào file pickle để train



**Install:**
```
pip install -r requirements.txt
```

**Train:**
```
python train.py
```

**Pose recognition:**
```
python main.py
```


