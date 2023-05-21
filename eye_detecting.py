# -*- coding: utf-8 -*-
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np # 数据处理的库 numpy
import argparse
import imutils
import time
import dlib
import cv2
import datetime,time
 
def eye_aspect_ratio(eye):
    # 垂直眼标志（X，Y）坐标
    A = dist.euclidean(eye[1], eye[5])# 计算两个集合之间的欧式距离
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平之间的欧几里得距离
    # 水平眼标志（X，Y）坐标
    C = dist.euclidean(eye[0], eye[3])
    # 眼睛长宽比的计算
    ear = (A + B) / (2.0 * C)
    # 返回眼睛的长宽比
    return ear
 
# 定义两个常数
# 眼睛长宽比
# 闪烁阈值
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 2
FATIGUE_EYE = 30  #瞌睡眨眼帧数，瞌睡时眨眼变慢，大约一秒
# 初始化帧计数器和眨眼总数
COUNTER = 0
TOTAL = 0
FATIGUE_TOTAL = 0
 
# 初始化DLIB的人脸检测器（HOG），然后创建面部标志物预测
print("[INFO] loading facial landmark predictor...")
# 第一步：使用dlib.get_frontal_face_detector() 获得脸部位置检测器
detector = dlib.get_frontal_face_detector()
# 第二步：使用dlib.shape_predictor获得脸部特征位置检测器
predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')
 
# 第三步：分别获取左右眼面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# 第四步：打开cv2 本地摄像头
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('F:/360下载软件/YawDD.rar/YawDD/YawDD dataset/Dash/Female/9-FemaleNoGlasses.mp4')
#cap = cv2.VideoCapture('../fatigue_detecting/images/疲劳状态.mp4')

flog_time = datetime.datetime.now()
# 从视频流循环帧
while True:
    # 第五步：进行循环，读取图片，并对图片做维度扩大，并进灰度化
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 第六步：使用detector(gray, 0) 进行脸部位置检测
    rects = detector(gray, 0)
    
    # 第七步：循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息
    for rect in rects:
        shape = predictor(gray, rect)
        
        # 第八步：将脸部特征信息转换为数组array的格式
        shape = face_utils.shape_to_np(shape)
        
        # 第九步：提取左眼和右眼坐标
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # 第十步：构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
 
        # 第十一步：使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
 
        # 第十二步：进行画图操作，用矩形框标注人脸
        #
        
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)    
 
        '''
            分别计算左眼和右眼的评分求平均作为最终的评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示进行了一次眨眼活动
        '''
        # 第十三步：循环，满足条件的，眨眼次数+1
        if ear < EYE_AR_THRESH:# 眼睛长宽比：0.2
            COUNTER += 1
           
        else:
            # 如果连续3帧都小于阈值，则表示进行了一次眨眼活动
            if COUNTER >= EYE_AR_CONSEC_FRAMES:# 阈值：2
                TOTAL += 1
            # 重置眼帧计数器
            if COUNTER >= FATIGUE_EYE:# 阈值：3
                FATIGUE_TOTAL += 1
            # 重置眼帧计数器

            COUNTER = 0
            

        # 第十四步：进行画图操作，68个特征点标识
        #for (x, y) in shape:
            #cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            
        # 第十五步：进行画图操作，同时使用cv2.putText将眨眼次数进行显示
        cv2.putText(frame, "Faces: {}".format(len(rects)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (150, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "COUNTER: {}".format(COUNTER), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (450, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #cv2.putText(frame, "FATIGUE_EAR: {:.2f}".format(FATIGUE_TOTAL), (400, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        #print('眼睛实时长宽比:{:.2f} '.format(ear))


# 获取当前时间
    current_time = datetime.datetime.now()
    
    # 计算当前时间与flog_time的时间的时间差（以秒为单位）
    time_difference = (current_time - flog_time).total_seconds()
    # 判断时间差是否大于60秒（1分钟）
    a = 120-time_difference
    #cv2.putText(frame, "countdown: {:.2f}".format(a), (10, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    if time_difference > 120:
        TOTAL = 0 
        FATIGUE_TOTAL = 0
        flog_time = datetime.datetime.now()
    # 确定疲劳提示:两分钟疲劳眨眼4次
    if FATIGUE_TOTAL >= 4 :
        cv2.putText(frame, "SLEEP!!!", (200, 200),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, "Press 'q': Quit", (20, 500),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 2)
    # 窗口显示 show with opencv
    cv2.imshow("Frame", frame)
    
    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# 释放摄像头 release camera
cap.release()
# do a bit of cleanup
cv2.destroyAllWindows()
