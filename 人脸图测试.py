import dlib
import cv2

# 加载 Dlib 的人脸检测器和人脸关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../fatigue_detecting/model/shape_predictor_68_face_landmarks.dat")

# 加载图像
img = cv2.imread('../fatigue_detecting/images/mt.jpg')

# 将图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用 Dlib 检测人脸
faces = detector(gray)

# 对于每个检测到的人脸，打印特征点
for face in faces:
    landmarks = predictor(gray, face)
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

# 显示图像

cv2.namedWindow("Image", 0)
cv2.resizeWindow("Image", 400, 600)

cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
