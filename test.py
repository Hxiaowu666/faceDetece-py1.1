import cv2 as cv
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
#汉字转换函数
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "chinese.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
# 人脸识别函数
def face_recongnition(img):
    #读取文件信息
    fr = open('test.txt', 'r')
    existPerson = fr.readlines()
    #print(existPerson)  # 已存储   人数  姓名[]
    fr.close()

    # img = cv.flip(img, 1)  # 图像翻转
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # 转为灰度图
    face_detect = cv.CascadeClassifier(
        'E:\yan\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')  # 调用分类器
    face = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  # 检测到5次 成功认定
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=3)  # 画框
        lable, confidence = recognizer.predict(gray[y:y + h, x:x + w])  # 返回标签和置信度
        print(existPerson[lable])
        if confidence >= 80:  # 置信度大于50，返回unknow
            cv.putText(img, 'unknow', (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:
            img = cv2AddChineseText(img, existPerson[lable], (x + 10, y - 30), (0, 255, 0), 30)
             # cv.putText(img,str(names[lable-1]),(x+10,y-10),cv.FONT_HERSHEY_SIMPLEX,0.75, (0,255,0),1)
            cv.putText(img, (str(round(confidence, 3))), (x + 125, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 0, 0), 1)
    cv.imshow('result', img)
# 人脸检测函数
def face_detect(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转化为灰度图片，简化矩阵、提高运算速度
    # 调用人脸检测的级联分类器
    face_classifier = cv.CascadeClassifier(
        'E:\yan\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
    # 对人脸进行检测，每次图像缩小的比例为1.1，每一个目标至少检测5次
    face_feature = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # face_feature  返回的是人脸的特征数据，一个人脸返回一组特征；x个人脸返回x组特征
    # print(face_feature)
    for x, y, w, h in face_feature:  # 遍历x个人脸
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=3)  # 对人脸画框
        # print(x,y,w,h)
    cv.imshow('人脸检测', img)
#保存人脸信息
def saveFacefunc(path):
    faceSample = []  # 存储人脸数据
    faceName = []  # 存储人脸姓名
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]  # 获取到所有目标文件的完整路径
    # os.listdir(path) 获取目标文件夹的内容，并以字母顺序进行排序-----------------------为什么要排序？
    # os.path.join(path,f) 表示获取到文件的完整路径
    # print(f'列表已存储文件：{imagePath}')

    face_detect = cv.CascadeClassifier(
        'E:\yan\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')  # 调用人脸检测的级联分类器
    for imagepath in imagePath:
        PIL_img = Image.open(imagepath).convert('L')  # 打开该图片，L表示转化为灰度图像 ---简化矩阵、加快运算速度
        img_numpy = np.array(PIL_img, 'uint8')  # 将图像数据转换为数组
        faces = face_detect.detectMultiScale(img_numpy)  # 获取图片的人脸特征
        # print(f'脸部特征为:{faces}')
        id = int(os.path.split(imagepath)[1].split('.')[1])  # 仅获取序号
        # 这里的路径也可以根据实际需求来写
        # print(f'当前id为{id}')

        for x, y, w, h in faces:
            faceName.append(id)
            faceSample.append(img_numpy[y:y + h, x:x + w])  # numpy数组切片，从y取到y+h行，从x取到x+w列，构成新的数组，把所画的方框放入列表中
    return faceName, faceSample

while True:
    print('请选择：1.人脸采集   2.人脸训练   3.人脸识别  4.退出系统\n')
    choice = input()
    if choice == '1':
        #读取文件信息
        fr = open('test.txt', 'r')
        existPerson = fr.readlines()
        print(existPerson)  # 已存储   人数  姓名[]
        fr.close()
        # 读取摄像头
        cap = cv.VideoCapture(0)  # 获取摄像头  也可以读取视频路径
        num = 1
        print('请输入名字\n')
        name = input()
        # 获取图像
        while cap.isOpened():
            flag, frame = cap.read()  # 获取帧图片
            if not flag:
                break
            face_detect(frame)  # 人脸检测
            if cv.waitKey(1) == ord('s'):
                # imgFile = './savePhotos/'+'capture_' + str(num) + '.jpg'
                # imgFile = 'E:\\yan\\test4_faceDetect\\opencv_python\\savePhotos\\' + 'capture_' + str(num) + '.jpg'
                # cv.imwrite("E:\\yan\\test4_faceDetect\\opencv_python\\savePhotos\\" + name + "." + str(num) + ".jpg", frame)
                imgFile = 'E:\\yan\\test4_faceDetect\\opencv_python\\savePhotos\\' + name + '.' + str(int(
                    existPerson[0]) + 1) + '.' + str(num) + '.jpg'
                cv.imencode('.jpg', frame)[1].tofile(imgFile)  # 存储中文图片

                # path = './savePhotos/'
                # cv.imwrite(path + 'capture_' + str(num) + '.jpg', frame)
                print('保存成功!')
                num += 1
            if cv.waitKey(1) == 27:  # Esc对应的ascall码
                break
        existPerson[0] = (str(int(existPerson[0]) + 1) + '\n')
        existPerson.append(name + '\n')
        fw = open('test.txt', 'w')
        fw.writelines(existPerson)  # 更新文件信息
        print('照片已存储\n')
        fw.close()
        cap.release()
        cv.destroyAllWindows()
    elif choice == '2':
        if __name__ == '__main__':
            faceName, faceSample = saveFacefunc(path='.//savePhotos')  # 获取姓名和脸部特征
            # 采用LBPH算法
            recognizer = cv.face.LBPHFaceRecognizer_create()
            recognizer.train(faceSample, np.array(faceName))  # 训练
            # 保存训练好的文件
            recognizer.write('trainer.yml')
            print('文件已保存')
    elif choice == '3':
        # 导入已经训练完成的模型
        recognizer = cv.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer.yml')
        # 进行人脸识别
        cap = cv.VideoCapture(0)  # 打开摄像头
        while True:
            flag, frame = cap.read()
            if not flag:
                break
            face_recongnition(frame)  # 传入图像进行人脸识别
            if cv.waitKey(1) == 27:
                break

        # 释放内存
        cap.release()
        cv.destroyAllWindows()
    elif choice == '4':
        print('退出系统\n')
        break
    else:
        print('输入错误，重新选择')








