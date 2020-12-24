import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
model=tf.keras.models.load_model("test6_80.h5")
img_path="./testimg/0cf01a354ec84e4fb3b13531484ee987.jpeg"
img = image.load_img(img_path, target_size=(100, 100))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
pred_class = model.predict(img)
max_index = np.argmax(pred_class, axis=-1)
acc=pred_class[0][int(max_index)]*100
weather_list=['雨凇','雾凇','雾霾','霜','露','结冰','降雨','降雪','冰雹']
print("结果为{}的准确率是{:.2f}%".format(weather_list[int(max_index)],acc))

import cv2

img=cv2.imread(img_path,cv2.IMREAD_COLOR)    # 打开文件
img=cv2.resize(img,(500,450))
cv2.imshow('weatherPredict',img)
cv2.waitKey(0)  # 窗口无限等待
