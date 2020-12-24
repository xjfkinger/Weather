import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing import image
dir='D:/weather/testdata/'
model = tf.keras.models.load_model("test6_80.h5")
test_csv = pd.read_csv(r"C:/Users/ASUS/Desktop/1.csv")
test_list =  test_csv['FileName'].tolist()
for filename in test_list:
    img = image.load_img(os.path.join(dir,filename), target_size=(100, 100))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    pred_class = model.predict(img)
    max_index = np.argmax(pred_class, axis=-1)
    import csv

    # 1. 创建文件对象
    # 3. 构建列表头
    head = ["FileName", "type"]
    data = [
        {"FileName": filename, "type": int(max_index)},
    ]

    with open("D:/weather/newtest.csv", "a+", encoding="utf-8", newline="") as f:
            csvf = csv.DictWriter(f, head)
            csvf.writerows(data)


    # 5. 关闭文件
    f.close()
    acc = pred_class[0][int(max_index)] * 100
    weather_list = ['雨凇', '雾凇', '雾霾', '霜', '露', '结冰', '降雨', '降雪', '冰雹']
    print("结果为{}的准确率是{:.2f}%".format(weather_list[int(max_index)], acc))
