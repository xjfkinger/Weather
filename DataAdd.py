import pandas as pd
import numpy as np
import tensorflow as tf

train_path = 'D:\\weather\\train\\Triandata/'
train_csv = pd.read_csv(r'D:/weather/newweather/train.csv')

# filenames = ['data/train/' + fname for fname in train_csv['id'].tolist()]
# labels = train_csv['has_cactus'].tolist()

train_name = ['D:/weather/Traindata/' + fname for fname in train_csv['FileName'].tolist()]
label_name = train_csv['type'].tolist()
IMAGE_SIZE = 100

import tensorflow as tf
import os

flag = 1


def trans(imgp, label, sav):
    img_path=img_dir+imgp
    # 读入图片
    img = tf.io.read_file(img_path)

    # 解码为tensor格式
    img = tf.image.decode_jpeg(img)
    print('shape:', img.shape, 'dtype:', img.dtype)

    # 翻转操作是镜像的，旋转是非镜像的
    imge = tf.image.random_brightness(img, max_delta=0.5)  # 随机增加亮度
    # 垂直翻转
    flipped_up_down = tf.image.flip_up_down(imge)
    # 水平翻转
    flipped_left_right = tf.image.flip_left_right(imge)

    # 编码tensor
    img_up = tf.image.encode_jpeg(flipped_up_down)
    img_left=tf.image.encode_jpeg(flipped_left_right)
    f1name = os.path.join(sav, (imgp[:-4] + '_aug' + '1' + '.jpg'))
    f2name = os.path.join(sav, (imgp[:-4] + '_aug' + '2' + '.jpg'))
    # 写入文件里面
    with tf.io.gfile.GFile(f1name, 'wb') as file1:
        file1.write(img_up.numpy())
    with tf.io.gfile.GFile(f2name, 'wb') as file2:
        file2.write(img_left.numpy())

    print("图像增强完毕")


    import csv

    # 1. 创建文件对象
    # 3. 构建列表头
    head = ["FileName", "type"]
    data = [
     {"FileName": f1name, "type": label},
     {"FileName": f2name, "type": label}
    ]
    if flag==1:
      with open("D:/weather/newtrain.csv", "a+", encoding="utf-8", newline="") as f:
            csvf = csv.DictWriter(f, head)
            csvf.writerows(data)
    else:
        with open("D:/weather/newtrain.csv", "a+", encoding="utf-8", newline="") as f:
            csvf = csv.DictWriter(f)
            csvf.writerows(data)

    # 5. 关闭文件
    f.close()

img_dir = 'D:/weather/Traindata/'
sav_dir = 'D:/weather/newweather/newTraindata/traindata/'
if not os.path.exists(sav_dir):  # 如果不存在target_file，则创造一个
    os.makedirs(sav_dir)

# 获取图像文件列表
img_list = train_csv['FileName'].tolist()
label_list = train_csv['type'].tolist()
num = len(label_list)
print(num)
# 遍历图像文件

for i in range(num):
    print("第%d张--------------", i)
    # 数据增强和扩增
    trans(img_list[i], label_list[i], sav_dir)


