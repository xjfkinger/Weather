# _*_coding:UTF-8_*_
import matplotlib;
from tensorflow.python.keras import regularizers

matplotlib.use('TkAgg')
import numpy as np
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import Image, display

IMAGE_SIZE = 100
BATCH_SIZE = 32


def do_img(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [100, 100])
    img /= 255.0
    print(img)
    return img, label


if __name__ == "__main__":
    train_path = 'D:\\weather\\train\\Triandata/'
    train_csv = pd.read_csv(r'D:/weather/train.csv')

    # filenames = ['data/train/' + fname for fname in train_csv['id'].tolist()]
    # labels = train_csv['has_cactus'].tolist()

    train_name = ['D:/weather/Traindata/' + fname for fname in train_csv['FileName'].tolist()]
    label_name = train_csv['type'].tolist()
    train_filenames, val_filenames, train_labels, val_labels = train_test_split(train_name, label_name, train_size=0.9,
                                                                                random_state=42)

    num_train = len(train_filenames)
    num_val = len(val_filenames)

    train_data = tf.data.Dataset.from_tensor_slices((tf.constant(train_filenames), tf.constant(train_labels))).map(
        do_img).shuffle(buffer_size=num_train).batch(BATCH_SIZE)

    val_data = tf.data.Dataset.from_tensor_slices((tf.constant(val_filenames), tf.constant(val_labels))).map(
        do_img).batch(BATCH_SIZE)

    print(num_train)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(100, 100, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'))
    model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.002)))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.002)))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],

                  )

    num_epochs = 80

    steps_per_epoch = round(num_train) // BATCH_SIZE

    val_steps = 20
    # history = model.fit(train_data, epochs=num_epochs, validation_data=val_data, validation_freq=1)
    history = model.fit(train_data.repeat(),
                        epochs=num_epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_data.repeat(),
                        validation_steps=val_steps,

                        )
    model.save('test6_80.h5')

    # predict = model.predict(val_data)
    # print(val_data[0])
    # print("****************************")
    #   # 取出第一张图片预测值最大值的索引，即预测为第几
    # print(np.argmax(predict[0]))

    acc = history.history['accuracy']

    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    y = model.predict(val_data)
    count = len(val_filenames)
    print(count)
    acc = 0
    for i in range(count):
        print("label: ", val_labels[i])
        print()
        print(y[i], '        -> Predict digit', np.argmax(y[i]))
        print("************")
        if (val_labels[i] == np.argmax(y[i])):
            acc += 1
    print("测试准确率：", acc / count)



