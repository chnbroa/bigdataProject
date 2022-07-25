from tensorflow.keras.models import load_model
import tensorflow as tf
import random
import numpy as np
import os
import cv2
import glob
from PIL import Image
import PIL.ImageOps
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.layers import Flatten, MaxPooling2D, Conv2D
from keras.callbacks import Callback


class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.05:
            print('\n Stop training.')
            self.model.stop_training = True


callbacks = myCallback()
file_dir = "C:/data/"
image_size = (224, 224)
cartegories = os.listdir(file_dir)
num_classes = len(cartegories)
X = []
Y = []

print(cartegories)
# 이미지 전처리
for i, categorie in enumerate(cartegories):
    label = [0 for i in range(num_classes)]
    label[i] = 1
    print(categorie)
    img_dir = file_dir+categorie + "/"
    img_list = os.listdir(img_dir)
    for img_list_dir in img_list:
        img_path = img_dir+img_list_dir
        img = cv2.imdecode(np.fromfile(
            img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, image_size)
        X.append(img / 256)
        Y.append(label)
        print(img_path)


# 참고: https://studying-modory.tistory.com/entry/210305-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%8F%99%EB%AC%BC-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%9D%B8%EC%8B%9D-%ED%8C%A8%EC%85%98-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%B6%84%EB%A5%98


Xtr = np.array(X)
Ytr = np.array(Y)


X_train, Y_train = Xtr, Ytr


print(X_train.shape)
print(X_train.shape[1:])


model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same', activation='relu',
                 input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(20, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Classifier
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(9, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=['accuracy'])

hist = model.fit(X_train, Y_train, epochs=20,
                 batch_size=4, callbacks=[callbacks])


model.save('model.h5')
print("Saved model")


model = load_model('model.h5')
model.summary()
test_x = []

img = cv2.imdecode(np.fromfile('C:/test/a1_56(8).jpg',
                   dtype=np.uint8), cv2.IMREAD_UNCHANGED)
img = cv2.resize(img, image_size)
test_x.append(img / 256)
a = np.array(test_x)
modelpredict = model.predict(a)
print(modelpredict)


# 해야할 것
# 테스트 이미지 전처리 다시하기
# 테스트 파일 만들기? 결과값 지정하기.
