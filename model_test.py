from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2

image_size = (224, 224)
model = load_model('model.h5')
# model.summary()
name = {
    0: '검은별무늬병',
    1: '그을음병',
    2: '노균병',
    3: '덩굴마름병',
    4: '모자이크병',
    5: '세균성모무늬병',
    6: '정상',
    7: '탄저병',
    8: '흰가루병',
}


base_dir = "C:/test/"
img_list = os.listdir(base_dir)
for test_img in img_list:
    test_x = []
    img_path = base_dir + test_img
    img = cv2.imdecode(np.fromfile(
        img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, image_size)
    test_x.append(img / 256)
    a = np.array(test_x)
    modelpredict = model.predict(a).reshape(-1)
    print(img_path)
    index = np.argmax(modelpredict)
    print(f"인덱스  {index}")
    result = "병명: "+name[index] + \
        ", 정확도:{:.3f}%".format(modelpredict[index] * 100)
    print(result)


# 예측결과 처참...
# 검은별무늬병 다시
# 그을음병은다시 ( 제외  고려)
# 노균병 다시
