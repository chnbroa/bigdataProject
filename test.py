import random
import numpy as np
import os
import cv2
import glob
from PIL import Image
import PIL.ImageOps    

#다음 변수를 수정하여 새로 만들 이미지 갯수를 정합니다.
num_augmented_images = 80

data_name = ["검은별무늬병", "그을음병", "노균병","덩굴마름병","모자이크병","세균성모무늬병","탄저병","흰가루병","정상"]
file_path = f'C:/data/{data_name[3]}/'
save_file_path = f'C:/data/{data_name[3]}증폭/'

if not os.path.isdir(save_file_path): #폴더가 존재하지 않는다면 폴더 생성추가
    os.makedirs(save_file_path)

file_names = os.listdir(file_path)
total_origin_image_num = len(file_names)
augment_cnt = 1

for i in range(1, num_augmented_images):
    change_picture_index = random.randrange(1, total_origin_image_num-1)
    print(change_picture_index)
    print(file_names[change_picture_index])
    file_name = file_names[change_picture_index]
    origin_image_path = file_path + file_name
    print(origin_image_path)
    image = Image.open(origin_image_path)
    random_augment = random.randrange(1,4)
    
    if(random_augment == 1):
        #이미지 좌우 반전
        print("invert")
        inverted_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        inverted_image.save(save_file_path + 'inverted_' + str(augment_cnt) + '.png')
        
    elif(random_augment == 2):
        #이미지 기울이기
        print("rotate")
        rotated_image = image.rotate(random.randrange(-20, 20))
        rotated_image.save(save_file_path + 'rotated_' + str(augment_cnt) + '.png')
        
    elif(random_augment == 3):
        #노이즈 처리
        #파일 cv2 한글처리? 추가
        img = cv2.imdecode(np.fromfile(origin_image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        #img = cv2.imread(origin_image_path)
        print("noise")
        row,col,ch= img.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy_array = img + gauss
        noisy_image = Image.fromarray(np.uint8(noisy_array)).convert('RGB')
        noisy_image.save(save_file_path + 'noiseAdded_' + str(augment_cnt) + '.png')
        
    augment_cnt += 1
    
    
#출처 https://github.com/BUZZINGPolarBear/Why_Am_I_ALONE/blob/master/Image_Augmentation.ipynb
