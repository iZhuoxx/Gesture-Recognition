import os
from paddle.io import Dataset
from paddle.vision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
import paddle
import cv2
from keras.preprocessing.image import ImageDataGenerator

# datagen=ImageDataGenerator(featurewise_center=False,
#   samplewise_center=False,
#   featurewise_std_normalization=False,
#   samplewise_std_normalization=False,
#   zca_whitening=True,
#   rotation_range=90,
#   width_shift_range=0.2,
#   height_shift_range=0.2,
#   shear_range=0,
#   zoom_range=0.001,
#   channel_shift_range=10,
#   fill_mode='nearest',
#   cval=0.,
#   horizontal_flip=True,
#   vertical_flip=True,
#   rescale=None,
#   preprocessing_function=None)

datagen = ImageDataGenerator(
    rotation_range=10,
    channel_shift_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# alist_2 = ['./data_final/0', './data_final/1', './data_final/2',
#          './data_final/3', './data_final/4']
alist_2 = './data_middle_test/'
alist_3 = './data_middle_test_after/'
ALPHA = 1.05  # 对比度
BETA = 30  # 亮度


# ALPHA = 0.7  # 对比度
# BETA = 0.5  # 亮度


def data_enhance(alist='./test_middle_Datasets'):
    imgs = []
    i = 0
    light_on = 0
    blist = [28, 20, 33]
    str = "\r{0}"
    list = os.listdir(alist)  # 数据集下面的子文档
    for path in list:
        crop_path = alist + '/' + path  # 子文档下面的目录，即具体图片的目录
        train_list = os.listdir(crop_path)
        for name in train_list:
            print(str.format(i), end='')
            img_path = alist + '/' + path + "/" + name
            imgs.append(img_path)

        for index in range(len(imgs)):
            # 读取图片
            if light_on == 1:
                img0 = Image.open(imgs[index])
                arr = light_adjust(img0, ALPHA, BETA)
                img1 = Image.fromarray(arr)
                img1 = img1.convert("RGB")
                # img0.show()
                # me = merge(img0, img1)
                # me = Image.fromarray(me)
                # me = me.convert("RGB")
                # me.save("test.jpg")
                img1.save("test.jpg")
                img = cv2.imread("test.jpg")
                # cv2.imshow('pic_name', img)
                # cv2.waitKey(0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.expand_dims(img, axis=0)
            if light_on == 0:
                img = cv2.imread(imgs[index])
                # img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
                # img = cv2.imwrite('./test_final_Datasets/' + "{0}/".format(i) + imgs[index].split('/')[-1], img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.expand_dims(img, axis=0)
            # 实例化对象
            agu = ImageDataGenerator(channel_shift_range=1, rotation_range=30, horizontal_flip=True)  # 随机旋转0-40度之间
            # datagen.fit(img)
            # print(type(agu))
            # 变换并保存
            for a in range(2):
                next(agu.flow(img, save_to_dir='./test_final_Datasets/' + "{0}/".format(i), save_format='jpg'))
                # next(datagen.flow(img, save_to_dir='./data_final_test/' + "{0}/".format(i), save_format='jpg'))
        i += 1
        imgs = []


# alist_1 = ["./datasets/animal/cat", "./datasets/animal/cow",
#            "./datasets/animal/hours", "./datasets/animal/dog", "./datasets/animal/pig"]
alist_1 = "./hand_Datasets"


# list = os.listdir(alist_1)  # 数据集下面的子文档


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError:
            pass


def data_processes(alist="./test_hand_Datasets"):
    # 数据提取

    imgs = []
    i = 0
    str = "\r{0}"
    list = os.listdir(alist)  # 数据集下面的子文档
    for path in list:
        crop_path = alist + '/' + path  # 子文档下面的目录，即具体图片的目录
        train_list = os.listdir(crop_path)
        for name in train_list:
            print(str.format(i), end='')
            img_path = alist + '/' + path + "/" + name
            imgs.append(img_path)
        for index in range(len(imgs)):
            img = Image.open(imgs[index])
            img = img.resize((120, 120), Image.BICUBIC)
            img.save("test.jpg")
            img = cv2.imread("test.jpg")
            img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
            img = cv2.imwrite('./test_middle_Datasets/' + "{0}/".format(i) + imgs[index].split('/')[-1], img)
            # img.save('./data_middle/' + "{0}/".format(i) + "y" + imgs[index].split('/')[-1])
            # img.save('./test_middle_Datasets/' + "{0}/".format(i) + imgs[index].split('/')[-1])
        i += 1
        imgs = []


# data_processes()


def light(img):
    h, w, c = img.shape
    print(h, w)
    start_x = 510
    start_y = 1000
    for i in range(200):
        for j in range(300):
            if img[start_x + i, start_y + j][0] > 40:
                a = random.randint(250, 255)
                b = random.randint(250, 255)
                c = random.randint(250, 255)
                img[start_x + i, start_y + j][0] = a
                img[start_x + i, start_y + j][1] = b
                img[start_x + i, start_y + j][2] = c
    cv2.imwrite('test.jpg', img)
    plt.imshow(img)
    plt.show()


def light_adjust(img, a, b):
    c, r = img.size
    arr = np.array(img)
    for i in range(r):
        for j in range(c):
            for k in range(3):
                temp = arr[i][j][k] * a + b
                if temp > 255:
                    arr[i][j][k] = 2 * 255 - temp
                else:
                    arr[i][j][k] = temp
    return arr


def merge(im1, im2):
    a1 = np.array(im1)
    a2 = np.array(im2)
    # arr = np.hstack((a1, a2))
    arr = np.vstack((a1, a2))
    return arr


# def merge_44([im1, im2, im3, im4, im5, im6, im7, im8,
#              im9, im10, im11, im12, im13, im14, im15, im16]):
def merge_44(list_pic):
    list_num = []
    for pic in list_pic:
        num = np.array(pic)
        list_num.append(num)
    arr = np.hstack((list_num[0], list_num[1], list_num[2])) #, list_num[3], list_num[4]))
    arr1 = np.hstack((list_num[3], list_num[4], list_num[5]))#, list_num[8], list_num[9]))
    arr2 = np.hstack((list_num[6], list_num[7], list_num[8]))#, list_num[11]))
    arr3 = np.hstack((list_num[9], list_num[10], list_num[11]))#, list_num[15]))
    arr4 = np.vstack((arr, arr1, arr2, arr3))
    return arr4


def _main():
    img = Image.open("t2.jpg")
    arr = light_adjust(img, ALPHA, BETA)
    img1 = Image.fromarray(arr)
    img1.save("t2_test.jpg")
    img2 = cv2.imread("t2_test.jpg")
    cv2.imshow('pic_name', img2)
    cv2.waitKey(0)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = np.expand_dims(img2, axis=0)
    # me = merge(img, img1)
    # img2 = Image.fromarray(img1)
    # cv2.imshow('pic_name', img2)
    # img1.show()


def data_enhance_single(alist=["./data_middle_test/4"]):
    imgs = []
    light_on = 0
    i = 4
    blist = [28, 20, 33]
    str = "\r{0}"
    for path in alist:
        train_list = os.listdir(path)
        for name in train_list:
            print(str.format(i), end='')
            img_path = path + "/" + name
            imgs.append(img_path)

        for index in range(len(imgs)):
            # 读取图片
            if light_on == 1:
                img0 = Image.open(imgs[index])
                arr = light_adjust(img0, ALPHA, BETA)
                img1 = Image.fromarray(arr)
                img1 = img1.convert("RGB")
                # img0.show()
                # me = merge(img0, img1)
                # me = Image.fromarray(me)
                # me = me.convert("RGB")
                # me.save("test.jpg")
                img1.save("test.jpg")
                img = cv2.imread("test.jpg")
                # cv2.imshow('pic_name', img)
                # cv2.waitKey(0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.expand_dims(img, axis=0)
            if light_on == 0:
                img = cv2.imread(imgs[index])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.expand_dims(img, axis=0)
            # 实例化对象
            agu = ImageDataGenerator(channel_shift_range=10, rotation_range=40, horizontal_flip=True)
            datagen.fit(img)
            # print(type(agu))
            # 变换并保存
            for a in range(1):
                next(agu.flow(img, save_to_dir='./data_final/' + "{0}/".format(i), save_format='jpg'))
        i += 1
        imgs = []


def data_processes_single(alist=["./hand_Datasets_single/4"]):
    # 数据提取

    imgs = []
    i = 4
    str = "\r{0}"
    for path in alist:
        train_list = os.listdir(path)
        for name in train_list:
            print(str.format(i), end='')
            img_path = path + "/" + name
            imgs.append(img_path)

        for index in range(len(imgs)):
            img = Image.open(imgs[index])
            img = img.resize((120, 120), Image.BICUBIC)
            img.save('./data_middle_single/' + "{0}/".format(i) + imgs[index].split('/')[-1])
        i += 1
        imgs = []


def noise_reduction():
    img = cv2.imread("test.jpg")
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    # img = cv2.GaussianBlur(img, (5, 5), 1)
    cv2.imshow('pic_name', img)
    # opening = cv2.cvtColor(opening, cv2.COLOR_BGR2RGB)
    cv2.waitKey(0)
    cv2.imwrite("test_after.jpg", img)

    #    opening = np.expand_dims(opening, axis=0)

    img = Image.open("test.jpg")
    img1 = Image.open("test_after.jpg")
    me = merge(img, img1)
    imgg = Image.fromarray(me)
    imgg.show()
    imgg.save("test_compare.jpg")


def merge_6():
    img0 = Image.open("0.jpg")
    img1 = Image.open("1.jpg")
    img2 = Image.open("2.jpg")
    img3 = Image.open("3.jpg")
    img4 = Image.open("4.jpg")
    img5 = Image.open("5.jpg")
    img6 = Image.open("6.jpg")
    me0 = merge(img0, img1)
    me1 = merge(me0, img2)
    me2 = merge(me1, img3)
    me3 = merge(me2, img4)
    me4 = merge(me3, img5)
    me5 = merge(me4, img6)
    imgg = Image.fromarray(me5)
    imgg.show()
    imgg.save("together.jpg")


def merge_16():
    imgs = []
    img_pic = []
    img_name = ["im1", "im2", "im3", "im4", "im5", "im6", "im7", "im8",
                "im9", "im10", "im11", "im12", "im13", "im14", "im15", "im16"]
    pic_list = os.listdir('./result')
    for name in pic_list:
        img_path = './result' + "/" + name
        imgs.append(img_path)

    for index in range(len(imgs)):
        img = Image.open(imgs[index])
        print(np.shape(img))
        img = img.resize((550, 470), Image.BICUBIC)
        img_pic.append(img)

    me = merge_44(img_pic)
    imgg = Image.fromarray(me)
    imgg.show()
    imgg.save("./result.png")


merge_16()

# merge_6()

# noise_reduction()

# _main()

# data_processes()

# data_enhance()

# data_processes_single()

# data_enhance_single()
