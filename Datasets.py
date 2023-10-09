import os
from paddle.io import Dataset
from paddle.vision import transforms
from PIL import Image
import numpy as np
from paddle.io import DataLoader
from sklearn.preprocessing import LabelBinarizer


# 创建MyDataset，继承paddle.io.Dataset这个类，构建一个迭代器
class MyDataset(Dataset):
    def __init__(self, alist='./data_final_after'):
        super(MyDataset, self).__init__()
        self.alist = alist  # 数据的路径
        self.source_types = []  # 定义label
        self.crop_imgs = []  # 定义图片数据
        i = 0
        j = 0
        str = "\r{0}"
        list = os.listdir(alist)  # 数据集下面的子文档
        for path in list:
            crop_path = alist + '/' + path  # 子文档下面的目录，即具体图片的目录
            crop_list = os.listdir(crop_path)  # 图片名称
            print("path is:{0}".format(path))
            print("j is:{0}".format(j))
            print(str.format(i))
            i += 1
            for name in crop_list:
                self.source_types.append(path)
                self.crop_imgs.append(crop_path + '/' + name)
            j += 1
        self.num = np.random.choice(len(self.crop_imgs), len(self.crop_imgs),
                                    replace=False)  # 图片数据的index，在__getitem__里面遍历打开图片的时候用到
        self.source_types = np.array(self.source_types).astype("int64")
        feature_transfer = self.source_types
        # print(self.source_types)

    def __len__(self):
        # print(len(self.crop_imgs))
        # print(self.source_types)
        # print(np.shape(self.source_types))
        return len(self.crop_imgs)

    def __getitem__(self, index):
        # 产生数据和label
        temp_img = Image.open(self.crop_imgs[self.num[index]])
        cropImg = temp_img.convert('RGB')
        cropImg = transforms.ToTensor()(cropImg)
        # print(temp_img.size)
        # if cropImg.size[0]>100:
        #     cropImg = cropImg.resize((100, 133), Image.BICUBIC)
        #     cropImg=cropImg.crop((0,33,100,133))
        # print(cropImg.size)
        # cropImg.show()
        # cropImg.show()

        return cropImg, self.source_types[self.num[index]]


def load_data(batch_size):
    dataset = MyDataset()
    data_loader = DataLoader(dataset, batch_size=batch_size)
    return data_loader

# data = MyDataset()
# len = data.__len__()
# print(len)
