import Datasets
import CNN_Net
import paddle.distributed as dist
from tqdm import tqdm
from paddle.static import InputSpec
import paddle
import matplotlib.pyplot as plt
import numpy as np
import time
from paddle.vision import transforms
from paddle.io import DataLoader
from sklearn.metrics import confusion_matrix
import warnings
import os
from PIL import Image


# 加载测试集
def load(alist='./test_final_Datasets'):
    list = os.listdir(alist)
    i = 0
    j = 0
    source_types = []
    crop_imgs = []
    str = "\r{0}"
    for path in list:
        crop_path = alist + '/' + path
        crop_list = os.listdir(crop_path)

        for name in crop_list:
            i += 1
            source_types.append(j)
            temp_img = Image.open(crop_path + '/' + name)
            cropImg = temp_img.convert('RGB')
            cropImg = transforms.ToTensor()(cropImg).unsqueeze(0)
            crop_imgs.append(cropImg)
            # crop_imgs.append(crop_path + '/' + name)
        j += 1
    source_types = np.array(source_types).astype("int64")
    return source_types, crop_imgs
    # print(source_types)


def plot_confusion_matrix(confusion_mat):
    """将混淆矩阵画图并显示出来"""
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.blue)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(confusion_mat.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# load()

# paddle.callbacks

batch_size = 36
epochs = 40

warnings.filterwarnings("ignore", category=Warning)  # 过滤报警信息
dist.init_parallel_env()

# net = paddle.DataParallel(net.net(32,32))
net = paddle.DataParallel(CNN_Net.GoogLeNet())

# 加载数据
dataset = Datasets.MyDataset()
TranData = DataLoader(dataset, batch_size=batch_size)

# 损失函数

# lossC=paddle.fluid.layers.cross_entropy()


optimizer = paddle.optimizer.Adam(parameters=net.parameters(), learning_rate=0.00001)
# optimizer_D=paddle.optimizer.Adam(parameters=net_D.parameters(),learning_rate=0.0001)

runloss = 0
# runloss_D=0

# 开始训练
t1 = time.time()
percente = []
loss123 = []
type1, imgs1 = load()
for epoch in range(epochs):
    runloss = 0
    # runloss_D = 0
    net.train()
    # net_D.train()

    for i, (cropimg, sourcetype) in tqdm(enumerate(TranData, 1)):
        # 清空梯度流
        optimizer.clear_grad()

        # 训练
        output = net(cropimg)
        # print(output)
        # print(output, '\n', sourcetype)

        loss = paddle.fluid.layers.cross_entropy(output, sourcetype)
        loss = paddle.fluid.layers.mean(loss)

        loss.backward()
        optimizer.step()

        # print("\nloss:", loss.item())
        # optimizer_G.step()

        runloss += loss.item()

    avgloss = runloss / i
    # avgloss_D=runloss_D/i

    print('[INFO] Generator Epoch %d loss: %.3f\n' % (epoch + 1, avgloss), end="")

    a = 0
    sum = 0
    for i in type1:
        t = net(imgs1[a])
        a += 1
        if i == np.argmax(t):
            sum += 1
    print("正确率：", sum / len(type1), "%", sep="")
    percente.append(sum / len(type1))
    loss123.append(avgloss)
    # print('[INFO] Descriminator Epoch %d loss: %.3f' % (epoch + 1, avgloss_D))
print("训练时间:", time.time() - t1)
print(percente)
print(loss123)

"""制作混淆矩阵"""
a = 0
sum = 0
test_label = []
for i in type1:
    t = net(imgs1[a])
    a += 1
    test_label.append(np.argmax(t))

confusion_mat = confusion_matrix(type1, test_label)
plot_confusion_matrix(confusion_mat)

paddle.jit.save(net, "example.dy_model_new_vgg16/linear", input_spec=[InputSpec(shape=(1, 3, 96, 96), dtype='float32')])
