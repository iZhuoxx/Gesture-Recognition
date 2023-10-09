import Datasets
import CNN_Net
import Trian_class
import paddle.distributed as dist
import paddle
import os
import matplotlib.pyplot as plt
import warnings

# os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
warnings.filterwarnings("ignore", category=Warning)  # 过滤报警信息
dist.init_parallel_env()

# if __name__ == '__main__':
#     dist.spawn(Trian_class.train)

# 加载数据
data_loader = Datasets.load_data(batch_size=40)
# 加载模型
model = paddle.DataParallel(CNN_Net.VGG19())
# model.eval()
# 训练模型
iters, losses = Trian_class.train(model=model, epoch_num=40, data_loader=data_loader, batch_size=40)

plt.figure()
plt.title("Accuracy", fontsize=20)
plt.xlabel("epoch", fontsize=14)
plt.ylabel("accuracy", fontsize=14)
plt.plot(iters, losses, 'r--')
plt.legend(["VGG19 Accuracy"])
plt.grid()
plt.show()
