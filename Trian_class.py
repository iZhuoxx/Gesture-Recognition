import paddle
import time
import numpy as np
from tqdm import tqdm
from paddle.static import InputSpec
import paddle.distributed as dist
from paddle.vision import transforms
from sklearn.preprocessing import LabelBinarizer


def train(model, epoch_num, data_loader, batch_size):
    use_gpu = False
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    # 修改1- 初始化并行环境
    # dist.init_parallel_env()
    # 修改2- 增加paddle.DataParallel封装
    # model = paddle.DataParallel(model)
    t1 = time.time()
    opt = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001, weight_decay=paddle.regularizer.L2Decay(coeff=1e-5))
    epoch = epoch_num
    model.train()
    iter = 0
    iters = []
    losses = []
    for num in range(epoch):
        runloss = 0
        # for batch_id, data in enumerate(data_loader()):
        for batch_id, (images, labels) in tqdm(enumerate(data_loader, 1)):
            # images = paddle.to_tensor(images)
            # labels = paddle.to_tensor(labels)
            opt.clear_grad()
            # print(np.shape(labels))

            # predicts, acc = model(images, labels)
            predicts = model(images)

            loss = paddle.fluid.layers.cross_entropy(predicts, labels)
            loss = paddle.fluid.layers.mean(loss)

            iters.append(batch_id + num * len(list(data_loader)))
            losses.append(loss)
            # print(batch_id)
            if batch_id % batch_size == 0:
                print('[INFO] Generator Epoch %d loss: %.3f\n' % (num + 1, loss))
                # print("epoch: {}, batch: {}, loss is: {}".format(num, batch_id, avg_loss.numpy()))

            loss.backward()
            opt.step()
            runloss += loss.item()

        avgloss = runloss / batch_id
        print('[INFO] Generator Epoch %d avg_loss: %.3f\n' % (num + 1, avgloss))
    # 保存模型参数
    print("训练时间:", time.time() - t1)
    paddle.jit.save(model, "model/hand_zhuo", input_spec=[InputSpec(shape=(1, 3, 24, 24), dtype='float32')])
    return iters, losses

