from paddle import nn
from paddle.nn import Layer
import paddle
from paddle.nn import Conv2D, MaxPool2D, AdaptiveAvgPool2D, Linear
import paddle.nn.functional as F


# 定义一个神经网络
class ResBlock(Layer):
    def __init__(self, inChannels, outChannels):
        super(ResBlock, self).__init__()
        self.cov1 = nn.Conv2D(kernel_size=3, out_channels=outChannels, in_channels=inChannels, stride=1, padding=3 // 2)
        self.Bn1 = nn.BatchNorm2D(num_features=outChannels)
        self.prelu = nn.PReLU()
        self.cov2 = nn.Conv2D(kernel_size=3, out_channels=outChannels, in_channels=inChannels, stride=1, padding=3 // 2)
        self.Bn2 = nn.BatchNorm2D(num_features=outChannels)

    # @paddle.jit.to_static
    def forward(self, input):
        x = input
        input = self.cov1(input)
        input = self.Bn1(input)
        input = self.prelu(input)
        input = self.cov2(input)
        input = self.Bn2(input)
        input += x
        return input


# 定义神经网络
class net(Layer):
    def __init__(self, inChannels, outChannels):
        super(net, self).__init__()

        # 卷积层
        self.cov1 = nn.Conv2D(3, outChannels, kernel_size=3, stride=1, padding=3 // 2)
        self.Bn1 = nn.BatchNorm2D(outChannels)
        self.prelu1 = nn.PReLU()
        self.cov2 = nn.Conv2D(inChannels, outChannels * 2, kernel_size=3, stride=1, padding=3 // 2)
        self.Bn2 = nn.BatchNorm2D(outChannels * 2)
        self.prelu2 = nn.PReLU()
        # 残差层
        self.Resblock = self.getRes(ResBlock, 128, 128, 8)
        # 卷积层
        self.cov4 = nn.Conv2D(inChannels * 2, outChannels * 2, kernel_size=3, stride=1, padding=3 // 2)
        self.Bn4 = nn.BatchNorm2D(outChannels * 2)
        self.prelu5 = nn.PReLU()

        self.covt2 = nn.Conv2D(inChannels * 2, outChannels * 4, kernel_size=3, stride=1, padding=3 // 2)
        self.Bn6 = nn.BatchNorm2D(outChannels * 4)
        self.prelu6 = nn.PReLU()

        self.adapt_avg = nn.AdaptiveAvgPool2D((6, 6))
        self.flatten = nn.Flatten()
        # 全链接层
        self.Dense1 = nn.Linear(in_features=outChannels * 6 * 6 * 4, out_features=outChannels * 16)
        self.lrelu1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(0.2)
        self.Dense2 = nn.Linear(in_features=outChannels * 16, out_features=outChannels)
        self.lrelu2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(0.2)
        self.Dense3 = nn.Linear(in_features=outChannels, out_features=2)
        self.softmax = nn.Softmax()

    def getRes(self, block, inChannels, outChannels, num):
        return nn.Sequential(*[block(inChannels, outChannels) for i in range(num)])

    # @paddle.jit.to_static
    def forward(self, input):
        input = self.cov1(input)
        input = self.Bn1(input)
        input = self.prelu1(input)
        input = self.cov2(input)
        input = self.Bn2(input)
        input = self.prelu2(input)

        x = input

        input = self.Resblock(input)

        input = self.cov4(input)
        input = self.Bn4(input)
        input += x
        input = self.prelu5(input)

        # input = self.covt1(input)
        # input = self.Bn5(input)
        # input = self.prelu4(input)
        input = self.covt2(input)
        input = self.Bn6(input)
        input = self.prelu6(input)

        input = self.adapt_avg(input)
        input = self.flatten(input)
        input = self.Dense1(input)
        input = self.lrelu1(input)
        input = self.drop1(input)
        input = self.Dense2(input)
        input = self.lrelu2(input)
        input = self.drop2(input)
        input = self.Dense3(input)
        input = self.softmax(input)
        return input


class VGG19(Layer):
    def __init__(self):
        super(VGG19, self).__init__()

        self.net = paddle.vision.models.vgg19(pretrained=False)
        self.net = nn.Sequential(*list(self.net.features.children())[::])  # [::])
        # self.adapt_avg = nn.AdaptiveAvgPool2D((7, 7))
        self.flatten = nn.Flatten()
        self.Dense1 = nn.Linear(in_features=4608, out_features=4096)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.Dense2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.Dense3 = nn.Linear(in_features=4096, out_features=7)
        self.softmax = nn.Softmax()

    def forward(self, input):
        input = self.net(input)
        # input=self.adapt_avg(input)
        input = self.flatten(input)
        input = self.Dense1(input)
        input = self.relu1(input)
        input = self.dropout1(input)
        input = self.Dense2(input)
        input = self.relu2(input)
        input = self.dropout2(input)
        input = self.Dense3(input)
        input = self.softmax(input)
        return input


class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(LeNet, self).__init__()
        # 创建卷积和池化层
        # 创建第1个卷积层
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        self.max_pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        # 尺寸的逻辑：池化层未改变通道数；当前通道数为6
        # 创建第2个卷积层
        self.conv2 = nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        # 创建第3个卷积层
        self.conv3 = nn.Conv2D(in_channels=16, out_channels=120, kernel_size=4)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        # 输入size是[28,28]，经过三次卷积和两次池化之后，C*H*W等于120
        self.fc1 = nn.Linear(in_features=120, out_features=64)
        # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        # 每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        x = F.sigmoid(x)
        x = self.max_pool1(x)
        x = F.sigmoid(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x


class Inception(paddle.nn.Layer):
    def __init__(self, c0, c1, c2, c3, c4, **kwargs):
        '''
        Inception模块的实现代码，

        c1,图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
        c2,图(b)中第二条支路卷积的输出通道数，数据类型是tuple或list,
               其中c2[0]是1x1卷积的输出通道数，c2[1]是3x3
        c3,图(b)中第三条支路卷积的输出通道数，数据类型是tuple或list,
               其中c3[0]是1x1卷积的输出通道数，c3[1]是3x3
        c4,图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
        '''
        super(Inception, self).__init__()
        # 依次创建Inception块每条支路上使用到的操作
        self.p1_1 = Conv2D(in_channels=c0, out_channels=c1, kernel_size=1, stride=1)
        self.p2_1 = Conv2D(in_channels=c0, out_channels=c2[0], kernel_size=1, stride=1)
        self.p2_2 = Conv2D(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1, stride=1)
        self.p3_1 = Conv2D(in_channels=c0, out_channels=c3[0], kernel_size=1, stride=1)
        self.p3_2 = Conv2D(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2, stride=1)
        self.p4_1 = MaxPool2D(kernel_size=3, stride=1, padding=1)
        self.p4_2 = Conv2D(in_channels=c0, out_channels=c4, kernel_size=1, stride=1)

        # # 新加一层batchnorm稳定收敛
        # self.batchnorm = paddle.nn.BatchNorm2D(c1+c2[1]+c3[1]+c4)

    def forward(self, x):
        # 支路1只包含一个1x1卷积
        p1 = F.relu(self.p1_1(x))
        # 支路2包含 1x1卷积 + 3x3卷积
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        # 支路3包含 1x1卷积 + 5x5卷积
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        # 支路4包含 最大池化和1x1卷积
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 将每个支路的输出特征图拼接在一起作为最终的输出结果
        return paddle.concat([p1, p2, p3, p4], axis=1)
        # return self.batchnorm()


class GoogLeNet(paddle.nn.Layer):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        # GoogLeNet包含五个模块，每个模块后面紧跟一个池化层
        # 第一个模块包含1个卷积层
        self.conv1 = Conv2D(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=1)
        # 3x3最大池化
        self.pool1 = MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第二个模块包含2个卷积层
        self.conv2_1 = Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        self.conv2_2 = Conv2D(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1)
        # 3x3最大池化
        self.pool2 = MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第三个模块包含2个Inception块
        self.block3_1 = Inception(192, 64, (96, 128), (16, 32), 32)
        self.block3_2 = Inception(256, 128, (128, 192), (32, 96), 64)
        # 3x3最大池化
        self.pool3 = MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第四个模块包含5个Inception块
        self.block4_1 = Inception(480, 192, (96, 208), (16, 48), 64)
        self.block4_2 = Inception(512, 160, (112, 224), (24, 64), 64)
        self.block4_3 = Inception(512, 128, (128, 256), (24, 64), 64)
        self.block4_4 = Inception(512, 112, (144, 288), (32, 64), 64)
        self.block4_5 = Inception(528, 256, (160, 320), (32, 128), 128)
        # 3x3最大池化
        self.pool4 = MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第五个模块包含2个Inception块
        self.block5_1 = Inception(832, 256, (160, 320), (32, 128), 128)
        self.block5_2 = Inception(832, 384, (192, 384), (48, 128), 128)
        # 全局池化，用的是global_pooling，不需要设置pool_stride
        self.pool5 = AdaptiveAvgPool2D(output_size=1)
        self.fc = Linear(in_features=1024, out_features=1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2_2(F.relu(self.conv2_1(x)))))
        x = self.pool3(self.block3_2(self.block3_1(x)))
        x = self.block4_3(self.block4_2(self.block4_1(x)))
        x = self.pool4(self.block4_5(self.block4_4(x)))
        x = self.pool5(self.block5_2(self.block5_1(x)))
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        return x


# ResNet中使用了BatchNorm层，在卷积层的后面加上BatchNorm以提升数值稳定性
# 定义卷积批归一化块
class ConvBNLayer(paddle.nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):

        """
        num_channels, 卷积层的输入通道数
        num_filters, 卷积层的输出通道数
        stride, 卷积层的步幅
        groups, 分组卷积的组数，默认groups=1不使用分组卷积
        """
        super(ConvBNLayer, self).__init__()

        # 创建卷积层
        self._conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False)

        # 创建BatchNorm层
        self._batch_norm = paddle.nn.BatchNorm2D(num_filters)

        self.act = act

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act == 'leaky':
            y = F.leaky_relu(x=y, negative_slope=0.1)
        elif self.act == 'relu':
            y = F.relu(x=y)
        return y


# 定义残差块
# 每个残差块会对输入图片做三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状与输入不一致，则对输入图片做1x1卷积，将其输出形状调整成一致
class BottleneckBlock(paddle.nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__()
        # 创建第一个卷积层 1x1
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        # 创建第二个卷积层 3x3
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        # 创建第三个卷积 1x1，但输出通道数乘以4
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        # 如果conv2的输出跟此残差块的输入数据形状一致，则shortcut=True
        # 否则shortcut = False，添加1个1x1的卷积作用在输入数据上，使其形状变成跟conv2一致
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        # 如果shortcut=True，直接将inputs跟conv2的输出相加
        # 否则需要对inputs进行一次卷积，将形状调整成跟conv2输出一致
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y


# 定义ResNet模型
class ResNet(paddle.nn.Layer):
    def __init__(self, layers=50, class_dim=1):
        """

        layers, 网络层数，可以是50, 101或者152
        class_dim，分类标签的类别数
        """
        super(ResNet, self).__init__()
        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            # ResNet50包含多个模块，其中第2到第5个模块分别包含3、4、6、3个残差块
            depth = [3, 4, 6, 3]
        elif layers == 101:
            # ResNet101包含多个模块，其中第2到第5个模块分别包含3、4、23、3个残差块
            depth = [3, 4, 23, 3]
        elif layers == 152:
            # ResNet152包含多个模块，其中第2到第5个模块分别包含3、8、36、3个残差块
            depth = [3, 8, 36, 3]

        # 残差块中使用到的卷积的输出通道数
        num_filters = [64, 128, 256, 512]

        # ResNet的第一个模块，包含1个7x7卷积，后面跟着1个最大池化层
        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool2d_max = nn.MaxPool2D(
            kernel_size=3,
            stride=2,
            padding=1)

        # ResNet的第二到第五个模块c2、c3、c4、c5
        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                # c3、c4、c5将会在第一个残差块使用stride=2；其余所有残差块stride=1
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        # 在c5的输出特征图上使用全局池化
        self.pool2d_avg = paddle.nn.AdaptiveAvgPool2D(output_size=1)

        # stdv用来作为全连接层随机初始化参数的方差
        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        # 创建全连接层，输出大小为类别数目，经过残差网络的卷积和全局池化后，
        # 卷积特征的维度是[B,2048,1,1]，故最后一层全连接的输入维度是2048
        self.out = nn.Linear(in_features=2048, out_features=class_dim,
                             weight_attr=paddle.ParamAttr(
                                 initializer=paddle.nn.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, [y.shape[0], -1])
        y = self.out(y)
        return y


class VGG16(Layer):
    def __init__(self):
        super(VGG16, self).__init__()

        self.net = paddle.vision.models.vgg16(pretrained=False)
        self.net = nn.Sequential(*list(self.net.features.children())[::])  # [::])
        # self.adapt_avg = nn.AdaptiveAvgPool2D((7, 7))
        self.flatten = nn.Flatten()
        self.Dense1 = nn.Linear(in_features=4608, out_features=4096)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.Dense2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.Dense3 = nn.Linear(in_features=4096, out_features=7)
        self.softmax = nn.Softmax()

    def forward(self, input):
        input = self.net(input)
        # input=self.adapt_avg(input)
        input = self.flatten(input)
        input = self.Dense1(input)
        input = self.relu1(input)
        input = self.dropout1(input)
        input = self.Dense2(input)
        input = self.relu2(input)
        input = self.dropout2(input)
        input = self.Dense3(input)
        input = self.softmax(input)
        return input
#
# def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
#     if type(kernel_size) is int:
#         use_large_impl = kernel_size > 5
#     else:
#         assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
#         use_large_impl = kernel_size[0] > 5
#     if in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
#         # TODO more efficient PyTorch implementations of large-kernel convolutions. Pull-requests are welcomed.
#         # TODO Or you may try MegEngine. We have integrated an efficient implementation into MegEngine and it will automatically use it.
#         return Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                          padding=padding, dilation=dilation, groups=groups, bias=bias)
#     else:
#         return Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                          padding=padding, dilation=dilation, groups=groups, bias=bias)
#
#
# def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
#     if padding is None:
#         padding = kernel_size // 2
#     result = nn.Sequential()
#     result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#                                          stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
#     result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
#     return result
#
#
# def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
#     if padding is None:
#         padding = kernel_size // 2
#     result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#                      stride=stride, padding=padding, groups=groups, dilation=dilation)
#     result.add_module('nonlinear', nn.ReLU())
#     return result
#
#
# def fuse_bn(conv, bn):
#     kernel = conv.weight
#     running_mean = bn.running_mean
#     running_var = bn.running_var
#     gamma = bn.weight
#     beta = bn.bias
#     eps = bn.eps
#     std = (running_var + eps).sqrt()
#     t = (gamma / std).reshape(-1, 1, 1, 1)
#     return kernel * t, beta - running_mean * gamma / std
#
#
# class ReparamLargeKernelConv(nn.Module):
#
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride, groups,
#                  small_kernel,
#                  small_kernel_merged=False):
#         super(ReparamLargeKernelConv, self).__init__()
#         self.kernel_size = kernel_size
#         self.small_kernel = small_kernel
#         # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
#         padding = kernel_size // 2
#         if small_kernel_merged:
#             self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#                                           stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
#         else:
#             self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#                                       stride=stride, padding=padding, dilation=1, groups=groups)
#             if small_kernel is not None:
#                 assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
#                 self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
#                                           stride=stride, padding=small_kernel // 2, groups=groups, dilation=1)
#
#     def forward(self, inputs):
#         if hasattr(self, 'lkb_reparam'):
#             out = self.lkb_reparam(inputs)
#         else:
#             out = self.lkb_origin(inputs)
#             if hasattr(self, 'small_conv'):
#                 out += self.small_conv(inputs)
#         return out
#
#     def get_equivalent_kernel_bias(self):
#         eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
#         if hasattr(self, 'small_conv'):
#             small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
#             eq_b += small_b
#             #   add to the central part
#             eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
#         return eq_k, eq_b
#
#     def merge_kernel(self):
#         eq_k, eq_b = self.get_equivalent_kernel_bias()
#         self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
#                                       out_channels=self.lkb_origin.conv.out_channels,
#                                       kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
#                                       padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
#                                       groups=self.lkb_origin.conv.groups, bias=True)
#         self.lkb_reparam.weight.data = eq_k
#         self.lkb_reparam.bias.data = eq_b
#         self.__delattr__('lkb_origin')
#         if hasattr(self, 'small_conv'):
#             self.__delattr__('small_conv')
#
#
# class ConvFFN(nn.Module):
#
#     def __init__(self, in_channels, internal_channels, out_channels, drop_path):
#         super().__init__()
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.preffn_bn = nn.BatchNorm2d(in_channels)
#         self.pw1 = conv_bn(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0,
#                            groups=1)
#         self.pw2 = conv_bn(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
#                            groups=1)
#         self.nonlinear = nn.GELU()
#
#     def forward(self, x):
#         out = self.preffn_bn(x)
#         out = self.pw1(out)
#         out = self.nonlinear(out)
#         out = self.pw2(out)
#         return x + self.drop_path(out)
#
#
# class RepLKBlock(nn.Module):
#
#     def __init__(self, in_channels, dw_channels, block_lk_size, small_kernel, drop_path, small_kernel_merged=False):
#         super().__init__()
#         self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
#         self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
#         self.large_kernel = ReparamLargeKernelConv(in_channels=dw_channels, out_channels=dw_channels,
#                                                    kernel_size=block_lk_size,
#                                                    stride=1, groups=dw_channels, small_kernel=small_kernel,
#                                                    small_kernel_merged=small_kernel_merged)
#         self.lk_nonlinear = nn.ReLU()
#         self.prelkb_bn = nn.BatchNorm2d(in_channels)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         print('drop path:', self.drop_path)
#
#     def forward(self, x):
#         out = self.prelkb_bn(x)
#         out = self.pw1(out)
#         out = self.large_kernel(out)
#         out = self.lk_nonlinear(out)
#         out = self.pw2(out)
#         return x + self.drop_path(out)
#
#
# class RepLKNetStage(nn.Module):
#
#     def __init__(self, channels, num_blocks, stage_lk_size, drop_path,
#                  small_kernel, dw_ratio=1, ffn_ratio=4,
#                  use_checkpoint=False,  # train with torch.utils.checkpoint to save memory
#                  small_kernel_merged=False):
#         super().__init__()
#         self.use_checkpoint = use_checkpoint
#         blks = []
#         for i in range(num_blocks):
#             block_drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path
#             #   Assume all RepLK Blocks within a stage share the same lk_size. You may tune it on your own model.
#             replk_block = RepLKBlock(in_channels=channels, dw_channels=int(channels * dw_ratio),
#                                      block_lk_size=stage_lk_size,
#                                      small_kernel=small_kernel, drop_path=block_drop_path,
#                                      small_kernel_merged=small_kernel_merged)
#             convffn_block = ConvFFN(in_channels=channels, internal_channels=int(channels * ffn_ratio),
#                                     out_channels=channels,
#                                     drop_path=block_drop_path)
#             blks.append(replk_block)
#             blks.append(convffn_block)
#         self.blocks = nn.ModuleList(blks)
#
#     def forward(self, x):
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x)  # Save training memory
#             else:
#                 x = blk(x)
#         return x
#
#
# class RepLKNet(nn.Module):
#
#     def __init__(self, large_kernel_sizes, layers, channels, drop_path_rate, small_kernel,
#                  dw_ratio=1, ffn_ratio=4, in_channels=3, num_classes=1000,
#                  use_checkpoint=False,
#                  small_kernel_merged=False):
#         super().__init__()
#
#         base_width = channels[0]
#         self.use_checkpoint = use_checkpoint
#         self.num_stages = len(layers)
#         self.stem = nn.ModuleList([
#             conv_bn_relu(in_channels=in_channels, out_channels=base_width, kernel_size=3, stride=2, padding=1,
#                          groups=1),
#             conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=1, padding=1,
#                          groups=base_width),
#             conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=1, stride=1, padding=0, groups=1),
#             conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=2, padding=1,
#                          groups=base_width)])
#         # stochastic depth. We set block-wise drop-path rate. The higher level blocks are more likely to be dropped. This implementation follows Swin.
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
#         self.stages = nn.ModuleList()
#         self.transitions = nn.ModuleList()
#         for stage_idx in range(self.num_stages):
#             layer = RepLKNetStage(channels=channels[stage_idx], num_blocks=layers[stage_idx],
#                                   stage_lk_size=large_kernel_sizes[stage_idx],
#                                   drop_path=dpr[sum(layers[:stage_idx]):sum(layers[:stage_idx + 1])],
#                                   small_kernel=small_kernel, dw_ratio=dw_ratio, ffn_ratio=ffn_ratio,
#                                   use_checkpoint=use_checkpoint, small_kernel_merged=small_kernel_merged)
#             self.stages.append(layer)
#             if stage_idx < len(layers) - 1:
#                 transition = nn.Sequential(
#                     conv_bn_relu(channels[stage_idx], channels[stage_idx + 1], 1, 1, 0, groups=1),
#                     conv_bn_relu(channels[stage_idx + 1], channels[stage_idx + 1], 3, stride=2, padding=1,
#                                  groups=channels[stage_idx + 1]))
#                 self.transitions.append(transition)
#
#         self.norm = nn.BatchNorm2d(channels[-1])
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.head = nn.Linear(channels[-1], num_classes)
#
#     def forward_features(self, x):
#         x = self.stem[0](x)
#         for stem_layer in self.stem[1:]:
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(stem_layer, x)  # save memory
#             else:
#                 x = stem_layer(x)
#         for stage_idx in range(self.num_stages):
#             x = self.stages[stage_idx](x)
#             if stage_idx < self.num_stages - 1:
#                 x = self.transitions[stage_idx](x)
#         return x
#
#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.norm(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.head(x)
#         return x
#
#     def structural_reparam(self):
#         for m in self.modules():
#             if hasattr(m, 'merge_kernel'):
#                 m.merge_kernel()
#
#     #   If your framework cannot automatically fuse BN for inference, you may do it manually.
#     #   The BNs after and before conv layers can be removed.
#     #   No need to call this if your framework support automatic BN fusion.
#     def deep_fuse_BN(self):
#         # TODO
#         pass
#
#
# def create_RepLKNet31B(num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
#     return RepLKNet(large_kernel_sizes=[31, 29, 27, 13], layers=[2, 2, 18, 2], channels=[128, 256, 512, 1024],
#                     drop_path_rate=0.3, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
#                     small_kernel_merged=small_kernel_merged)
#
#
# def create_RepLKNet31L(num_classes=1000, use_checkpoint=True, small_kernel_merged=False):
#     return RepLKNet(large_kernel_sizes=[31, 29, 27, 13], layers=[2, 2, 18, 2], channels=[192, 384, 768, 1536],
#                     drop_path_rate=0.3, small_kernel=5, num_classes=num_classes, use_checkpoint=use_checkpoint,
#                     small_kernel_merged=small_kernel_merged)
#
#
# if __name__ == '__main__':
#     model = create_RepLKNet31B(small_kernel_merged=False)
#     model.eval()
#     print('------------------- training-time model -------------')
#     print(model)
#     x = torch.randn(2, 3, 224, 224)
#     origin_y = model(x)
#     model.structural_reparam()
#     print('------------------- after re-param -------------')
#     print(model)
#     reparam_y = model(x)
#     print('------------------- the difference is ------------------------')
#     print((origin_y - reparam_y).abs().sum())


# model = net(64, 64)
# model.train()



from paddle import nn
from paddle.nn import Layer
import paddle
import os
from paddle.io import Dataset
from paddle.vision import transforms
from PIL import Image
import numpy as np
import paddle.nn.functional as F
import paddle.distributed as dist
from tqdm import tqdm
from paddle.static import InputSpec
import time
from paddle.io import DataLoader
import warnings
import matplotlib.pyplot as plt


class VGG16(Layer):
    def __init__(self):
        super(VGG16, self).__init__()

        self.net = paddle.vision.models.vgg16(pretrained=False)
        self.net = nn.Sequential(*list(self.net.features.children())[::])  # [::])
        # self.adapt_avg = nn.AdaptiveAvgPool2D((7, 7))
        self.flatten = nn.Flatten()
        self.Dense1 = nn.Linear(in_features=4608, out_features=4096)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.Dense2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.Dense3 = nn.Linear(in_features=4096, out_features=7)
        self.softmax = nn.Softmax()

    def forward(self, input):
        input = self.net(input)
        # input=self.adapt_avg(input)
        input = self.flatten(input)
        input = self.Dense1(input)
        input = self.relu1(input)
        input = self.dropout1(input)
        input = self.Dense2(input)
        input = self.relu2(input)
        input = self.dropout2(input)
        input = self.Dense3(input)
        input = self.softmax(input)
        return input


class VGG19(Layer):
    def __init__(self):
        super(VGG19, self).__init__()

        self.net = paddle.vision.models.vgg19(pretrained=False)
        self.net = nn.Sequential(*list(self.net.features.children())[::])  # [::])
        # self.adapt_avg = nn.AdaptiveAvgPool2D((7, 7))
        self.flatten = nn.Flatten()
        self.Dense1 = nn.Linear(in_features=4608, out_features=4096)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.Dense2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.Dense3 = nn.Linear(in_features=4096, out_features=7)
        self.softmax = nn.Softmax()

    def forward(self, input):
        input = self.net(input)
        # input=self.adapt_avg(input)
        input = self.flatten(input)
        input = self.Dense1(input)
        input = self.relu1(input)
        input = self.dropout1(input)
        input = self.Dense2(input)
        input = self.relu2(input)
        input = self.dropout2(input)
        input = self.Dense3(input)
        input = self.softmax(input)
        return input


class MmdLoss(nn.Layer):

    def __init__(self, gama=1e3):
        super(MmdLoss, self).__init__()
        self.gama = gama

    def forward(self, source_features, target_features):

        mmd_loss = source_features.mean() - target_features.mean()

        return self.gama * mmd_loss

# ResNet中使用了BatchNorm层，在卷积层的后面加上BatchNorm以提升数值稳定性
# 定义卷积批归一化块
class ConvBNLayer(paddle.nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):

        """
        num_channels, 卷积层的输入通道数
        num_filters, 卷积层的输出通道数
        stride, 卷积层的步幅
        groups, 分组卷积的组数，默认groups=1不使用分组卷积
        """
        super(ConvBNLayer, self).__init__()

        # 创建卷积层
        self._conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False)

        # 创建BatchNorm层
        self._batch_norm = paddle.nn.BatchNorm2D(num_filters)

        self.act = act

    def forward(self, inputs):
        y = self._conv(inputs)
        # y = self._batch_norm(y)
        if self.act == 'leaky':
            y = F.leaky_relu(x=y, negative_slope=0.1)
        elif self.act == 'relu':
            y = F.relu(x=y)
        return y


# 定义残差块
# 每个残差块会对输入图片做三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状与输入不一致，则对输入图片做1x1卷积，将其输出形状调整成一致
class BottleneckBlock(paddle.nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__()
        # 创建第一个卷积层 1x1
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        # 创建第二个卷积层 3x3
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        # 创建第三个卷积 1x1，但输出通道数乘以4
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        # 如果conv2的输出跟此残差块的输入数据形状一致，则shortcut=True
        # 否则shortcut = False，添加1个1x1的卷积作用在输入数据上，使其形状变成跟conv2一致
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        # 如果shortcut=True，直接将inputs跟conv2的输出相加
        # 否则需要对inputs进行一次卷积，将形状调整成跟conv2输出一致
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y


# 定义ResNet模型
class ResNet(paddle.nn.Layer):
    def __init__(self, layers=50, class_dim=7):
        """

        layers, 网络层数，可以是50, 101或者152
        class_dim，分类标签的类别数
        """
        super(ResNet, self).__init__()
        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            # ResNet50包含多个模块，其中第2到第5个模块分别包含3、4、6、3个残差块
            depth = [3, 4, 6, 3]
        elif layers == 101:
            # ResNet101包含多个模块，其中第2到第5个模块分别包含3、4、23、3个残差块
            depth = [3, 4, 23, 3]
        elif layers == 152:
            # ResNet152包含多个模块，其中第2到第5个模块分别包含3、8、36、3个残差块
            depth = [3, 8, 36, 3]

        # 残差块中使用到的卷积的输出通道数
        num_filters = [64, 128, 256, 512]

        # ResNet的第一个模块，包含1个7x7卷积，后面跟着1个最大池化层
        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool2d_max = nn.MaxPool2D(
            kernel_size=3,
            stride=2,
            padding=1)

        # ResNet的第二到第五个模块c2、c3、c4、c5
        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                # c3、c4、c5将会在第一个残差块使用stride=2；其余所有残差块stride=1
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        # 在c5的输出特征图上使用全局池化
        self.pool2d_avg = paddle.nn.AdaptiveAvgPool2D(output_size=1)

        # stdv用来作为全连接层随机初始化参数的方差
        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        # 创建全连接层，输出大小为类别数目，经过残差网络的卷积和全局池化后，
        # 卷积特征的维度是[B,2048,1,1]，故最后一层全连接的输入维度是2048
        self.out = nn.Linear(in_features=2048, out_features=class_dim,
                             weight_attr=paddle.ParamAttr(
                                 initializer=paddle.nn.initializer.Uniform(-stdv, stdv)))
        self.softmax=nn.Softmax()

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, [y.shape[0], -1])
        y = self.out(y)
        y=self.softmax(y)
        return y



class MyDataset(Dataset):
    def __init__(self,alist='/home/aistudio/data/data125664/data_final_after'):
        super(MyDataset, self).__init__()
        # self.num=np.random.choice(14109, 14109, replace=False)
        self.alist=alist
        self.source_types=[]
        self.crop_imgs=[]
        i=0
        j=0
        str="\r{0}"
        list=os.listdir(alist)
        for path in list:
            crop_path=alist+'/'+path
            crop_list=os.listdir(crop_path)
            for name in crop_list:
                i += 1
                self.source_types.append(j)
                self.crop_imgs.append(crop_path+'/'+name)
            j += 1
        self.source_types=np.array(self.source_types).astype("int64")
        print(str.format(i))
        # self.source_types=to_categorical(self.source_types)
        # print(self.source_types)

    def __len__(self):
        # print(len(self.crop_imgs))
        self.num = np.random.choice(len(self.source_types), len(self.source_types), replace=False)
        # print(self.num)
        return len(self.crop_imgs)

    def __getitem__(self, index):

        temp_img = Image.open(self.crop_imgs[self.num[index]])
        cropImg = temp_img.convert('RGB')
        # print(temp_img.size)
        # if cropImg.size[0]>100:
        #     cropImg = cropImg.resize((100, 133), Image.BICUBIC)
        #     cropImg=cropImg.crop((0,33,100,133))
            # print(cropImg.size)
            # cropImg.show()
        # cropImg.show()
        cropImg=transforms.ToTensor()(cropImg)
        # print(cropImg)

        return cropImg, self.source_types[self.num[index]]



#加载测试集
def load(alist='/home/aistudio/data/data125664/test_final_Datasets'):
    list = os.listdir(alist)
    i = 0
    j = 0
    source_types=[]
    crop_imgs=[]
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
    return source_types,crop_imgs
    # print(source_types)

# load()

# paddle.callbacks

batch_size=64
epochs=40

warnings.filterwarnings("ignore", category=Warning)  # 过滤报警信息
dist.init_parallel_env()

# net = paddle.DataParallel(net.net(32,32))
net = paddle.DataParallel(ResNet())

#加载数据
dataset=MyDataset()
TranData=DataLoader(dataset,batch_size=batch_size)

#损失函数

# lossC=paddle.fluid.layers.cross_entropy()


optimizer=paddle.optimizer.Adam(parameters=net.parameters(),learning_rate=0.00000000000001)  # , weight_decay=paddle.regularizer.L2Decay(coeff=1e-5))
# 谷歌
# optimizer = paddle.optimizer.Momentum(learning_rate=0.00001, momentum=0.9, parameters=net.parameters(), weight_decay=0.001)
# optimizer_D=paddle.optimizer.Adam(parameters=net_D.parameters(),learning_rate=0.0001)

runloss=0
# runloss_D=0

#开始训练
t1=time.time()
percente=[]
loss123=[]
items=[]
for epoch in range(epochs):
    items.append(epoch)
    runloss = 0
    # runloss_D = 0
    net.train()
    # if epoch>=20:
       #  optimizer = paddle.optimizer.Adam(parameters=net.parameters(), learning_rate=0.000001)
    # net_D.train()

    for i, (cropimg,sourcetype) in tqdm(enumerate(TranData,1)):

        #清空梯度流
        optimizer.clear_grad()

        #训练
        output=net(cropimg)
        # print(output)
        # print(output, '\n', sourcetype)

        loss=paddle.fluid.layers.cross_entropy(output,sourcetype)
        loss = paddle.fluid.layers.mean(loss)

        loss.backward()
        optimizer.step()

        # print("\nloss:", loss.item())
        # optimizer_G.step()

        runloss+=loss.item()

    avgloss=runloss/i
    # avgloss_D=runloss_D/i

    print('[INFO] Generator Epoch %d loss: %.3f\n' % (epoch + 1, avgloss),end="")
    type1,imgs1=load()
    a=0
    sum=0
    for i in type1:
        t=net(imgs1[a])
        a+=1
        if i==np.argmax(t):
            sum+=1
    print("正确率：",sum/len(type1),"%",sep="")
    percente.append(sum/len(type1))
    loss123.append(avgloss)
    paddle.save(optimizer.state_dict(), './{}/mnist_epoch{}'.format("model",epoch)+'.pdopt')
    paddle.save(net.state_dict(), './{}/mnist_epoch{}'.format("model",epoch)+'.pdparams')
    # print('[INFO] Descriminator Epoch %d loss: %.3f' % (epoch + 1, avgloss_D))

print("训练时间:",time.time()-t1)
print(percente)
print(loss123)
paddle.jit.save(net,"/home/aistudio/ResNet/ResNet",input_spec=[InputSpec(shape=(1,3,96,96), dtype='float32')])
paddle.save(net.state_dict(), '/home/aistudio/ResNet-50.pdparams')