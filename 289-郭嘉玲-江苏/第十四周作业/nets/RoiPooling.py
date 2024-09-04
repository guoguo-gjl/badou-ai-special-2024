# 定义了一个自定义的 Keras 层 RoiPoolingConv，用于在目标检测中处理候选区域（RoI）

from keras.engine.topology import Layer
import keras.backend as K

# 如果后端是 TensorFlow，则导入 tensorflow，因为后续代码中会使用 TensorFlow 的特定函数
if K.backend() == 'tensorflow':
    import tensorflow as tf

class RoiPoolingConv(Layer):   # 继承自keras的Layer类

    # 初始化方法。初始化池化大小和候选区域数量
    def __init__(self, pool_size, num_rois, **kwargs):
        # 获取当前的图像数据格式。
        # ‘tf’ 表示 TensorFlow 风格，即 (height, width, channels)；
        # ‘th’ 表示 Theano 风格，即 (channels, height, width)
        self.dim_ordering = K.image_data_format()
        self.pool_size = pool_size  # 池化层的尺寸
        self.num_rois = num_rois  # 区域建议的数量
        super(RoiPoolingConv, self).__init__(**kwargs)  # 调用基类的初始化方法。**kwargs 允许灵活地传递参数，使得子类可以接受并处理父类需要的所有参数。

    # build方法根据输入的形状设置 nb_channels，即输入图像的通道数
    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]  # 提取通道数

    # 计算输出形状
    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    # 层的前向传播计算
    def call(self, x, mask=None):
        assert(len(x) == 2)  # 确保输入 x 的长度为2
        # 从输入 x 中取出两个元素，img 是图像数据，rois 是感兴趣区域
        img = x[0]
        rois = x[1]
        outputs = []
        # 处理每个区域建议roi，从rois中提取出x, y, w, h的位置和尺寸信息
        for roi_idx in range(self.num_rois):
            # 从 rois 中提取每个区域的 x 坐标、y 坐标、宽度 w 和高度 h
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]
            # 将 x、y、w 和 h 转换为 int32 数据类型
            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')
            # 从图像 img 中提取出感兴趣区域（ROI），并使用 tf.image.resize_images 将其调整为指定的 self.pool_size 大小
            rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            # 将处理后的感兴趣区域 rs 添加到 outputs 列表中
            outputs.append(rs)

        # 将列表上所有结果在第0维连接，接着重新塑形以符合形状，最后调整维度顺序并返回最终输出
        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output