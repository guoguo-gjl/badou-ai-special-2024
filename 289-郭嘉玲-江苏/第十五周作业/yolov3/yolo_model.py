import numpy as np
import tensorflow as tf
import os

class yolo:

    # 初始化函数
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path, pre_train):
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.pre_train = pre_train
        self.anchors = self._get_anchors()
        self.classes = self._get_class()

    # 获取种类和先验框
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)  # 展开路径中 ~ 符号展开为用户主目录路径
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # 获取anchors
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path, 'r', encoding='utf-8') as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    # 正则化
    # 对输入的特征图进行批量归一化，改善训练稳定性和收敛速度。之后，使用 Leaky ReLU 激活函数进行非线性变换
    def _batch_normalization_layer(self, input_layer, name=None, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        bn_layer = tf.layers.batch_normalization(inputs=input_layer, momentum=norm_decay, epsilon=norm_epsilon, center=True, scale=True, training=training, name=name)
        return tf.nn.leaky_relu(bn_layer, alpha=0.1)

    # 卷积
    def _conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias=False, strides=1):
        conv = tf.layers.conv2d(
            inputs=inputs, filters=filters_num, kernel_size=kernel_size, strides=[strides, strides], kernel_initializer=tf.glorot_uniform_initializer(),
            padding=('SAME' if strides == 1 else 'VALID'),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4),
            use_bias=use_bias, name=name)
        return conv

    # 残差卷积
    def _Residual_block(self, inputs, filters_num, blocks_num, conv_index, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        layer = self._conv2d_layer(inputs, filters_num, kernel_size=3, strides=2, name='conv2d_'+str(conv_index))
        layer = self._batch_normalization_layer(layer, name='batch_normalization_'+str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            layer = self._conv2d_layer(layer, filters_num//2, kernel_size=1, strides=1, name='conv2d_'+str(conv_index))
            layer = self._batch_normalization_layer(layer, name='batch_normalization_'+str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index

    # 生成darknet53
    # YOLOv3 网络的基础特征提取器。该方法接收输入张量inputs和当前卷积层索引conv_index，并返回三个具有不同特征图大小的张量 route1、route2 和最终的 conv
    def _darknet53(self, inputs, conv_index, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        with tf.variable_scope('darknet53'):
            conv = self._conv2d_layer(inputs, filters_num=32, kernel_size=3, strides=1, name='conv2d'+str(conv_index))
            conv = self._batch_normalization_layer(conv, name='batch_normalization_'+str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=64, blocks_num=1, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=128, blocks_num=2, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=256, blocks_num=8, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            route1 = conv
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=512, blocks_num=8, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            route2 = conv
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=1024, blocks_num=4, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            return route1, route2, conv, conv_index

    # yolo_block 用于构建 YOLO 的特定网络块
    def _yolo_block(self, inputs, filters_num, out_filters, conv_index, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        conv = self._conv2d_layer(inputs, filters_num=filters_num, kernel_size=1, strides=1, name='conv2d'+str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization'+str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num*2, kernel_size=3, strides=1, name='conv2d'+str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization'+str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1, name='conv2d'+str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization'+str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num*2, kernel_size=3, strides=1, name='conv2d'+str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization'+str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1, name='conv2d'+str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization'+str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        route = conv
        conv = self._conv2d_layer(conv, filters_num=filters_num*2, kernel_size=3, strides=1, name='conv2d'+str(conv_index))
        conv = self._batch_normalization_layer(conv, name='batch_normalization'+str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=out_filters, kernel_size=1, strides=1, name='conv2d'+str(conv_index), use_bias=True)
        conv_index += 1

        # route保存最后一个1*1卷积层的输出，用于与其他特征图进行拼接或跳跃连接
        # conv保存最后卷积层的输出
        return route, conv, conv_index


    # 实现了YOLO网络的推理过程，包括特征提取、特征层的构建、上采样和连接等步骤。它使用了Darknet53作为骨干网络，并在其基础上构建了YOLO的检测头
    def yolo_inference(self, inputs, num_anchors, num_classes, training=True):

        # 调用darknet53方法提取特征，返回多个卷积层的输出
        conv_index = 1
        conv2d_26, conv2d_43, conv, conv_index = self._darknet53(inputs, conv_index, training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)

        # yolo网络主结构
        with tf.variable_scope('yolo'):

            # yolo_block 获得特征层conv2d_57、conv2d_59
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, 512, num_anchors*(num_classes+5), conv_index=conv_index, training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            # 特征图的上采样和连接到中尺度特征图
            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num=256, kernel_size=1, strides=1, name='conv2d'+str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name='batch_normalization'+str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            conv_index += 1
            # 特征图上采样
            unsample_0 = tf.image.resize_nearest_neighbor(conv2d_60, [2*tf.shape(conv2d_60)[1], 2*tf.shape(conv2d_60)[1]], name='unsample_0')  # unsample_0 = 26, 26, 256
            # 特征图与中尺度特征图连接
            route0 = tf.concat([unsample_0, conv2d_43], axis=-1, name='route_0')  # route0 = 26, 26, 768
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, num_anchors*(num_classes+5), conv_index=conv_index, training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)  # conv2d_65 = 52, 52, 256, conv2d_67 = 26, 26, 255

            # 获得第二个特征层
            conv2d_68 = self._conv2d_layer(conv2d_65, filters_num=128, kernel_size=1, strides=1, name='conv2d'+str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name='batch_normalization'+str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            conv_index += 1
            # 特征图上采样
            unsample_1 = tf.image.resize_nearest_neighbor(conv2d_68, [2*tf.shape(conv2d_68)[1], 2*tf.shape(conv2d_68)[1]], name='unsample_1')
            # 特征图与大尺度特征层连接
            route1 = tf.concat([unsample_1, conv2d_26], axis=-1, name='route_1')
            _, conv2d_75, _ = self._yolo_block(route1, 128, num_anchors*(num_classes+5), conv_index=conv_index, training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)

        # 返回三个特征图 conv2d_59、conv2d_67 和 conv2d_75
        return [conv2d_59, conv2d_67, conv2d_75]