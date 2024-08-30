import os
import random
import colorsys
import numpy as np
import tensorflow as tf
from yolo_model import yolo


class yolo_predictor:

    def __init__(self, obj_threshold, nms_threshold, classes_file, anchors_file):
        # 保存传入的目标检测置信度阈值和非极大值抑制阈值
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        # 保存类别文件和锚框文件的路径
        self.classes_path = classes_file
        self.anchors_path = anchors_file
        # 读取类别名称
        self.class_names = self._get_class()
        # 读取锚框信息
        self.anchors = self._get_anchors()
        # 生成HSV颜色，将HSV转为RGB
        hsv_tuples = [(x/len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), self.colors))
        # 打乱颜色顺序
        random.seed(10101)
        random.shuffle(self.colors)
        random.seed(None)

    # 读取类别名称
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        classes_names = [c.strip() for c in class_names]
        return class_names

    # 读取锚框数据
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readlines()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    # 从YOLO模型最后一层输出提取预测框的坐标、宽高、置信度和类别概率
    def _get_feats(self, feats, anchors, num_classes, input_shape):
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        # 重新调整形状
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
        # 创建网格坐标
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, tf.float32)
        # 计算边界框位置、置信度、类别概率
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])

        return box_xy, box_wh, box_confidence, box_class_probs

    # 获取边界框和边界框得分
    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        # 获得特征
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        # 纠正和归一化坐标
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])
        # 计算置信得分
        box_scores = box_confidence*box_class_probs  # 计算锚点的置信得分，锚框的置信度和类别概率的乘积
        box_scores = tf.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores

    # 将预测框从网络输入尺寸（通常是缩放过的图像）转换到原始图像的坐标系统
    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        # 物体框的坐标和宽高的反向顺序（从 (x, y, w, h) 转换为 (y, x, h, w)）
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = tf.cast(input_shape, dtype=tf.float32)
        image_shape = tf.cast(image_shape, dtype=tf.float32)
        # 计算新的形状。tf.reduce_min(input_shape / image_shape) 是保持长宽比的缩放因子。
        new_shape = tf.round(image_shape*tf.reduce_min(input_shape/image_shape))
        # 计算缩放因子和偏移量，用于将框从输入图像的坐标系统转换到实际图像的坐标系统。
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale
        # 计算物体框的最小值和最大值
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx - (box_hw / 2.)
        # 拼接box_mins,box_maxes
        boxes = tf.concat([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis=-1)
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        return boxes

    # eval方法。根据YOLO模型输出执行非极大值抑制，得到最终的检测框、得分和类别
    def eval(self, yolo_outputs, image_shape, max_boxes=20):
        # 定义每个特征层对应的锚点掩码，并初始化空的框和分数列表
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        input_shape = tf.shape(yolo_outputs[0])[1:3]*32
        # 遍历每个特征层，通过 boxes_and_scores 函数计算边界框和分数，并将结果存储到列表中
        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]], len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)
        # 创建一个布尔掩码来筛选得分高于 obj_threshold 的框，并初始化结果列表
        mask = box_scores >= self.obj_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []
        # 对每个类别进行非极大值抑制，筛选出最终的边界框和得分，并生成类别标签
        for c in range(len(self.class_names)):
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_boxes_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(class_boxes, class_boxes_scores, max_boxes_tensor, iou_threshold=self.nms_threshold)
            class_boxes = tf.gather(class_boxes, nms_index)
            class_boxes_scores = tf.gather(class_boxes_scores, nms_index)
            classes = tf.ones_like(class_boxes_scores, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_boxes_scores)
            classes_.append(classes)
        # 将结果合并并返回
        boxes_ = tf.concat(boxes_, axis=0)
        scores_ = tf.concat(scores_, axis=0)
        classes_ = tf.concat(classes_, axis=0)
        return boxes_, scores_, classes_

    # predict方法。执行yolo模型的预测，返回检测框、得分和类别
    def predict(self, inputs, image_shape):
        model = yolo(config.norm_epsilon, config.norm_decay, self.anchors_path, self.classes_path, pre_train=False)
        output = model.yolo_inference(inputs, config.num_anchors//3, config.num_classes, training=False)
        boxes, scores, classes = self.eval(output, image_shape, max_boxes=30)

        return boxes, scores, classes