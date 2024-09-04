# YOLO进行图像检测

import os
from utils import config
import numpy as np
import tensorflow as tf
from yolo_predict import yolo_predictor
from PIL import Image, ImageFont, ImageDraw
from yolo_tools import letterbox_image, load_weights

# 环境配置
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_index

# detect函数
def detect(image_path, model_path, yolo_weights = None):

    # 图片预处理
    image = Image.open(image_path)
    resize_image = letterbox_image(image, (416, 416))
    image_data = np.array(resize_image, dtype=np.float32)
    image_data /= 255.
    image_data = np.expand_dims(image_data, axis=0)  # 为图像数据增加一个维度

    # 输入占位符和预测
    input_image_shape = tf.placeholder(dtype=tf.int32, shape=(2, ))
    input_image = tf.placeholder(shape=[None, 416, 416, 3], dtype=tf.float32)

    # 创建yolo预测对象，进行物体检测测
    predcitor = yolo_predictor(config.obj_threshold, config.nms_threshold, config.classes_path, config.anchors_path)
    # 处理权重的加载
    with tf.Session() as sess:
        if yolo_weights is not None:
            with tf.variable_scope('predict'):
                boxes, scores, classes = predcitor.predict(input_image, input_image_shape)  # 获取模型预测结果
            load_op = load_weights(tf.global_variables(scope='predict'), weight_file = yolo_weights)
            sess.run(load_op)
            out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes],
            feed_dict={input_image: image_data, input_image_shape: [image.size[1], image.size[0]]})
        else:
            boxes, scores, classes = predcitor.predict(input_image, input_image_shape)
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes],
            feed_dict={input_image: image_data, input_image_shape: [image.size[1], image.size[0]]})

    # 结果显示
    print('Found{} boxes for {}'.format(len(out_boxes), 'img'))
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    # 遍历检测到的框
    for i, c in reversed(list(enumerate(out_classes))):
        predcitor_class = predcitor.class_names[c]
        box = out_boxes[i]
        scores = out_scores[i]
        label = '{}{:.2f}'.format(predcitor_class, scores)
        draw = ImageDraw.Draw(image)
        # 计算标签大小
        label_size = draw.textsize(label, font)
        # 计算框的边界
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1] - 1, np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0] - 1, np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))
        # 绘制矩形框
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top-label_size[1]])
        else:
            text_origin = np.array([left, top+1])
        for i in range(thickness):
            draw.rectangle([left+i, top+i, right-i, bottom-i], outline=predcitor.colors[c])
        # 绘制标签背景和文本
        draw.rectangle([tuple(text_origin), tuple(text_origin+label_size)], fill = predcitor.colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)

    # 保存和展示图像
    image.show()
    image.save('./img/result1.jpg')


if __name__ == '__main__':
    if config.pre_train_yolo3 == True:
        detect(config.image_file, config.model_dir, config.yolo3_weights_path)
    else:
        detect(config.image_file, config.model_dir)

