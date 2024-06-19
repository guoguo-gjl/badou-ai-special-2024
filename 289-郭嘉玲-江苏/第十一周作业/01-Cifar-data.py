# cifar数据处理

import os
import tensorflow as tf
num_classes = 10
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_test = 10000

# 定义空类，用于返回读取的cifar-10数据
class Cifar10Record(object):
    pass
# 定义读取cifar-10的函数，目标是读取目标文件的内容
def read_cifar_10(file_queue):
    result = Cifar10Record()
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.depth * result.width
    record_bytes = label_bytes + image_bytes

    # 创建一个读取类，目的是读取文件
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # 使用该类的read()函数从文件队列里读取文件，读出来key和value
    # 【读取文件的方法是？文件的数据格式是什么样的？key和value怎么与文件数据对应的？】
    result.key, value = reader.read(file_queue)

    # 将字符串形式解析成uint8形式
    record_bytes = tf.decode_raw(value, tf.uint8)
    # 用strided_slice()函数把标签提取出来，用cast()函数将标签转成int32格式
    # strided_slice()函数，第一个参数是输入数据，从起始位置[0]即第一个元素开始取，取到终止位置[label_bytes]，不包括[label_bytes]位置的元素
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes], tf.int32))

    # 将一维数据转换成三维数据
    # 同上
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.depth, result.height, result.width])

    # 将分割好的图片数据使用transpose()函数将depth_major的（D,H,W)转换为(H,W,D)的顺序
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result

# 对数据进行预处理
def inputs(data_dir,  batch_size, distorted):
   # 创建一个包含了5个文件名的列表，这些文件名分别为 'data_batch_1.bin' 到 'data_batch_5.bin'，它们位于 data_dir 目录下
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' %i) for i in range(1, 6)]
    # 根据已有文件地址创建文件队列
    file_queue = tf.train.string_input_producer(filenames)
    # 根据已有的文件队列使用定义好的文件读取
    read_input = read_cifar_10(file_queue)
    # 将转换好的图片再次转为float32的形式
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    num_examples_per_epoch = num_examples_pre_epoch_for_train

    # 如果预处理中distorted参数不为空，就要对图片进行增强处理
    # 【distorted 参数什么情况下会参数为空？】
    if distorted != None:
        # 剪切
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])
        # 左右翻转
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        # 亮度调整
        adjusted_contrast = tf.image.random_brightness(flipped_image, max_delta=0.8)
        # 图片标准化操作
        float_image = tf.image.per_image_standardization(adjusted_contrast)
        # 设置图片数据及标签形状
        float_image.set_shape([24, 24, 3])
        # 【这一行代码不理解】
        read_input.label.set_shape([1])
        min_queue_examples = int(num_examples_pre_epoch_for_test * 0.4)
        print('filling queue with %d cifar images before starting to train.this will take a few minuts'% min_queue_examples)
        # 【这一行代码的参数对应的什么意思？】
        # 使用tf.train.shuffle_batch()函数随机产生一个batch的image和label。
        # 【为什么要随机产生一个batch的image和label
        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label],batch_size=batch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3*batch_size,
                                                            min_after_dequeue=min_queue_examples,)
    else:
        # 用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
        # 剪切完之后进行图片标准化操作
        float_image = tf.image.per_image_standardization(resized_image)
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])
        min_queue_examples = int(num_examples_per_epoch * 0.4)
        images_test, label_test = tf.train.batch([float_image, read_input.label],
                                                 batch_size=batch_size,
                                                 num_threads=16,
                                                 capacity=min_queue_examples + 3 * batch_size)
        return images_test, tf.reshape(label_test, [batch_size])
