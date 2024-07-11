import VGG16_net
import tensorflow as tf
import VGG16_utils

img1 = VGG16_utils.load_image('C:\\Users\\guogu\\PycharmProjects\\第十一周\\代码\\VGG16-tensorflow-master\\test_data\\dog.jpg')
inputs = tf.placeholder(tf.float32, [None, None, 3])  # 传递的图片数据为浮点数（tf.float32），大小为任意尺寸的宽度、高度和3个颜色通道。
resized_img = VGG16_utils.resize_image(inputs, (224, 224))
# 建立模型网络结构
prediction = VGG16_net.vgg16_16(resized_img)
# 载入模型
sess = tf.Session()
ckpt_filename = './vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()  # 创建用于恢复模型的saver对象
saver.restore(sess, ckpt_filename)  # 从检查点文件恢复模型的参数到当前会话
# 进行softmax预测
pro = tf.nn.softmax(prediction)
pre = sess.run(pro, feed_dict={inputs: img1})
# 打印预测结果
VGG16_utils.print_prob(pre[0], 'C:\\Users\\guogu\\PycharmProjects\\第十一周\\代码\\VGG16-tensorflow-master\\test_data\\synset.txt')
