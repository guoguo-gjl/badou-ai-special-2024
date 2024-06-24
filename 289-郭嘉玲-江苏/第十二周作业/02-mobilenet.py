import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K

# 构建深度可分离卷积块 depthwise_separable_convolution
def _depthwise_conv_block(inputs,
                          pointwise_conv_filters,
                          depth_multipiler=1,
                          strides=(1, 1),
                          block_id=1):
    # depthwise卷积
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multipiler=depth_multipiler,  # 输出通道数是输入通道数的倍数
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d'%block_id)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)
    # pointwise卷积
    x = Conv2D(pointwise_conv_filters,
               (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

# 普通卷积块
def _conv_block(inputs, filters, kernel=(3, 3), strides = (1, 1)):
    x = Conv2D(filters,
               kernel,
               padding='same',
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)

# 定义整个模型结构
def MobileNet(input_shape = (224, 224, 3),
              depth_multiplier=1,
              dropout=1e-3,
              classes=1000):

    img_input = Input(shape=input_shape)
    x = _conv_block(img_input, 32, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)
    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=3)
    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)
    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)

    for i in range(5):
        x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7+i)

    x = _depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    inputs = img_input

    model = Model(inputs, x, name='mobilenet_1_0_224_tf')
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)

    return model

def preprocess_input(x):
    x /= 255
    x -= 0.5
    x *= 2.
    return x

# 限制输出值最大为6
def relu6(x):
    return K.relu(x, max_value=6)

if __name__ == '__main__':
    model = MobileNet(input_shape=[224, 224, 3])
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)  # 将图像数据转为numpy数组形式
    x = np.expand_dims(x, axis=0)  # 在数组x的第0维增加了一个维度，增加维度的默认值为1
    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds, 1))