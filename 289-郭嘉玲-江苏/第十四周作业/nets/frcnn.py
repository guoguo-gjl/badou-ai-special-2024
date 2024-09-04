# 用于目标检测模型的构建函数

from nets.resnet import ResNet50, classifier_layers
from nets.RoiPooling import RoiPoolingConv
from keras.layers import Conv2D, Input, TimeDistributed, Flatten, Dense, Reshape
from keras.models import Model


# get_rpn函数。定义rpn网络，用于生成候选区域
# 举例：
    # 假设我们有一张输入图像的特征图经过网络处理后得到的 base_layers 形状是 (24, 24, 1024)。RPN 将会进行如下处理：
    # 卷积层1 生成 (24, 24, 512) 的特征图。
    # 类别预测卷积层 生成 (24, 24, 9)，表示每个位置上有 9 个锚框，每个锚框的分类概率。
    # 回归预测卷积层 生成 (24, 24, 36)，表示每个位置上有 9 个锚框，每个锚框的回归值（x, y, w, h）。
    # 最终，x_class 形状 (5184, 1) 和 x_regr 形状 (5184, 4) 将被返回，用于后续的训练和预测。
def get_rpn(base_layers, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    # x_class 和 x_regr 层：分别用于生成每个位置的类别预测和回归预测
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    # Reshape：将输出调整为 RPN 所需的形状
    x_class = Reshape((-1, 1), name="classification")(x_class)
    x_regr = Reshape((-1, 4), name="regression")(x_regr)
    return [x_class, x_regr, base_layers]


# get_classifier函数。定义了用于候选区域的分类器。它处理输入的候选区域并输出类别和回归结果
def get_classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]


# get_model函数。
# 创建了三个模型：
    # model_rpn：仅包含 RPN 部分；
    # model_classifier：仅包含分类器部分；
    # model_all：包含 RPN 和分类器两个部分
def get_model(config, num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    base_layers = ResNet50(inputs)

    # 计算anchor boxes 的数量
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn[:2])

    # 输出每个ROI的分类结果和边界框调整值
    classifier = get_classifier(base_layers, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier = Model([inputs, roi_input], classifier)

    # 组合rpn前两个输出和分类结果，用于整体目标检测任务
    model_all = Model([inputs, roi_input], rpn[:2]+classifier)
    return model_rpn, model_classifier, model_all


# get_predict_model函数。创建两个模型，用于实际的预测阶段
    # model_rpn：用于生成候选区域。
    # model_classifier_only：用于对候选区域进行分类和回归
def get_predict_model(config, num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None,None,1024))

    base_layers = ResNet50(inputs)
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers, num_anchors)  # 基于特征图和anchor boxes数量构建rpn网络，并输出分类分数和回归值。
    model_rpn = Model(inputs, rpn)

    classifier = get_classifier(feature_map_input, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    return model_rpn, model_classifier_only