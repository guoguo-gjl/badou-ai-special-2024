import numpy as np
import cv2

# 定义一个双线性插值函数，img是输入图像，out_dim是输出的高度和宽度
def bilinear_interpolation(img,out_dim):

    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h, src_w =", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)

    # 输出图像和输入图像高宽相等，返回输入图像
    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    # 定义一个高宽与输出图像一致的空像素图像
    # np.zeros()是numpy库中创建指定形状的全零数组的函数
        # 语法：np.zeros((形状)，数据类型，存储顺序）
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)

    # 求图像宽高的缩放比例
    scale_x, scale_y = float(src_w)/ dst_w, float(src_h)/ dst_h

    # 计算目标图像像素在原始图像中的对应位置，根据在原始坐标的位置找到对应的像素值
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                # 中心对齐
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                # 左上对齐
                # src_x = dst_x * scale_x
                # src_y = dst_y * scale_y

                # 为了不超边界的防越界检查
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 计算插值
                # 本质：求虚拟点的像素值，temp0、1是在求什么？img是像素值吗？
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return  dst_img

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow("lenna", img)
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()