import cv2
import numpy as np
import time
import os

# 均值哈希
# 像素灰度值与图像灰度平均值比较
def aHash(img, width=8, high=8):
    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    avg = s/64
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

# 差值感知算法
# 前一个像素灰度值与后一个像素灰度值比较
def dHash(img, width=9, high=9):
    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(high):
        for j in range(width):
            if gray[i, j] > gray[i, j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

# 哈希值对比
# 计算哈希值相似度
def cmp_hash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n+1
    return 1 - n/len(hash2)

# 感知哈希算法
def pHash(img_file, width=64, high=64):
    img = cv2.imread(img_file, 0)
    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)
    # 将调整后的灰度图像数据填充到vis0
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img
    # 二维dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    vis1.resize(32, 32)
    # 把二维list变为一维list
    img_list = vis1.flatten()
    # 计算均值
    avg = sum(img_list) * 1. / len(img_list)
    # 根据均值生成哈希
    avg_list = ['0' if i > avg else '1' for i in img_list]
    # 每组4位二进制转换为对应的16进制字符
    # 二进制转为整数：int(''.join(avg_list[x:x+4]), 2)
    # 整数转为十六字符：'%x' % integer_value
    # 将转换后的十六进制字符连接成一个完整的哈希值字符串：''.join(...)
    hash_value = ''.join(['%x' % int(''.join(avg_list[x:x+4]), 2) for x in range(0, 32*32, 4)])
    return hash_value

def hamming_dist(s1, s2):
    # zip(s1, s2)将字符串s1,s2对应位置字符配对起来
    # [ch1 != ch2 for ch1, ch2 in zip(s1, s2)]生成一个布尔值列表，表示对应位置上s1、s2对应位置上字符是否相同
    # sum([...])对布尔值列表求和
    # 1 - sum([…]) * 1. / (32*32/4)计算相似度得分
    return 1-sum([ch1 != ch2 for (ch1, ch2) in zip(s1, s2)]) * 1. / (32*32/4)

def concat_info(type_str, score, time):
    temp = '%s相似度：%.2f %% -----time=%.4f ms' % (type_str, score*100, time)
    print(temp)
    return temp

def test_diff_hash(img1_path, img2_path, loops=1000):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    # 使用time.time()函数获取当前时间，将其存储在start_time变量中
    start_time = time.time()
    for _ in range(loops):
        hash1 = dHash(img1)
        hash2 = dHash(img2)
        cmp_hash(hash1, hash2)
    print(">>> 执行%s次耗费的时间是%.4f s."%(loops, time.time()-start_time))

def test_aHash(img1, img2):
    time1 = time.time()
    hash1 = aHash(img1)
    hash2 = aHash(img2)
    n = cmp_hash(hash1, hash2)
    return concat_info("均值哈希算法", n, time.time() - time1) + "\n"

def test_dHash(img1, img2):
    time1 = time.time()
    hash1 = dHash(img1)
    hash2 = dHash(img2)
    n = cmp_hash(hash1, hash2)
    return concat_info("插值哈希算法", n, time.time() - time1) + "\n"

def test_pHash(img1_path, img2_path):
    time1 = time.time()
    hash1 = pHash(img1_path)
    hash2 = pHash(img2_path)
    n = hamming_dist(hash1, hash2)
    return concat_info("感知哈希算法", n, time.time() - time1) + "\n"

def deal(img1_path, img2_path):
    info = ''
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    info = info + test_aHash(img1, img2)
    info = info + test_dHash(img1, img2)
    info = info + test_pHash(img1_path, img2_path)
    return info

def contact_path(file_name):
    output_path = r"C:\Users\guogu\PycharmProjects\我的作业\第八周\01-ransac"
    return os.path.join(output_path, file_name)

def main():
    data_img_name = 'lenna.png'
    data_img_name_base = data_img_name.split(".")[0]
    base = contact_path(data_img_name)
    light = contact_path("%s_light.jpg" % data_img_name_base)
    resize = contact_path("%s_resize.jpg" % data_img_name_base)
    contrast = contact_path("%s_contrast.jpg" % data_img_name_base)
    sharp = contact_path("%s_sharp.jpg" % data_img_name_base)
    blur = contact_path("%s_blur.jpg" % data_img_name_base)
    color = contact_path("%s_color.jpg" % data_img_name_base)
    rotate = contact_path("%s_rotate.jpg" % data_img_name_base)
    # 测试算法的效率
    test_diff_hash(base, base)
    test_diff_hash(base, light)
    test_diff_hash(base, resize)
    test_diff_hash(base, contrast)
    test_diff_hash(base, sharp)
    test_diff_hash(base, blur)
    test_diff_hash(base, color)
    test_diff_hash(base, rotate)
    # 测试算法精度
    deal(base, light)

if __name__ == '__main__':
    main()