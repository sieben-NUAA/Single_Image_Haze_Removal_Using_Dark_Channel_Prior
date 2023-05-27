import numpy as np
import cv2
import time


def get_rgb_min(img):
    """
    获取RGB三通道的最小\n
    :param img: 输入图像
    :return: 最小值的图像
    """
    min_rgb = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            min_pixel = min(img[i, j, 0], img[i, j, 1], img[i, j, 2])
            min_rgb[i, j] = min_pixel
    return min_rgb


def get_filter_min(img, x1, x2, y1, y2):
    """
    获取filter内的最小值\n
    :param img: 图像
    :param x1: (x1, x2)
    :param x2: (x1, x2)
    :param y1: (y1, y2)
    :param y2: (y1, y2)
    :return: 最小值
    """
    min_l = 255
    for i in range(x1, x2 + 1):
        for j in range(y1, y2 + 1):
            if img.item(i, j) < min_l:
                min_l = img.item(i, j)
    return min_l


def DarkChannel(img, patch_size=3):
    """
    获取暗通道图像，为防止图像变小且边缘信息丢失，采用填充方法（padding）\n
    :param img: 输入的图像
    :param patch_size: 计算p*p大小矩阵中的最小值，或者说是filter size
    :return: 暗通道图像
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        print("Image error!")
        return None

    img_min = get_rgb_min(img)

    if patch_size < 3 or patch_size % 2 == 0:
        print("In get_Dark_Channel, parameter \"patch_size\" error!\n")
        return None

    padding_size = int((patch_size - 1) / 2)

    x = img_min.shape[0]
    y = img_min.shape[1]

    x_new = int(x + patch_size - 1)
    y_new = int(y + patch_size - 1)

    padding_img = np.zeros((x_new, y_new))
    padding_img[:, :] = 255
    padding_img[padding_size: x_new - padding_size, padding_size: y_new - padding_size] = img_min

    Dark_Channel = np.zeros((x, y), dtype=np.float64)
    for i in range(padding_size, x_new - padding_size):
        for j in range(padding_size, y_new - padding_size):
            Dark_Channel[i - padding_size, j - padding_size] = get_filter_min(padding_img,
                                                                              i - padding_size,
                                                                              i + padding_size,
                                                                              j - padding_size,
                                                                              j + padding_size)

    return Dark_Channel


def Atmospheric_Light(dark_channel, img, percentage=0.001):
    """
    估算全局大气光\n
    :param dark_channel: 暗通道图像
    :param img: 原图
    :param percentage: 按百分比获得最亮像素
    :return: 全局大气光
    """
    light_pixel = []
    height = dark_channel.shape[0]
    width = dark_channel.shape[1]

    for i in range(height):
        for j in range(width):
            light_pixel.append((dark_channel.item(i, j), i, j))

    light_pixel.sort(reverse=True)
    nums = int(height * width * percentage)

    total = 0
    for i in range(0, nums):
        for j in range(0, 3):
            total = total + img[light_pixel[i][1], light_pixel[i][2], j]

    atomsphericLight = total / (nums * 3)

    return atomsphericLight


def Transmission(AtL, img, w=0.95, patch_size=3):
    """
    获取透射率\n
    :param AtL: 大气透射率
    :param img: 图像暗通道
    :param w: 去雾比例
    :param patch_size: 窗口大小
    :return: 透射率矩阵
    """
    # darkChannel = np.float64(darkChannel)
    transmission = 1 - w * DarkChannel(img / AtL, patch_size)
    return transmission


def GuidedFilter(img, guide_img, radius=(20, 20), eps=0.0001):
    """
    通过查询文献，发现soft matting方法费时，使用何凯明大佬的另一个导向滤波算法\n
    :param img: 原始图像
    :param guide_img: 导向图，这里选择使用原图的灰度图
    :param radius: 滤波核大小
    :param eps: 惩罚项系数
    :return: 滤波后图像
    """

    # 获取均值
    gui_mean = cv2.blur(guide_img, radius)
    img_mean = cv2.blur(img, radius)
    gui2_mean = cv2.blur(guide_img * guide_img, radius)
    img_gui_mean = cv2.blur(guide_img * img, radius)

    # 计算方差与协方差
    gui_var = gui2_mean - gui_mean * gui_mean
    img_gui_cov = img_gui_mean - img_mean * gui_mean
    a = img_gui_cov / (gui_var + eps)
    b = img_mean - a * gui_mean

    mean_a = cv2.blur(a, radius)
    mean_b = cv2.blur(b, radius)

    return mean_a * guide_img + mean_b


def guided_filter(p, i, r, e):
    """
    :param p: input image
    :param i: guidance image
    :param r: radius
    :param e: regularization
    :return: filtering output q
    """
    # 1
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
    # 2
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    # 3
    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I
    # 4
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    # 5
    q = mean_a * i + mean_b
    return q


def ImageHazingRemoval(path, t0=0.3):
    img_input = cv2.imread(path)
    img = img_input.astype('float64') / 255
    img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY).astype('float64') / 255

    dark = DarkChannel(img, 13)
    cv2.imwrite("./img/dark_channel_lights.png", dark * 255)

    AL = Atmospheric_Light(dark, img)
    # AL = min(AL, 220/255)
    print("Atmospheric_Light: ", AL)
    trans = Transmission(AL, img, patch_size=13)
    # grey guide filter
    trans_guided = guided_filter(trans, img_gray, 20, 0.001)
    t = cv2.max(trans_guided, t0)
    p = "./img/trans_lights.png"
    cv2.imwrite(p, t * 255)

    Res = np.empty_like(img)
    for c in range(0, 3):
        Res[:, :, c] = (img[:, :, c] - AL) / t + AL
    name = "./img/res_lights.png"
    # name = "./img/res_12-constrain.png"
    cv2.imwrite(name, Res * 255)

    """
    trans_no_filter = cv2.max(trans, t0)
    cv2.imwrite("./img/trans_my-down_without_filter.png", trans_no_filter * 255)
    Resx = np.empty_like(img)
    for c in range(0, 3):
        Resx[:, :, c] = (img[:, :, c] - AL) / trans_no_filter + AL
    cv2.imwrite("./img/res_my-down_without_filter.png", Resx * 255)
    # trans_guided = GuidedFilter(trans, img_gray)

    # RGB Guide filter
    start = time.time()
    image1 = cv2.split(img_input)[0]#蓝通道
    image2 = cv2.split(img_input)[1]
    image3 = cv2.split(img_input)[2]
    image1 = image1 / 255.0
    image2 = image2 / 255.0
    image3 = image3 / 255.0
    gf1 = guided_filter(trans, image1, 17, 0.01)
    gf2 = guided_filter(trans, image2, 17, 0.01)
    gf3 = guided_filter(trans, image3, 17, 0.01)
    trans_RGB = (gf1 + gf2 + gf3) / 3
    trans_RGB = cv2.max(trans_RGB, t0)
    cv2.imwrite("./img/trans_my-down_RGB.png", trans_RGB * 255)
    end = time.time()
    print('RGB Guide filter执行时间 = {} min {} s'.format(float((end - start) / 60), float((end - start) % 60)))
    Res1 = np.empty_like(img)
    for c in range(0, 3):
        Res1[:, :, c] = (img[:, :, c] - AL) / trans_RGB + AL
    cv2.imwrite("./img/res_my-down_RGB.png", Res1 * 255)
    """


if __name__ == "__main__":
    path = "./origin-img/lights.jpg"
    ImageHazingRemoval(path)
