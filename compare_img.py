import cv2

def img_error(img1, img2):
    # 计算两个图片之间的差值
    error = ((img1 - img2) ** 2).sum() / img1.size
    return error

# 梯度下降法函数，输入一个大图，一个小图，初始值(列表，[x0, y0])
# 可以寻找在初始值附近是否有相同图像
def gradient_descent(big_img, small_img, inital_position, Binari):

    # 判断是否是相同的标志
    are_same = False
    # 初始点在输入的初值
    [x_0, y_0] = inital_position
    x_n, y_n = x_0, y_0
    # 设置范围,超出范围直接飞出
    x_Range, y_Range = 50, 50
    # 初始化离散化求导的步长
    dx, dy = 1, 1
    # 获取小图的大小，光翼展开
    Lx, Ly = int(small_img.shape[1]/2), int(small_img.shape[0]/2)

    # 搜索深度，一般成功的话几次就行了
    depth = 20

    # 视情况而定是否二值化，因为结算界面用三维比较明显
    # 初始搜索步长
    eta = 10 ** 4
    eta_n = eta
    # 初始化误差
    error_bar = 0.02

    # 将大图进行二值化
    big_img = cv2.cvtColor(big_img, cv2.COLOR_BGR2GRAY)
    ret, big_img = cv2.threshold(big_img, Binari, 255, cv2.THRESH_BINARY)

    # 小图进行二值化
    small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    ret, small_img = cv2.threshold(small_img, Binari, 255, cv2.THRESH_BINARY)

    # 迭代循环，采用梯度下降算法
    for _ in range(depth):
        # 提取现在位置的图片信息
        img = big_img[y_n - Ly : y_n + Ly, x_n - Lx : x_n + Lx]

        # 提取x方向周围图片的信息
        img_pdx = big_img[y_n - Ly : y_n + Ly, x_n + dx - Lx : x_n + dx + Lx]
        img_ndx = big_img[y_n - Ly : y_n + Ly, x_n - dx - Lx : x_n - dx + Lx]

        # 提取y方向周围图片的信息
        img_pdy = big_img[y_n + dy - Ly : y_n + dy + Ly, x_n - Lx : x_n + Lx]
        img_ndy = big_img[y_n - dy - Ly : y_n - dy + Ly, x_n - Lx : x_n + Lx]

        # 搜索步长随着迭代变小，较好的办法就是用误差来计算
        current_error = img_error(img, small_img)
        eta_n = eta * current_error

        # 进行迭代，注意要取整，此处采取中心差分计算偏导数
        x_n = x_n - int(eta_n * (img_error(img_pdx, small_img) - img_error(img_ndx, small_img))/(2 * dx))
        y_n = y_n - int(eta_n * (img_error(img_pdy, small_img) - img_error(img_ndy, small_img))/(2 * dy))

        # 如果小于阈值，跳出并输出两图片相同
        if current_error < error_bar:
            are_same = True
            break

        #如果超出范围，立马退出
        if abs(x_n - x_0) > x_Range or abs(y_n - y_0) > y_Range:
            break
    return are_same
