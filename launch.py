import cv2
import os
import shutil
import json
import numpy as np
from tqdm import tqdm
import compare_img

def are_tiaoguo(big_img):
    # 跳过图标中心位置
    tiaoguo_position = [172, 963]
    # 加载'跳过'图片
    tiaoguo_img = cv2.imread(os.getcwd() + '\\images\\process\\tiaoguo.jpg')
    # 将跳过图片在目标位置附近进行搜索，启动二值化
    Bin = 100
    flag = compare_img.gradient_descent(big_img, tiaoguo_img, tiaoguo_position, Bin)
    # 返回是否有跳过图标
    return flag

# 对跳过位置照一个快照，目的是对比与上一帧是否相同
def snapshot(big_img):
    # 跳过图标中心位置
    shot = [172, 963]
    # 跳过图标的展宽
    Lx, Ly = 72, 33
    # 照快照
    image = big_img[shot[1] - Ly : shot[1] + Ly, shot[0] - Lx : shot[0] + Lx]
    # 返回快照
    return image

# 判断这一帧的图片是否与上一帧相同
def are_same_pic(current_img, last_img):
    same_pic = False
    shot = [172, 963]
    Bin = 100
    same_pic = compare_img.gradient_descent(current_img, last_img, shot, Bin)
    return same_pic

# 统计刺客是否枪人，枪的第几个
def cike_shot(big_img):
    shot = False
    # 导入刺客枪人的位置
    px = [155, 550, 945, 1340]
    py = [285, 466, 648, 829]
    # 导入枪口图片
    cike = cv2.imread(os.getcwd() + '\\images\\process\\cike.jpg')
    # j表示第几名幸运鸭子被枪
    j = 0
    lucky_duck = 0
    # 对每一只鸭子进行遍历
    for y in py:
        for x in px:
            Bin = 20
            position = [x, y]
            # 将鸭子位置和枪口进行对比
            shotted = compare_img.gradient_descent(big_img, cike, position, Bin)
            if shotted:
                shot = True
                lucky_duck = j
            # 对准下一位玩家的鸭头
            j += 1
    return shot, lucky_duck

# 数死鸭子头
def count_death(big_img):
    # 导入死鸭子头的位置
    px = [290, 685, 1080, 1475]
    py = [285, 466, 648, 829]
    # 准备好记录的列表
    count_dead = []
    # 导入死鸭子头
    dead = cv2.imread(os.getcwd() + '\\images\\process\\dead.jpg')
    # j表示第几名幸运鸭子被杀
    j = 1
    # 对每一只鸭子进行遍历
    for y in py:
        for x in px:
            Bin = 177
            position = [x, y]
            # 将鸭子位置和死亡图像进行对比
            dead_duck = compare_img.gradient_descent(big_img, dead, position, Bin)
            if dead_duck:
                count_dead.append(j)
            # 对准下一位玩家的鸭头
            j += 1
    return count_dead

# 检查中间是否有胜利，即结算界面
def are_settlement(image):
    # 首先来到结算界面路径找出所有胜利图片
    chara_path = os.getcwd() + '\\images\\chara\\'
    files = [i for i in os.listdir(chara_path)]
    # 胜利图标的位置
    win_position = [950, 400]
    # 切割上半部分
    left1, left2 = 200, 1710
    up1, up2 = 190, 245

    # 切割下半部分
    down1, down2= 580, 800
    # 二值化指标
    Bin = 140

    # 将胜利信息与所有图片进行比较
    for file in files:
        chara = cv2.imread(chara_path + file)
        win_chara = compare_img.gradient_descent(image, chara, win_position, Bin)
        if win_chara == True:
            img_up = image[up1:up2, left1:left2]
            img_down = image[down1:down2, left1:left2]
            img_con = np.concatenate((img_up, img_down))
            # 返回是否是结算界面，什么阵营赢了，和拼接的照片
            name = list(file)
            del name[len(name)-4:len(name)]
            name = ''.join(name)
            return True, name, img_con

    # 没找到就返回空
    return False, None, None

# 进入每一局游戏的循环判断
# count_rounds 输入这是第几轮，与刺客枪人有关
def one_play(image, record, cike_record, last_img):

    # 如果左下角有跳过，那么是报警开饭界面，进入判断
    if are_tiaoguo(image):
        # 判断这一帧与上一帧是否相同，如果相同要进行刺客枪人判断
        if are_same_pic(image, last_img):
            shotted, num = cike_shot(image)
            # 如果检测到开枪打人了，且这一枪的记录与上一个不同，计入总数，num + 1是自然计数
            cike_information = [len(record), num + 1]
            if shotted and cike_information not in cike_record:
                cike_record.append(cike_information)
        # 如果不相同，那就是第一次进入报警
        # 这个时候统计死去的鸭子们
        else:
            dead_duck = count_death(image)
            record.append(dead_duck) 
    # 没有跳过，那么检查是否是结算界面
    else:
        are_settle, chara, name_img = are_settlement(image)
        if are_settle:
            # 如果是结算界面，那么返回总体的记录，刺客的记录
            # 第三个是表示'是'结算界面，给出胜利阵营，并且给出姓名图片
            return record, cike_record, True, chara, name_img
    # 如果不是，就返回总体的记录和刺客记录
    return record, cike_record, False, None, None 

# 主函数，整个流程都在这
def main(video_filename):
     # 导入视频
    camera = cv2.VideoCapture(video_filename)

    # 帧率
    fps = camera.get(cv2.CAP_PROP_FPS)

    # 视频总帧数
    total_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

    # 六秒钟取一帧，可以调节读取秒数
    frameFrequency = 6 * int(fps)

    # 获取第一帧图片
    ret, first_image = camera.read()

    # 第一个图片初始化
    last_img = snapshot(first_image)

    # 初始化刺客和一局的情况
    record = []
    cike_record = []

    # 记录视频中的所有信息
    video_information = {}
    play_num = 0
    last_fps = 0
    last_record_fps = 0

    # 开始循环
    for i in tqdm(range(total_frames - 1)):
        # 获取当前视频帧位置
        now_fps = camera.get(1)

        # 如果到了读取帧，就进行读取
        if (now_fps % frameFrequency == 0):
            # 读取这一帧的图像
            ret, image = camera.read()

            # 进入图像解析，一局游戏之间的判断
            record_length = len(record)
            record, cike_record, play_end, chara, name_img = one_play(image, record, cike_record, last_img)
            now_length = len(record)
            # 对record进行判断,如果数目增加了，就要计入总帧数
            if now_length > record_length:
                player_position = [188, 894, 90, 1640]
                player_array = image[player_position[0]:player_position[1],player_position[2]:player_position[3]]
                if now_length >= 2 and record[-1] == record[-2] and abs(last_record_fps - now_fps)/fps < 100:
                    del record[-1]
                last_record_fps = now_fps
            
            # 如果游戏结束，并且与上一个结束之间的距离大于10s
            # 就判断真的结束了，将这一局的保存，并且清空计数
            if play_end and abs(last_fps - now_fps)/fps > 10:
                # 局数加1
                play_num += 1
                # 如果游戏结束，那么增加新的信息
                new_information = {
                    str(play_num):{
                        'record' : record,
                        'cike_record' : cike_record,
                        'winner' : chara,
                        'name_img' : name_img,
                        'player_array' : player_array
                    }
                }
                video_information.update(new_information)

                # 清空记录
                record = []
                cike_record = []

                # 记录现在的帧，假设两次结束之间的距离超过10s
                last_fps = now_fps

            # 在相同位置照快照，不论怎么样都要照快照
            last_img = snapshot(image)
        else:
            # 如果不是，就进行跳帧
            ret = camera.grab()

    return video_information


if __name__ == '__main__':
    video_filename = 'template.mp4'
    video_information = main(video_filename)

    information_path = os.getcwd() + '\\video_information\\'
    try:
        os.makedirs(information_path)
    except FileExistsError:
        shutil.rmtree(information_path)
        os.makedirs(information_path)

    real_information = {}

    for rounds in video_information:
        the_play = video_information[rounds]
        real_information.update({
            rounds:{
            'record' : the_play['record'],
            'cike_record' : the_play['cike_record'],
            'winner' : the_play['winner']
            }
        })
        cv2.imwrite(information_path + rounds + 'name.jpg', the_play['name_img'])
        cv2.imwrite(information_path + rounds + 'array.jpg', the_play['player_array'])

    output_name = information_path + 'result.json'
    with open(output_name, 'w') as f_new:
        json.dump(real_information, f_new, ensure_ascii=False, indent=2, sort_keys =True)