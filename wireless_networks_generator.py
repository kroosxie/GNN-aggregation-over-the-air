import numpy as np
import helper_functions
def generate_layouts(general_para, number_of_layouts):
    N = general_para.n_links  # D2D直连链路数目，即N个tx及N个rx
    print("<<<<<<<<<<<<<{} layouts: {}>>>>>>>>>>>>".format(
        number_of_layouts, general_para.setting_str))
    layouts = []
    dists = []
    for i in range(number_of_layouts):  # layouts：产生训练集个数的layout
        layout, dist = helper_functions.layout_generate(general_para)  # 返回tx、rx的位置坐标以及所有链路的距离
        layouts.append(layout)  # 将tx、rx的位置坐标堆叠到layouts中
        dists.append(dist)  # 将所有链路的距离堆叠到dists中
    layouts = np.array(layouts)  # 创建数组对象，便于后面reshape等操作
    dists = np.array(dists)
    # 断点语句：assert 是一种用来检测调试代码问题的语句，当条件为 True 时会直接通过，当为 False 时会抛出错误，可以用来定位和修改代码。
    assert np.shape(layouts)==(number_of_layouts, N, 4)
    assert np.shape(dists)==(number_of_layouts, N, N)
    return layouts, dists  # 返回训练集个数的tx、rx分布以及所有链路的距离信息（相当于生成训练集个数的地图）

def compute_path_losses(general_para, distances):
    N = np.shape(distances)[-1]  # 直连链路的数目
    assert N==general_para.n_links  # 断点语句：结果为true直接通过
    h1 = general_para.tx_height  # tx高度
    h2 = general_para.rx_height  # rx高度
    signal_lambda = 2.998e8 / general_para.carrier_f  # wavelength
    antenna_gain_decibel = general_para.antenna_gain_decibel  # 直连链路天线增益 decibel：dB
    # compute relevant quantity
    Rbp = 4 * h1 * h2 / signal_lambda  # threshold for a two-way path-loss model, LOS wave and reflected wave bp is breakpoint
    # LOS波和反射波双向路径损耗模型的阈值，bp是断点距离
    Lbp = abs(20 * np.log10(np.power(signal_lambda, 2) / (8 * np.pi * h1 * h2)))  # 基本断点距离
    # compute coefficient matrix for each Tx/Rx pair 计算每一个收发对的信道系数矩阵
    sum_term = 20 * np.log10(distances / Rbp)
    Tx_over_Rx = Lbp + 6 + sum_term + ((distances > Rbp).astype(int)) * sum_term  # adjust for longer path loss 注：dB形式
    pathlosses = -Tx_over_Rx + np.eye(N) * antenna_gain_decibel  # only add antenna gain for direct channel 仅在直连链路中加入天线增益 np.eye是对角元素
    pathlosses = np.power(10, (pathlosses / 10))  # convert from decibel to absolute 将dB形式转化为绝对值
    return pathlosses  # 返回所有链路的路径损耗的绝对值