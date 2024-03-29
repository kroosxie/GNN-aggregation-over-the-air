from os import times
import scipy.io as sio                     
import numpy as np                         
import matplotlib.pyplot as plt           
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
import wireless_networks_generator as wg
import helper_functions
import time

class init_parameters():
    def __init__(self):
        # wireless network settings
        self.n_links = train_K  # 这里是D2D直连链路数目
        self.field_length = 500
        self.shortest_directLink_length = 2
        self.longest_directLink_length = 40
        self.shortest_crossLink_length = 1  # crosslink即干扰链路，设置最小干扰链路距离防止产生过大的干扰
        self.bandwidth = 5e6
        self.carrier_f = 2.4e9
        self.tx_height = 1.5
        self.rx_height = 1.5
        self.antenna_gain_decibel = 2.5
        self.tx_power_milli_decibel = 40
        self.tx_power = np.power(10, (self.tx_power_milli_decibel-30)/10)
        self.noise_density_milli_decibel = -169
        self.input_noise_power = np.power(10, ((self.noise_density_milli_decibel-30)/10)) * self.bandwidth
        self.output_noise_power = self.input_noise_power
        self.setting_str = "{}_links_{}X{}_{}_{}_length".format(self.n_links, self.field_length, self.field_length, self.shortest_directLink_length, self.longest_directLink_length)

def normalize_data(train_data,test_data):  # 将数据标准化
    #normalize train directlink
    mask = np.eye(train_K)  # 生成对角1矩阵
    train_copy = np.copy(train_data)  # 生成副本
    diag_H = np.multiply(mask,train_copy)  # 用对角矩阵提取直连链路信道系数
    diag_mean = np.sum(diag_H)/train_layouts/train_K /frame_num  # 计算直连链路H均值
    diag_var = np.sqrt(np.sum(np.square(diag_H))/train_layouts/train_K/frame_num)  # 计算标准差
    tmp_diag = (diag_H - diag_mean)/diag_var  # 标准化为正态分布
    #normalize train interference link
    off_diag = train_copy - diag_H
    off_diag_mean = np.sum(off_diag)/train_layouts/train_K/(train_K-1)/frame_num
    off_diag_var = np.sqrt(np.sum(np.square(off_diag))/train_layouts/train_K/(train_K-1)/frame_num)
    tmp_off = (off_diag - off_diag_mean)/off_diag_var
    tmp_off_diag = tmp_off - np.multiply(tmp_off,mask)
    
    norm_train = np.multiply(tmp_diag,mask) + tmp_off_diag  # 规范化后的训练集
    
    # normlize test
    mask = np.eye(test_K)
    test_copy = np.copy(test_data)
    diag_H = np.multiply(mask,test_copy)
    tmp_diag = (diag_H - diag_mean)/diag_var
    
    off_diag = test_copy - diag_H
    tmp_off = (off_diag - off_diag_mean)/off_diag_var
    tmp_off_diag = tmp_off - np.multiply(tmp_off,mask)
    
    norm_test = np.multiply(tmp_diag,mask) + tmp_off_diag  # 规范后的测试集
    
    return norm_train, norm_test

def build_graph(loss, norm_loss, K):  # 构建图
    x1 = np.expand_dims(np.diag(norm_loss),axis=1)  # 在列维度上扩充一个维度，方便用于后续转化为张量。np.expand_dims用于在数组的指定轴上扩展维度。
    x2 = np.zeros((K,graph_embedding_size))
    x = np.concatenate((x1,x2),axis=1)  # 在列维度上合并
    x = torch.tensor(x, dtype=torch.float)  # 将数组转化为张量，方便深度学习自动求导等操作
    
    #conisder fully connected graph 构建全连接图
    loss2 = np.copy(loss)  # 路径损耗副本
    mask = np.eye(K)
    diag_loss2 = np.multiply(mask,loss2)  # 提取D2D直连链路路径损耗
    loss2 = loss2 - diag_loss2  # 提取干扰链路
    attr_ind = np.nonzero(loss2)  # 用于获取数组中非零元素的索引，输出是一个包含两个数组的元组，分别对应行和列的非零元素索引。
    edge_attr = norm_loss[attr_ind]  # 提取非零干扰链路的标准化值
    edge_attr = np.expand_dims(edge_attr, axis = -1)  # 在最后一个维度上（列维度）扩充一个维度
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # 边特征，将数组转化为张量，方便深度学习自动求导等操作。
    
    attr_ind = np.array(attr_ind)  # 转化为numpy数组，
    adj = np.zeros(attr_ind.shape)  # 构建邻接矩阵
    adj[0,:] = attr_ind[1,:]
    adj[1,:] = attr_ind[0,:]
    edge_index = torch.tensor(adj, dtype=torch.long)  # 邻接矩阵

    y = torch.tensor(np.expand_dims(loss,axis=0), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.contiguous(), edge_attr = edge_attr, y=y)  # 使用PYG构件带有节点及边特征值的图。其中.contiguous()可以用来确保张量的内存布局是连续的
    return data

def proc_data(HH, norm_HH, K):
    n = HH.shape[0]  # 即layouts的数目
    data_list = []
    for i in range(n):
        data = build_graph(HH[i,:,:], norm_HH[i,:,:], K)
        data_list.append(data)
    return data_list

class GConv(MessagePassing):  # 图卷积模块 见文章（6-8）式
    def __init__(self, mlp1, mlp2, **kwargs):
        super(GConv, self).__init__(aggr='max', **kwargs)
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        
    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        comb = self.mlp2(tmp)
        return torch.cat([x[:,:1], comb],dim=1)
        
    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1,self.mlp2)

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])

class MPNN(torch.nn.Module):
    def __init__(self):
        super(MPNN, self).__init__()

        self.mlp1 = MLP([2+graph_embedding_size, 32, 32])
        self.mlp2 = MLP([33+graph_embedding_size, 16, graph_embedding_size])
        self.conv = GConv(self.mlp1,self.mlp2)
        self.h2o = MLP([graph_embedding_size, 16])
        self.h2o = Seq(*[self.h2o,Seq(Lin(16, 1, bias = True), Sigmoid())])

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x = x0, edge_index = edge_index, edge_attr = edge_attr)
        x2 = self.conv(x = x1, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        output = self.h2o(out[:,1:])
        return output
        
# xjc：这里是全局的损失函数
def sr_loss(data, out, K):
    power = out
    power = torch.reshape(power, (-1, K, 1))    
    abs_H_2 = data.y
    abs_H_2 = abs_H_2.permute(0,2,1)
    rx_power = torch.mul(abs_H_2, power)
    mask = torch.eye(K)
    mask = mask.to(device)
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)
    interference = torch.sum(torch.mul(rx_power, 1 - mask), 1) + var
    rate = torch.log2(1 + torch.div(valid_rx_power, interference))*overhead_ratio
    sr = torch.mean(torch.sum(rate, 1))
    loss = torch.neg(sr)
    return loss

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = sr_loss(data,out,train_K)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / train_layouts / frame_num

def test():
    model.eval()  # 将模型设置为评估模式：即固定参数
    total_loss = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            loss = sr_loss(data,out,test_K)
            total_loss += loss.item() * data.num_graphs
    return total_loss / test_layouts / frame_num


train_K = 20  # 20个D2D对用于训练
test_K = 20  # 20个D2D对用于测试
train_layouts = 2000    # 2000个不同的训练集
test_layouts = 500    # 500个不同的测试集
frame_num = 10  # 10帧
test_config = init_parameters()
train_config = init_parameters()
var = train_config.output_noise_power / train_config.tx_power
frame_length = 3000  # 一个frame里面有3000个symbol
graph_embedding_size = 8  #
overhead_csi = 1
overhead_mp = 5

print('Data generation')
#Data generation
#Train data
layouts, train_dists = wg.generate_layouts(train_config, train_layouts)  # 创建训练集个数的tx、rx分布以及所有链路的距离信息（相当于生成训练集个数的地图）
train_path_losses = wg.compute_path_losses(train_config, train_dists)  # 计算所有链路的路径损耗的绝对值，这里的loss是path_loss，不是loss_function
train_channel_losses = helper_functions.add_fast_fading_sequence(frame_num, train_path_losses)  # 在每一个帧加入快衰落
#Treat multiple frames as multiple samples for MPNN 在MPNN中将多个帧视为多个采样
train_channel_losses = train_channel_losses.reshape(train_layouts*frame_num,train_K,train_K)  # 信道系数作为训练集
#Test data 
layouts, test_dists = wg.generate_layouts(test_config, test_layouts)
test_path_losses = wg.compute_path_losses(test_config, test_dists)
test_channel_losses = helper_functions.add_fast_fading_sequence(frame_num,test_path_losses)
#Treat multiple frames as multiple samples for MPNN
test_channel_losses = test_channel_losses.reshape(test_layouts*frame_num,test_K,test_K)

#Data normalization 数据规范化
norm_train_losses, norm_test_losses = normalize_data(np.sqrt(train_channel_losses),np.sqrt(test_channel_losses) )
print('Graph data processing')
#Graph data processing 图数据处理
train_data_list = proc_data(train_channel_losses, norm_train_losses, train_K)  # 使用PYG生成带节点及边特征的图
test_data_list = proc_data(test_channel_losses, norm_test_losses, test_K)


directLink_channel_losses = helper_functions.get_directLink_channel_losses(test_channel_losses)
crossLink_channel_losses = helper_functions.get_crossLink_channel_losses(test_channel_losses)
mpnn_overhead_ratio = (frame_length-overhead_csi*train_K*train_K-overhead_mp*(train_K)*3)/frame_length  # *3？
wmmse_overhead_ratio = (frame_length-overhead_csi*train_K*train_K)/frame_length
print('WMMSE and EPA computation')
#sum rate of wmmse
Pini = np.random.rand(test_layouts*frame_num,test_K,1 )
Y2 = helper_functions.batch_WMMSE(Pini,np.ones([test_layouts*frame_num, test_K]),np.sqrt(test_channel_losses),1,var)
rates_wmmse = helper_functions.compute_rates(test_config, 
            Y2, directLink_channel_losses, crossLink_channel_losses)
sum_rate_wmmse = np.mean(np.sum(rates_wmmse,axis=1))*wmmse_overhead_ratio   # *wmmse_overhead_ratio以计算有效通信的速率
print('WMMSE average sum rate:',sum_rate_wmmse)

#sum rate of epa 全1功率发射
Pepa = np.ones((test_layouts*frame_num,test_K))
rates_epa = helper_functions.compute_rates(test_config, 
            Pepa, directLink_channel_losses, crossLink_channel_losses)
sum_rate_epa = np.mean(np.sum(rates_epa,axis=1))
print('EPA average sum rate:',sum_rate_epa)

#train of MPNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MPNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002) # Adam优化器优化
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)  # 学习率调整
train_loader = DataLoader(train_data_list, batch_size=50, shuffle=True,num_workers=0)
test_loader = DataLoader(test_data_list, batch_size=50, shuffle=False, num_workers=0)

#Total 2000X10=20000 samples, each epoch with 20000/50 = 400 iterations
for epoch in range(1, 6):
    overhead_ratio = mpnn_overhead_ratio
    loss1 = train()
    loss2 = test()
    print('Epoch {:03d}, Train Loss: {:.4f}, Test Loss: {:.4f},'.format(
            epoch, loss1, loss2))   
    scheduler.step()

#Test for scalability and various system parameters, an example
gen_tests = [10, 15, 20, 25, 30]
overhead_csi = 2
overhead_mp = 20
frame_length = 3000
frame_num = 10
density = train_config.field_length**2/train_K
for test_K in gen_tests:
    print('<<<<<<<<<<<<<< Num of Links is {:03d} >>>>>>>>>>>>>:'.format(test_K))
    # generate test data
    test_config.n_links = test_K
    field_length = int(np.sqrt(density*test_K))
    test_config.field_length = field_length
    layouts, test_dists = wg.generate_layouts(test_config, test_layouts)
    test_path_losses = wg.compute_path_losses(test_config, test_dists)
    test_channel_losses = helper_functions.add_fast_fading_sequence(frame_num,test_path_losses)
    #Treat multiple frames as multiple samples for MPNN
    test_channel_losses = test_channel_losses.reshape(test_layouts*frame_num,test_K,test_K)

    mpnn_overhead_ratio = (frame_length-overhead_csi*test_K*test_K-overhead_mp*(test_K)*3)/frame_length
    wmmse_overhead_ratio = (frame_length-overhead_csi*test_K*test_K)/frame_length
    mpnn_overhead_ratio = max(mpnn_overhead_ratio,0)
    wmmse_overhead_ratio = max(wmmse_overhead_ratio,0)
    directLink_channel_losses = helper_functions.get_directLink_channel_losses(test_channel_losses)
    crossLink_channel_losses = helper_functions.get_crossLink_channel_losses(test_channel_losses)

    #test for wmmse
    Pini = np.random.rand(test_layouts*frame_num,test_K,1 )
    Y2 = helper_functions.batch_WMMSE(Pini,np.ones([test_layouts*frame_num, test_K]),np.sqrt(test_channel_losses),1,var)
    rates_wmmse = helper_functions.compute_rates(test_config, 
            Y2, directLink_channel_losses, crossLink_channel_losses)
    sum_rate_wmmse = np.mean(np.sum(rates_wmmse,axis=1))*wmmse_overhead_ratio
    print('WMMSE average sum rate:',sum_rate_wmmse)

    #test for epa
    Pepa = np.ones((test_layouts*frame_num,test_K))
    rates_epa = helper_functions.compute_rates(test_config, 
            Pepa, directLink_channel_losses, crossLink_channel_losses)
    sum_rate_epa = np.mean(np.sum(rates_epa,axis=1))
    print('EPA average sum rate:',sum_rate_epa)

    #test for mpnn
    norm_train_losses, norm_test_losses = normalize_data(np.sqrt(train_channel_losses),np.sqrt(test_channel_losses) )
    test_data_list = proc_data(test_channel_losses, norm_test_losses, test_K)
    test_loader = DataLoader(test_data_list, batch_size=50, shuffle=False, num_workers=0)
    overhead_ratio = mpnn_overhead_ratio
    loss2 = test()
    print('MPNN average sum rate:',-loss2)


