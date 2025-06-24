import argparse
import scipy.sparse as sp
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
import networkx as nx
import multiprocessing as mp
import torch.nn.functional as F
from functools import partial
import random
from sklearn.metrics import roc_auc_score, f1_score
from copy import deepcopy
from scipy.spatial.distance import pdist,squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--size', type=int, default=100)

    parser.add_argument('--epochs', type=int, default=2010,
                help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--batch_nums', type=int, default=6000, help='number of batches per epoch')
    parser.add_argument('--batch_size', type=int, default=40, help='number of batches per epoch')


    parser.add_argument('--imbalance', action='store_true', default=False)
    parser.add_argument('--setting', type=str, default='no', 
        choices=['no','upsampling', 'smote','reweight','embed_up', 'recon','newG_cls','recon_newG'])
    #upsampling: oversample in the raw input; smote: ; reweight: reweight minority classes; 
    # embed_up: 
    # recon: pretrain; newG_cls: pretrained decoder; recon_newG: also finetune the decoder

    parser.add_argument('--opt_new_G', action='store_true', default=False) # whether optimize the decoded graph based on classification result.
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--up_scale', type=float, default=1.0)
    parser.add_argument('--im_ratio', type=float, default=0.5)
    parser.add_argument('--rec_weight', type=float, default=0.000001)
    parser.add_argument('--model', type=str, default='sage', 
        choices=['sage','gcn','GAT'])
    parser.add_argument('--n_clusters', type=int, default=3, help='n_clusters.')
    parser.add_argument('--k_neighbors', type=int, default=3, help='Number of neighbors for k_neighbors.')
    parser.add_argument('--max_k', type=int, default=5, help='max_k.')
    parser.add_argument('--num_episodes', type=int, default=4, help='num_episodes.')
    parser.add_argument('--reduce_classes', type=int, default=1, help='reduce_classes.')

    return parser

def split_arti(labels, c_train_num):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    #cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    c_num_mat[:,1] = 25
    c_num_mat[:,2] = 55

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        train_idx = train_idx + c_idx[:c_train_num[i]]
        c_num_mat[i,0] = c_train_num[i]

        val_idx = val_idx + c_idx[c_train_num[i]:c_train_num[i]+25]
        test_idx = test_idx + c_idx[c_train_num[i]+25:c_train_num[i]+80]

    random.shuffle(train_idx)

    #ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    #c_num_mat = torch.LongTensor(c_num_mat)


    return train_idx, val_idx, test_idx, c_num_mat

def split_genuine(labels):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    #cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num <4:
            if c_num < 3:
                print("too small class type")
                ipdb.set_trace()
            c_num_mat[i,0] = 1
            c_num_mat[i,1] = 1
            c_num_mat[i,2] = 1
        else:
            c_num_mat[i,0] = int(c_num/4)
            c_num_mat[i,1] = int(c_num/4)
            c_num_mat[i,2] = int(c_num/2)


        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)

    #ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    #c_num_mat = torch.LongTensor(c_num_mat)


    return train_idx, val_idx, test_idx, c_num_mat

def split_genuine(labels, reduce_classes=0):
    """
    改进的数据集划分函数，支持类别过滤
    
    参数:
        labels: n-dim Longtensor, 每个元素在[0,...,m-1]范围内
        reduce_classes: 要去掉的类别数量（从最后开始去掉）
    """
    # 获取原始类别信息
    unique_classes = torch.unique(labels).tolist()
    num_classes = len(unique_classes)
    
    # 确定要保留的类别
    if reduce_classes > 0:
        keep_classes = sorted(unique_classes)[:-reduce_classes]  # 保留前面的类别
        mask = torch.isin(labels, torch.tensor(keep_classes))
        labels = labels[mask]
        print(f"已去掉最后{reduce_classes}个类别，保留{len(keep_classes)}个类别")
    
    # 重新计算类别信息
    num_classes = len(torch.unique(labels))
    c_idxs = []  # 按类别存储索引
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes, 3)).astype(int)

    for i in range(num_classes):
        c_idx = (labels == i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        print(f"{i}-th class sample number: {len(c_idx)}")
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        # 根据样本数量确定划分比例
        if c_num < 4:
            if c_num < 3:
                print(f"警告: 类别{i}样本数太少({c_num})，全部分配到训练集")
                c_num_mat[i,0] = c_num
                c_num_mat[i,1] = 0
                c_num_mat[i,2] = 0
            else:
                c_num_mat[i,0] = 1
                c_num_mat[i,1] = 1
                c_num_mat[i,2] = c_num - 2
        else:
            c_num_mat[i,0] = int(c_num * 0.25)  # 25%训练
            c_num_mat[i,1] = int(c_num * 0.25)  # 25%验证
            c_num_mat[i,2] = c_num - c_num_mat[i,0] - c_num_mat[i,1]  # 剩余测试

        # 分配索引
        train_idx += c_idx[:c_num_mat[i,0]]
        val_idx += c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx += c_idx[c_num_mat[i,0]+c_num_mat[i,1]:]

    # 打乱顺序
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)

    return torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx), c_num_mat

def print_edges_num(dense_adj, labels):
    c_num = labels.max().item()+1
    dense_adj = np.array(dense_adj)
    labels = np.array(labels)

    for i in range(c_num):
        for j in range(c_num):
            #ipdb.set_trace()
            row_ind = labels == i
            col_ind = labels == j

            edge_num = dense_adj[row_ind].transpose()[col_ind].sum()
            print("edges between class {:d} and class {:d}: {:f}".format(i,j,edge_num))


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# def print_class_acc(output, labels, class_num_list, pre='valid'):
#     pre_num = 0
#     #print class-wise performance
#     '''
#     for i in range(labels.max()+1):
        
#         cur_tpr = accuracy(output[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
#         print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

#         index_negative = labels != i
#         labels_negative = labels.new(labels.shape).fill_(i)
        
#         cur_fpr = accuracy(output[index_negative,:], labels_negative[index_negative])
#         print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))

#         pre_num = pre_num + class_num_list[i]
#     '''

#     #ipdb.set_trace()
#     if labels.max() > 1:
#         auc_score = roc_auc_score(labels.detach(), F.softmax(output, dim=-1).detach(), average='macro', multi_class='ovr')
#     else:
#         auc_score = roc_auc_score(labels.detach(), F.softmax(output, dim=-1)[:,1].detach(), average='macro')

#     macro_F = f1_score(labels.detach(), torch.argmax(output, dim=-1).detach(), average='macro')
#     print(str(pre)+' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score,macro_F))

#     return


def print_class_acc(output, labels, class_num_list, pre='valid'):
    pre_num = 0

    # 将输出和标签移到CPU上
    output_cpu = output.detach().cpu()
    labels_cpu = labels.detach().cpu()

    # 计算AUC-ROC分数
    if labels.max() > 1:
        auc_score = roc_auc_score(labels_cpu.numpy(), F.softmax(output_cpu, dim=-1).numpy(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels_cpu.numpy(), F.softmax(output_cpu, dim=-1)[:,1].numpy(), average='macro')

    # 计算F1分数
    macro_F = f1_score(labels_cpu.numpy(), torch.argmax(output_cpu, dim=-1).numpy(), average='macro')
    
    print(f"{pre} current auc-roc score: {auc_score:.6f}, current macro_F score: {macro_F:.6f}")
    return

def src_upsample(adj,features,labels,idx_train, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    adj_back = adj.to_dense()
    chosen = None

    #ipdb.set_trace()
    avg_number = int(idx_train.shape[0]/(c_largest+1))

    for i in range(im_class_num):
        new_chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        if portion == 0:#refers to even distribution
            c_portion = int(avg_number/new_chosen.shape[0])

            for j in range(c_portion):
                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)

        else:
            c_portion = int(portion)
            portion_rest = portion-c_portion
            for j in range(c_portion):
                num = int(new_chosen.shape[0])
                new_chosen = new_chosen[:num]

                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)
            
            num = int(new_chosen.shape[0]*portion_rest)
            new_chosen = new_chosen[:num]

            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
            

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0]+add_num, adj_back.shape[0]+add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:,:]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen,:]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:,chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen,:][:,chosen]

    #ipdb.set_trace()
    features_append = deepcopy(features[chosen,:])
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0]+add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features,features_append), 0)
    labels = torch.cat((labels,labels_append), 0)
    idx_train = torch.cat((idx_train,idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train



def src_smote(adj,features,labels,idx_train, portion=1.0, im_class_num=3):
    c_largest = labels.max().item() #c_largest:6
    adj_back = adj.to_dense()
    chosen = None
    new_features = None

    #ipdb.set_trace()
    avg_number = int(idx_train.shape[0]/(c_largest+1))#avg_number:15

    for i in range(im_class_num):
        new_chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        if portion == 0:#refers to even distribution
            c_portion = int(avg_number/new_chosen.shape[0])

            portion_rest = (avg_number/new_chosen.shape[0]) - c_portion

        else:
            c_portion = int(portion)
            portion_rest = portion-c_portion
            
        for j in range(c_portion):
            num = int(new_chosen.shape[0])
            new_chosen = new_chosen[:num]

            chosen_embed = features[new_chosen,:]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance,distance.max()+100)

            idx_neighbor = distance.argmin(axis=-1)
            
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor,:]-chosen_embed)*interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed),0)
            
        num = int(new_chosen.shape[0]*portion_rest)
        new_chosen = new_chosen[:num]

        chosen_embed = features[new_chosen,:]
        distance = squareform(pdist(chosen_embed.cpu().detach()))
        np.fill_diagonal(distance,distance.max()+100)

        idx_neighbor = distance.argmin(axis=-1)
            
        interp_place = random.random()
        embed = chosen_embed + (chosen_embed[idx_neighbor,:]-chosen_embed)*interp_place

        if chosen is None:
            chosen = new_chosen
            new_features = embed
        else:
            chosen = torch.cat((chosen, new_chosen), 0)
            new_features = torch.cat((new_features, embed),0)
            

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0]+add_num, adj_back.shape[0]+add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:,:]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen,:]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:,chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen,:][:,chosen]

    #ipdb.set_trace()
    features_append = deepcopy(new_features)
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0]+add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features,features_append), 0)
    labels = torch.cat((labels,labels_append), 0)
    idx_train = torch.cat((idx_train,idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train


# smote过采样
def recon_upsample(embed, labels, idx_train, adj=None, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0]/(c_largest+1))
    #ipdb.set_trace()
    adj_new = None

    for i in range(im_class_num):
        chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        num = int(chosen.shape[0]*portion)
        if portion == 0:
            c_portion = int(avg_number/chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen,:]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance,distance.max()+100)

            idx_neighbor = distance.argmin(axis=-1)

            interp_place = random.random()
            new_embed = embed[chosen,:] + (chosen_embed[idx_neighbor,:]-embed[chosen,:])*interp_place


            new_labels = labels.new(torch.Size((chosen.shape[0],1))).reshape(-1).fill_(c_largest-i)
            idx_new = np.arange(embed.shape[0], embed.shape[0]+chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed,new_embed), 0)
            labels = torch.cat((labels,new_labels), 0)
            idx_train = torch.cat((idx_train,idx_train_append), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(torch.clamp_(adj[chosen,:] + adj[idx_neighbor,:], min=0.0, max = 1.0))
                else:
                    temp = adj.new(torch.clamp_(adj[chosen,:] + adj[idx_neighbor,:], min=0.0, max = 1.0))
                    adj_new = torch.cat((adj_new, temp), 0)

    if adj is not None:
        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[0]+add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:,:]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:,:]

        return embed, labels, idx_train, new_adj.detach()

    else:
        return embed, labels, idx_train

# kmeans聚类+smote随机过采样
# def recon_upsample_kmeans_smote(embed, labels, idx_train, adj=None, portion=1.0, im_class_num=3, n_clusters=3):
    # print("---------KMeans + SMOTE随机过采样---------")
    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0]/(c_largest+1))
    adj_new = None

    for i in range(im_class_num):
        chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        num = int(chosen.shape[0]*portion)
        if portion == 0:
            c_portion = int(avg_number/chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        chosen_embed = embed[chosen,:]
        cluster_labels = kmeans.fit_predict(chosen_embed.cpu().detach().numpy())
        
        for cluster in range(n_clusters):
            cluster_idx = chosen[cluster_labels == cluster]
            cluster_embed = embed[cluster_idx, :]
            
            # SMOTE within each cluster
            for j in range(c_portion):
                cluster_idx = cluster_idx[:num]
                
                distance = squareform(pdist(cluster_embed.cpu().detach()))
                np.fill_diagonal(distance, distance.max() + 100)

                idx_neighbor = distance.argmin(axis=-1)

                interp_place = random.random()
                new_embed = embed[cluster_idx, :] + (cluster_embed[idx_neighbor, :] - embed[cluster_idx, :]) * interp_place

                new_labels = labels.new(torch.Size((cluster_idx.shape[0], 1))).reshape(-1).fill_(c_largest-i)
                idx_new = np.arange(embed.shape[0], embed.shape[0]+cluster_idx.shape[0])
                idx_train_append = idx_train.new(idx_new)

                embed = torch.cat((embed, new_embed), 0)
                labels = torch.cat((labels, new_labels), 0)
                idx_train = torch.cat((idx_train, idx_train_append), 0)

                if adj is not None:
                    if adj_new is None:
                        adj_new = adj.new(torch.clamp_(adj[cluster_idx,:] + adj[idx_neighbor,:], min=0.0, max=1.0))
                    else:
                        temp = adj.new(torch.clamp_(adj[cluster_idx,:] + adj[idx_neighbor,:], min=0.0, max=1.0))
                        adj_new = torch.cat((adj_new, temp), 0)

    if adj is not None:
        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[0]+add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:,:]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:,:]

        return embed, labels, idx_train, new_adj.detach()

    else:
        return embed, labels, idx_train


def recon_upsample_kmeans_smote1(embed, labels, idx_train, adj=None, portion=1.0, im_class_num=3, n_clusters=3):
    # 确定新的多数类样本数量 N1_new
    c_largest = labels.max().item()
    N2 = idx_train[(labels==(c_largest-im_class_num+1))[idx_train]].shape[0]
    N1_new = int(portion * N2)
    
    adj_new = None

    for i in range(im_class_num):
        chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        N1_largest = chosen.shape[0]

        # 确定每个簇中抽取的样本量
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        chosen_embed = embed[chosen,:]
        cluster_labels = kmeans.fit_predict(chosen_embed.cpu().detach().numpy())
        
        for cluster in range(n_clusters):
            cluster_idx = chosen[cluster_labels == cluster]
            N_l1 = cluster_idx.shape[0]
            
            # 计算当前簇中抽取的样本数量 ni
            n_i = int((N_l1 / N1_largest) * N1_new)

            # 取出当前簇中的嵌入向量
            cluster_embed = embed[cluster_idx, :]

            # SMOTE within each cluster
            for j in range(n_i):
                cluster_idx_sampled = cluster_idx[:n_i]
                
                distance = squareform(pdist(cluster_embed.cpu().detach()))
                np.fill_diagonal(distance, distance.max() + 100)

                idx_neighbor = distance.argmin(axis=-1)

                interp_place = random.random()
                new_embed = embed[cluster_idx_sampled, :] + (embed[idx_neighbor, :] - embed[cluster_idx_sampled, :]) * interp_place

                new_labels = labels.new(torch.Size((cluster_idx_sampled.shape[0], 1))).reshape(-1).fill_(c_largest-i)
                idx_new = np.arange(embed.shape[0], embed.shape[0]+cluster_idx_sampled.shape[0])
                idx_train_append = idx_train.new(idx_new)

                embed = torch.cat((embed, new_embed), 0)
                labels = torch.cat((labels, new_labels), 0)
                idx_train = torch.cat((idx_train, idx_train_append), 0)

                if adj is not None:
                    if adj_new is None:
                        adj_new = adj.new(torch.clamp_(adj[cluster_idx_sampled,:] + adj[idx_neighbor,:], min=0.0, max=1.0))
                    else:
                        temp = adj.new(torch.clamp_(adj[cluster_idx_sampled,:] + adj[idx_neighbor,:], min=0.0, max=1.0))
                        adj_new = torch.cat((adj_new, temp), 0)

    if adj is not None:
        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[0]+add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:,:]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:,:]

        return embed, labels, idx_train, new_adj.detach()

    else:
      return embed, labels, idx_train

def cnn_denoising(embed, labels, idx_train, im_class_num):
    
    core_samples = idx_train[labels[idx_train] > (labels.max() - im_class_num)]
    core_samples_set = set(core_samples.tolist())
    maj_samples = idx_train[labels[idx_train] <= (labels.max() - im_class_num)]
    maj_samples_set = set(maj_samples.tolist())
    for i in maj_samples_set:
        neighbors = find_nearest_neighbors(embed[i], embed[idx_train], k=5)
        neighbors = [n for n in neighbors if n != i]
        if any(labels[n] > (labels.max() - im_class_num) for n in neighbors):
            core_samples_set.add(i)
    remaining_maj_samples = {i for i in maj_samples_set if i not in core_samples_set}
    for i in remaining_maj_samples:
        neighbors = find_nearest_neighbors(embed[i], embed[idx_train], k=5)
        neighbors = [n for n in neighbors if n != i]
        if all(n in core_samples_set for n in neighbors):
            core_samples_set.add(i)
    remaining_samples = {i for i in idx_train.tolist() if i not in core_samples_set}
    for i in list(remaining_samples):
        neighbors = find_nearest_neighbors(embed[i], embed[idx_train], k=5)
        neighbor_labels = [labels[n] for n in neighbors]
        from collections import Counter
        label_counts = Counter(neighbor_labels)
        my_label = labels[i]
        if all(label!= my_label for label in neighbor_labels):
            remaining_samples.remove(i)
    core_samples_set.update(remaining_samples)

    idx_train=torch.tensor(list(core_samples_set), dtype=torch.long)
    return embed, labels, idx_train

def find_nearest_neighbors(sample, all_samples, k=3):
    distances = torch.norm(all_samples - sample, dim=1)
    return distances.topk(k, largest=False).indices.tolist()




# def recon_upsample_kmeans_smote(embed, labels, idx_train, adj=None, portion=1.0, im_class_num=3, n_clusters=3, k_neighbors=3):
#     embed, labels, idx_train = cnn_denoising(embed, labels, idx_train, im_class_num)
#     c_largest = labels.max().item()
#     labels = labels.to(embed.device)  
#     idx_train = idx_train.to(embed.device)  
#     N2 = idx_train[(labels == (c_largest - im_class_num - 2))[idx_train]].shape[0]
#     # N2 = idx_train[(labels == (c_largest - im_class_num - 1))].shape[0]
#     N1_new = int(portion * N2)
#     silhouette_scores = []  
#     for i in range(im_class_num):
#         chosen = idx_train[(labels == (c_largest - i))[idx_train]]
#         # chosen = idx_train[(labels[idx_train] == (c_largest - i))]
#         N1_largest = chosen.shape[0]
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         chosen_embed = embed[chosen, :]
#         cluster_labels = kmeans.fit_predict(chosen_embed.cpu().detach().numpy())
#         cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=embed.dtype).to(embed.device)
#         score = silhouette_score(chosen_embed.cpu().detach().numpy(), cluster_labels)
#         silhouette_scores.append(score)
#         for cluster in range(n_clusters):
#             cluster_idx = chosen[cluster_labels == cluster]
#             N_l1 = cluster_idx.shape[0]
#             # n_i = int((N_l1 / N1_largest) * N1_new)
#             n_i = int((N_l1 / N1_largest) * (N1_new-N1_largest))
#             # SMOTE within each cluster
#             for j in range(n_i):
#                 cluster_idx_sampled = cluster_idx[:n_i]
                
#                 distance = squareform(pdist(embed[cluster_idx_sampled].cpu().detach()))
#                 np.fill_diagonal(distance, distance.max() + 100)

#                 idx_neighbor = distance.argmin(axis=-1)

#                 interp_place = random.random()
#                 new_embed = embed[cluster_idx_sampled, :] + (embed[idx_neighbor, :] - embed[cluster_idx_sampled, :]) * interp_place

#                 new_labels = labels.new(torch.Size((cluster_idx_sampled.shape[0], 1))).reshape(-1).fill_(c_largest - i)
#                 idx_new = np.arange(embed.shape[0], embed.shape[0] + cluster_idx_sampled.shape[0])
#                 idx_train_append = idx_train.new(idx_new)

#                 embed = torch.cat((embed, new_embed), 0)
#                 labels = torch.cat((labels, new_labels), 0)
#                 idx_train = torch.cat((idx_train, idx_train_append), 0)

#                 new_size = adj.shape[0] + cluster_idx_sampled.shape[0]
#                 adj_expanded = torch.zeros((new_size, new_size), device=adj.device)

#                 adj_expanded[:adj.shape[0], :adj.shape[0]] = adj

#                 center_idx = embed.shape[0] - cluster_idx_sampled.shape[0]
#                 adj_expanded[center_idx:center_idx+n_i, cluster_idx_sampled] = 1.0
#                 adj_expanded[cluster_idx_sampled, center_idx:center_idx+n_i] = 1.0

#                 nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(embed.cpu().detach().numpy())
#                 distances, neighbors = nbrs.kneighbors(embed[cluster_idx_sampled].cpu().detach().numpy())
#                 neighbors = torch.tensor(neighbors, dtype=torch.long).to(embed.device)

#                 for k in range(cluster_idx_sampled.shape[0]):
#                     adj_expanded[center_idx + k, neighbors[k]] = 1.0
#                     adj_expanded[neighbors[k], center_idx + k] = 1.0

#                 adj = adj_expanded

#     if adj is not None:
#         return embed, labels, idx_train, adj.detach(),silhouette_scores
#     else:
#         return embed, labels, idx_train,silhouette_scores

def recon_upsample_kmeans_smote(embed, labels, idx_train, adj=None, portion=1.0, im_class_num=3, n_clusters=3, k_neighbors=3):
    assert embed.shape[0] == labels.shape[0]
    if len(idx_train) == 0:
        return embed, labels, idx_train, adj, []
    
    labels = labels.to(embed.device)
    idx_train = idx_train.to(embed.device)
    
    embed, labels, idx_train = cnn_denoising(embed, labels, idx_train, im_class_num)
    device = embed.device
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    c_largest = labels.max().item()
    N2 = idx_train[(labels == (c_largest - im_class_num - 2))[idx_train]].shape[0]
    N1_new = max(1, int(portion * N2))  # 确保至少生成1个样本
    silhouette_scores = []

    for i in range(im_class_num):
        chosen = idx_train[(labels == (c_largest - i))[idx_train]]
        N1_largest = chosen.shape[0]
        
        if N1_largest < 1:
            print(f"警告: 类别 {c_largest-i} 无训练样本，跳过")
            silhouette_scores.append(-1)  # 无效分数
            continue
            
        chosen_embed = embed[chosen, :]
        
        actual_clusters = min(n_clusters, N1_largest)
        if actual_clusters < 2:
            print(f"警告: 类别 {c_largest-i} 样本不足({N1_largest})，使用单簇")
            cluster_labels = np.zeros(N1_largest)
            cluster_centers = chosen_embed.mean(dim=0, keepdim=True)
        else:
            kmeans = KMeans(n_clusters=actual_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(chosen_embed.cpu().detach().numpy())
            cluster_centers = torch.tensor(kmeans.cluster_centers_, 
                                         dtype=embed.dtype).to(embed.device)
        
        if N1_largest >= 2 and actual_clusters >= 2:
            score = silhouette_score(chosen_embed.cpu().detach().numpy(), cluster_labels)
        else:
            score = -1  # 无效分数
        silhouette_scores.append(score)

        for cluster in range(actual_clusters):
            cluster_idx = chosen[cluster_labels == cluster]
            N_l1 = cluster_idx.shape[0]
            
            if N1_largest == 0:
                n_i = 0
            else:
                n_i = max(1, int((N_l1 / N1_largest) * (N1_new - N1_largest)))
            
            if n_i <= 0:
                continue

            cluster_idx_sampled = cluster_idx[:min(N_l1, n_i)]
            
            for j in range(n_i):
                if len(cluster_idx_sampled) < 2:
                    new_embed = embed[cluster_idx_sampled].mean(dim=0, keepdim=True)
                else:
                    distance = squareform(pdist(embed[cluster_idx_sampled].cpu().detach()))
                    np.fill_diagonal(distance, distance.max() + 100)
                    idx_neighbor = distance.argmin(axis=-1)
                    interp_place = random.random()
                    embed_neighbor = embed[idx_neighbor].to(embed.device)
                    new_embed = embed[cluster_idx_sampled] + (embed_neighbor - embed[cluster_idx_sampled]) * interp_place

                new_labels = torch.full((new_embed.shape[0],), c_largest - i, 
                                      dtype=labels.dtype, device=labels.device)
                idx_new = torch.arange(embed.shape[0], embed.shape[0] + new_embed.shape[0], 
                                     device=idx_train.device)
                
                embed = torch.cat((embed, new_embed), 0)
                labels = torch.cat((labels, new_labels), 0)
                idx_train = torch.cat((idx_train, idx_new), 0)

                if adj is not None:
                    new_size = embed.shape[0]
                    adj_expanded = torch.zeros((new_size, new_size), device=adj.device)
                    adj_expanded[:adj.shape[0], :adj.shape[0]] = adj
                    
                    if new_embed.shape[0] > 0:
                        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, embed.shape[0]-1))
                        nbrs.fit(embed.cpu().detach().numpy())
                        distances, neighbors = nbrs.kneighbors(new_embed.cpu().detach().numpy())
                        
                        for k in range(new_embed.shape[0]):
                            for nb in neighbors[k]:
                                adj_expanded[embed.shape[0]-new_embed.shape[0]+k, nb] = 1.0
                                adj_expanded[nb, embed.shape[0]-new_embed.shape[0]+k] = 1.0
                    adj = adj_expanded

    return embed, labels, idx_train, adj.detach() if adj is not None else None, silhouette_scores





def adj_mse_loss(adj_rec, adj_tgt, adj_mask = None):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0]**2

    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt==0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    return loss




