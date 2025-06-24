import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import models
import utils
import data_load
import random
import ipdb
import copy
# from models import GAT, select_edges_based_on_attention
#from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
import umap.umap_ as umap
import os
# Training setting
parser = utils.get_parser()

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

'''
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
'''

# Load data
if args.dataset == 'cora':
    adj, features, labels = data_load.load_data()
    # print('adj:',adj)
    # print("features:",features)
    # print("labels:",labels)
    class_sample_num = 20
    im_class_num = 3
elif args.dataset == 'citeseer':
    adj, features, labels = data_load.load_cite()
    im_class_num = 2 #set it to be the number less than 100
    class_sample_num = 20 #not used
elif args.dataset == 'pubmed':
    adj, features, labels = data_load.load_pub()
    class_sample_num = 20
    im_class_num = 1
elif args.dataset == 'BlogCatalog':
    adj, features, labels = data_load.load_data_Blog()
    im_class_num = 18 #set it to be the number less than 100
    class_sample_num = 20 #not used
elif args.dataset == 'twitter':
    adj, features, labels = data_load.load_data_twitter()
    im_class_num = 1
    class_sample_num = 20 #not used
elif args.dataset == 'graph':
    adj, features, labels = data_load.load_graph()
    class_sample_num = 200
    im_class_num = 1
else:
    print("no this dataset: {args.dataset}")


#for artificial imbalanced setting: only the last im_class_num classes are imbalanced
c_train_num = []
for i in range(labels.max().item() + 1):
    if args.imbalance and i > labels.max().item()-im_class_num: #only imbalance the last classes
        c_train_num.append(int(class_sample_num*args.im_ratio))

    else:
        c_train_num.append(class_sample_num)

print('c_train_num',c_train_num) #存储每个类别的训练样本数量
#get train, validatio, test data split
if args.dataset == 'BlogCatalog':
    reduce_classes = 1  # 可以通过args传入
    
    # 获取划分
    idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine1(
        labels, 
        reduce_classes=reduce_classes
    )
# if args.dataset == 'BlogCatalog':
#     idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)
    # print(class_num_mat)
elif args.dataset == 'cora':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_arti(labels, c_train_num)

elif args.dataset == 'citeseer':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_arti(labels, c_train_num)
elif args.dataset == 'pubmed':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_arti(labels, c_train_num)
elif args.dataset == 'twitter':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)
elif args.dataset == 'graph':
    idx_train, idx_val, idx_test, class_num_mat = utils.split_arti(labels, c_train_num)

#method_1: oversampling in input domain
if args.setting == 'upsampling':
    adj,features,labels,idx_train = utils.src_upsample(adj,features,labels,idx_train,portion=args.up_scale, im_class_num=im_class_num)
if args.setting == 'smote':
    adj,features,labels,idx_train = utils.src_smote(adj,features,labels,idx_train,portion=args.up_scale, im_class_num=im_class_num)


# Model and optimizer
#if oversampling in the embedding space is required, model need to be changed
if args.setting != 'embed_up':
    if args.model == 'sage':
        encoder = models.Sage_En(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.Sage_Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)
    elif args.model == 'gcn':
        encoder = models.GCN_En(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.GCN_Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)
    elif args.model == 'GAT':
        encoder = models.GAT_En(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.GAT_Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)
else:
    if args.model == 'sage':
        encoder = models.Sage_En2(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)
    elif args.model == 'gcn':
        encoder = models.GCN_En2(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)
    elif args.model == 'GAT':
        encoder = models.GAT_En2(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.Classifier(nembed=args.nhid, 
                nhid=args.nhid, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout)



decoder = models.Decoder(nembed=args.nhid,
        dropout=args.dropout)


optimizer_en = optim.Adam(encoder.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
optimizer_cls = optim.Adam(classifier.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
optimizer_de = optim.Adam(decoder.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)



if args.cuda:
    encoder = encoder.cuda()
    classifier = classifier.cuda()
    decoder = decoder.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


# Q-learning参数
alpha = 0.5 
gamma = 0.95  
epsilon = 0.2 
n_clusters_min = 2  
n_clusters_max = 8  
n_actions = n_clusters_max - n_clusters_min + 1 
Q_table = np.zeros(n_actions) 

best_k = None
best_reward = -float('inf')

train_loss_history = []   
val_loss_history = []      
train_acc_history = []    
val_acc_history = []      

def update_best_k(k, reward):
    global best_k, best_reward
    if reward > best_reward:
        best_k = k
        best_reward = reward

def choose_action(Q_table, epsilon):
    a=random.uniform(0, 1)
    if a < epsilon:
        b=random.randint(0, n_actions-1)
        return b 
    else:
        c=np.argmax(Q_table)
        return c  




def calculate_clustering_reward(embed, labels_new):

    embed_cpu = embed.detach().cpu().numpy()
    labels_new_cpu = labels_new.detach().cpu().numpy()


    reward = silhouette_score(embed_cpu, labels_new_cpu)

    return reward
def train(epoch):
    global best_k
    t = time.time()
    encoder.train()
    classifier.train()
    decoder.train()
    optimizer_en.zero_grad()
    optimizer_cls.zero_grad()
    optimizer_de.zero_grad()

    embed = encoder(features, adj)# 通过编码器获取节点嵌入
    # print("embed:",embed)

    if args.setting == 'recon_newG' or args.setting == 'recon' or args.setting == 'newG_cls':
        ori_num = labels.shape[0]
        #smote过采样
        
        # embed, labels_new, idx_train_new, adj_up = utils.recon_upsample(embed, labels, idx_train, adj=adj.detach().to_dense(),portion=args.up_scale, im_class_num=im_class_num)

      
        action = choose_action(Q_table, epsilon)
      
        n_clusters = max(2, action+2)   

        embed, labels_new, idx_train_new, adj_up,silhouette_scores = utils.recon_upsample_kmeans_smote(embed, labels, idx_train, adj=adj.detach().to_dense(), portion=args.up_scale, im_class_num=im_class_num, n_clusters=n_clusters, k_neighbors=args.k_neighbors)
        
        reward=max(silhouette_scores)
        reward = calculate_clustering_reward(embed, labels_new)
        for i in range(n_actions):
            next_action = choose_action(Q_table, epsilon) 
          
            Q_table[action] += alpha * (reward + gamma * np.max(Q_table[next_action]) - Q_table[action])
           

            update_best_k(n_clusters, reward)



        generated_G = decoder(embed)# 使用解码器生成新的图结构

        #计算重构损失
        loss_rec = utils.adj_mse_loss(generated_G[:ori_num, :][:, :ori_num], adj.detach().to_dense())

        #ipdb.set_trace()

        if not args.opt_new_G:#是否根据分类结果优化解码后的图,默认False
            #对生成图的二值化处理
            adj_new = copy.deepcopy(generated_G.detach())
            threshold = 0.5
            adj_new[adj_new<threshold] = 0.0
            adj_new[adj_new>=threshold] = 1.0
            #ipdb.set_trace()
            edge_ac = adj_new[:ori_num, :ori_num].eq(adj.to_dense()).double().sum()/(ori_num**2)
        else:
             # 如果使用新的图结构，计算与原始图的差异
            adj_new = generated_G
            edge_ac = F.l1_loss(adj_new[:ori_num, :ori_num], adj.to_dense(), reduction='mean')

        # 计算现有节点和生成节点的边概率，并打印相关信息
        #calculate generation information
        exist_edge_prob = adj_new[:ori_num, :ori_num].mean() #edge prob for existing nodes
        generated_edge_prob = adj_new[ori_num:, :ori_num].mean() #edge prob for generated nodes
        print("edge acc: {:.4f}, exist_edge_prob: {:.4f}, generated_edge_prob: {:.4f}".format(edge_ac.item(), exist_edge_prob.item(), generated_edge_prob.item()))

        # print(adj_new)
        # 对邻接矩阵的过滤和更新操作
        adj_new = torch.mul(adj_up, adj_new)
        exist_edge_prob = adj_new[:ori_num, :ori_num].mean() #edge prob for existing nodes#
        generated_edge_prob = adj_new[ori_num:, :ori_num].mean() #edge prob for generated nodes
        print("after filtering, edge acc: {:.4f}, exist_edge_prob: {:.4f}, generated_edge_prob: {:.4f}".format(edge_ac.item(), exist_edge_prob.item(), generated_edge_prob.item()))

        adj_new[:ori_num, :][:, :ori_num] = adj.detach().to_dense()# 将原始节点间的邻接关系恢复为原始邻接矩阵的值
        #adj_new = adj_new.to_sparse()
        #ipdb.set_trace()

        if not args.opt_new_G:
            adj_new = adj_new.detach()

        if args.setting == 'newG_cls':
            idx_train_new = idx_train

    elif args.setting == 'embed_up':

        #perform SMOTE in embedding space
        embed, labels_new, idx_train_new = utils.recon_upsample(embed, labels, idx_train,portion=args.up_scale, im_class_num=im_class_num)
        adj_new = adj
    else:
        labels_new = labels
        idx_train_new = idx_train
        adj_new = adj
       


    #ipdb.set_trace()节点分类器
    output = classifier(embed, adj_new) # 使用分类器进行前向传播

    # 根据参数 args.setting 决定是否需要对损失函数进行加权处理
    if args.setting == 'reweight':
        weight = features.new((labels.max().item()+1)).fill_(1)
        weight[-im_class_num:] = 1+args.up_scale
        loss_train = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new], weight=weight)
    else:
        loss_train = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new])

    acc_train = utils.accuracy(output[idx_train], labels_new[idx_train])# 计算训练准确率

    # 计算总损失，可能包括重构损失和训练损失
    if args.setting == 'recon_newG':
        loss = loss_train+loss_rec*args.rec_weight
    elif args.setting == 'recon':
        loss = loss_rec + 0*loss_train
    else:
        loss = loss_train#只关心节点分类任务
        loss_rec = loss_train


    loss.backward() # 反向传播，计算梯度
    # 根据设置更新优化器的梯度
    if args.setting == 'newG_cls':
        optimizer_en.zero_grad()
        optimizer_de.zero_grad()
    else:
        optimizer_en.step()

    optimizer_cls.step()# 更新分类器的参数

    if args.setting == 'recon_newG' or args.setting == 'recon':
        optimizer_de.step()# 更新解码器的参数

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])# 计算验证损失
    acc_val = utils.accuracy(output[idx_val], labels[idx_val])# 计算验证准确率

    #ipdb.set_trace()
    utils.print_class_acc(output[idx_val], labels[idx_val], class_num_mat[:,1])

    print('Epoch: {:05d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_rec: {:.4f}'.format(loss_rec.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
        # 在print之后添加记录指标
   

def test(epoch=0):
    encoder.eval()
    classifier.eval()
    decoder.eval()
    embed = encoder(features, adj)
    output = classifier(embed, adj)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = utils.accuracy(output[idx_test], labels[idx_test])
    
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    utils.print_class_acc(output[idx_test], labels[idx_test], class_num_mat[:,2], pre='test')

    '''
    if epoch==40:
        torch
    '''
        

def save_model(epoch):
    saved_content = {}

    saved_content['encoder'] = encoder.state_dict()
    saved_content['decoder'] = decoder.state_dict()
    saved_content['classifier'] = classifier.state_dict()

    torch.save(saved_content, 'checkpoint/{}/{}_{}_{}_{}.pth'.format(args.dataset,args.setting,epoch, args.opt_new_G, args.im_ratio))
    return

def load_model(filename):
    loaded_content = torch.load('checkpoint/{}/{}.pth'.format(args.dataset,filename), map_location=lambda storage, loc: storage)

    encoder.load_state_dict(loaded_content['encoder'])
    decoder.load_state_dict(loaded_content['decoder'])
    classifier.load_state_dict(loaded_content['classifier'])

    print("successfully loaded: "+ filename)

    return

# Train model
if args.load is not None:
    load_model(args.load)

t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)

    if epoch % 10 == 0:
        test(epoch) 

    if epoch % 100 == 0:
        save_model(epoch)


print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))



# 在训练完成后调用
# print_best_k()
# print(f"Q_table after update: {Q_table}")
# print(f'The best k (n_clusters) value is: {np.argmax(Q_table)+2}')
# Testing
test()
