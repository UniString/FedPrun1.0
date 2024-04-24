import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import copy
import numpy as np
import random
import Algorithm
import argparse
import sys
import io
import wandb

from models.Fed import Aggregation
from utils.utils import save_result, save_fedmut_result,save_model
from models.test import test_img
from models.Update import DatasetSplit
from optimizer.Adabelief import AdaBelief
from Algorithm.core import Masking, CosineDecay


class LocalUpdate_FedMut(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ensemble_alpha = args.ensemble_alpha
        self.verbose = verbose

    def train(self, net):

        net.to(self.args.device)

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0

        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                model_output = net(images)
                predictive_loss = self.loss_func(model_output['output'], labels)

                loss = predictive_loss
                Predict_loss += predictive_loss.item()

                loss.backward()
                optimizer.step()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        # net.to('cpu')

        return net.state_dict()

def FedMut(args, net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()###train SM?
    acc = []
    w_locals = []
    sim_arr = []
    lr_decay=1
    initial_lr = args.lr
    #args.lr = initial_lr * (lr_decay ** (iter // 100))


    m = max(int(args.frac * args.num_users), 1)
    for i in range(m):
        w_locals.append(copy.deepcopy(net_glob.state_dict()))#复制客户端参数m份
    
    delta_list = []
    max_rank = 0

    w_old = copy.deepcopy(net_glob.state_dict())#复制旧参数

    #w_old_s1 = copy.deepcopy(net_glob.state_dict())

    for iter in range(args.epochs):
        args.lr = initial_lr * (0.6 ** (iter // 100))

        w_old = copy.deepcopy(net_glob.state_dict())
        print('*' * 80)
        print('Round {:3d}'.format(iter))

        



        m = max(int(args.frac * args.num_users), 1)          #每次参与客户端数
        idxs_users = np.random.choice(range(args.num_users), m, replace=False) #抽取客户端
        for i, idx in enumerate(idxs_users):
            




            
            net_glob.load_state_dict(w_locals[i])
            # print('准备剪枝')
            # 保存原始的标准输出
            original_stdout = sys.stdout

            # 创建一个丢弃输出的流
            sys.stdout = io.StringIO()

            # 在这个代码块下，所有的print调用都会被忽略
            # 例如:
            # my_class_instance.method_that_prints()

            # 恢复标准输出

            ####剪枝
            net_local = None
            net_local = copy.deepcopy(net_glob).to(args.device)
            mask = None   #储存掩码
            
            optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
            
            train_loader = torch.utils.data.DataLoader(
                dataset_train,
                50,  #batchsize 
                num_workers=12,
                pin_memory=True, shuffle=True)
            
            if args.sparse:
                decay = CosineDecay(args.death_rate, len(train_loader)*(args.epochs*args.multiplier))        #稀疏率，训练总步数 计算衰减率
                mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                           redistribution_mode=args.redistribution, args=args,train_loader=train_loader)
                mask.add_module(net_local, sparse_init=args.sparse_init, density=args.density_local)
               # mask.get_mask 获取模型稀疏矩阵

            sys.stdout = original_stdout
            

            local = LocalUpdate_FedMut(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=net_local)
            w_locals[i] = copy.deepcopy(w)

        # update global weights
        w_glob = Aggregation(w_locals, None) # Global Model Generation

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        if iter % 3 == 2:
            item_acc = test(net_glob, dataset_test, args)
            acc.append(item_acc)
            wandb.log({"epoch":iter,"acc": item_acc})

        w_delta = FedSub(w_glob, w_old, 1.0)
        rank = delta_rank(args,w_delta)
        print(rank)
        if rank > max_rank:
            max_rank = rank
        alpha = args.radius
        # alpha = min(max(args.radius, max_rank/rank),(10.0-args.radius) * (1 - iter/args.epochs) + args.radius)
        w_locals = mutation_spread(args, iter, w_glob, w_old, w_locals, m, w_delta, alpha)
        


    save_fedmut_result(acc, 'test_acc', args)
    save_model(net_glob.state_dict(), 'test_model', args)
    wandb.finish()
    # save_result(sim_arr, 'sim', args)



def mutation_spread(args, iter, w_glob, w_old, w_locals, m, w_delta, alpha):
    # w_delta = FedSub(w_glob,w_old,(args.radius - args.min_radius) * (1.0 - iter/args.epochs) + args.min_radius)
    # if iter/args.epochs > 0.5:
    #     w_delta = FedSub(w_glob,w_old,(args.radius - args.min_radius) * (1.0 - iter/args.epochs)*2 + args.min_radius)
    # else:
        # w_delta = FedSub(w_glob,w_old,(args.radius - args.min_radius) * (iter/args.epochs)*2 + args.min_radius)
    # w_delta = FedSub(w_glob, w_old, args.radius)


    w_locals_new = []
    ctrl_cmd_list = []
    ctrl_rate = args.mut_acc_rate * (1.0 - min(iter*1.0/args.mut_bound,1.0))

    for k in w_glob.keys():
        ctrl_list = []
        for i in range(0,int(m/2)):
            ctrl = random.random()
            if ctrl > 0.5:
                ctrl_list.append(1.0)
                ctrl_list.append(1.0 * (-1.0 + ctrl_rate))
            else:
                ctrl_list.append(1.0 * (-1.0 + ctrl_rate))
                ctrl_list.append(1.0)
        random.shuffle(ctrl_list)
        ctrl_cmd_list.append(ctrl_list)
    cnt = 0
    for j in range(m):
        w_sub = copy.deepcopy(w_glob)
        if not (cnt == m -1 and m%2 == 1):
            ind = 0
            for k in w_sub.keys():
                w_sub[k] = w_sub[k] + w_delta[k]*ctrl_cmd_list[ind][j]*alpha
                ind += 1
        cnt += 1
        w_locals_new.append(w_sub)


    return w_locals_new
#老方法
'''def mutation_spread_weight(args, iter, w_glob, w_old, w_locals, m, w_delta, alpha)
    # w_delta = FedSub(w_glob,w_old,(args.radius - args.min_radius) * (1.0 - iter/args.epochs) + args.min_radius)
    # if iter/args.epochs > 0.5:
    #     w_delta = FedSub(w_glob,w_old,(args.radius - args.min_radius) * (1.0 - iter/args.epochs)*2 + args.min_radius)
    # else:
        # w_delta = FedSub(w_glob,w_old,(args.radius - args.min_radius) * (iter/args.epochs)*2 + args.min_radius)
    # w_delta = FedSub(w_glob, w_old, args.radius)


    w_locals_new = []
    ctrl_cmd_list = []
    ctrl_rate = args.mut_acc_rate * (1.0 - min(iter*1.0/args.mut_bound,1.0))

    for k in w_glob.keys():
        ctrl_list = []
        for i in range(0,int(m/2)):
            ctrl = random.random()
            if ctrl > 0.5:
                ctrl_list.append(1.0)
                ctrl_list.append(1.0 * (-1.0 + ctrl_rate))
            else:
                ctrl_list.append(1.0 * (-1.0 + ctrl_rate))
                ctrl_list.append(1.0)
        random.shuffle(ctrl_list)
        ctrl_cmd_list.append(ctrl_list)
    cnt = 0
    for j in range(m):
        w_sub = copy.deepcopy(w_glob)
        if not (cnt == m -1 and m%2 == 1):
            ind = 0
            for k in w_sub.keys():
                w_sub[k] = w_sub[k] + w_delta[k]*ctrl_cmd_list[ind][j]*alpha
                ind += 1
        cnt += 1
        w_locals_new.append(w_sub)


    return w_locals_new'''


def test(net_glob, dataset_test, args):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()


def FedSub(w, w_old, weight):
    w_sub = copy.deepcopy(w)
    for k in w_sub.keys():
        w_sub[k] = (w[k] - w_old[k])*weight

    return w_sub

def delta_rank(args,delta_dict):
    cnt = 0
    dict_a = torch.Tensor(0)
    s = 0
    for p in delta_dict.keys():
        a = delta_dict[p]
        a = a.view(-1)
        if cnt == 0:
            dict_a = a
        else:
            dict_a = torch.cat((dict_a, a), dim=0)
               
        cnt += 1
            # print(sim)
    s = torch.norm(dict_a, dim=0)
    return s