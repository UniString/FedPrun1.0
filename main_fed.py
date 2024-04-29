#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import copy
import Algorithm
import argparse
import wandb


from utils.options import args_parser
from utils.set_seed import set_random_seed
from models.Update import *
from models.Nets import *
from models.Fed import Aggregation,AggregationMut
from models.test import test_img
from models.resnetcifar import *
from models import *
from utils.get_dataset import get_dataset
from utils.utils import save_result,save_model
from Algorithm.Training_FedGen import FedGen
from Algorithm.Training_FedMut import FedMut
from Algorithm.core import Masking, CosineDecay
import sys
import io   

def get_densitys(model):
    densitys={}
    total_size=0
    sparse_size=0
    for name, weight in model.named_parameters():       
        #print(name, 'density:',(weight!=0).sum().item()/weight.numel())
        densitys[name]=(weight!=0).sum().item()/weight.numel()
        total_size += weight.numel()
        sparse_size += (weight != 0).sum().int().item()
    Total_density=sparse_size / total_size
    #print('Total Model parameters:', total_size)
    print('Total parameters under sparsity level of {0}: {1}'.format(args.density_local, sparse_size / total_size))
    return Total_density

def Lr_decay(args,lr_decay,initial_lr,iter):       #指数学习率衰减函数
    args.lr = initial_lr * (lr_decay ** (iter // 100))


def FedAvg(net_glob, dataset_train, dataset_test, dict_users):
    
    net_glob.train()
    #
    # training
    acc = []
    densitys = {}
    densitys_get = True
    #density_local_store=args.density_local
    initial_lr=args.lr
    initial_fix= args.density_fix  #Fix 的上限
    args.density_fix=0.3
    #args.density_local=0
    
    
    for iter in range(args.epochs):
        # if iter % 50 == 0 :
        #     TurnFlag=not TurnFlag
        #     print(TurnFlag)
        
        if args.density_fix < initial_fix   :
            args.density_fix+=args.density_gr#固定率线性增加
            # else:
            #     args.density_fix-=args.density_gr#固定率线性衰减
        
        if args.density_local > 0  :
             args.density_local=args.density_local-args.density_dr #稀疏率线性衰减
        # if sparse_learning_epoch > 0 and args.density_fix >= 0.5:
        #     args.density_fix-=args.density_gr
        #     args.density_local+=args.density_dr
        #     sparse_learning_epoch-=1
        
        #if iter > 400:
        Lr_decay(args,args.lr_decay,initial_lr,iter)   


        print(args.density_local)

        print('*'*80)
        print('Round {:3d}'.format(iter))


        w_locals = []#包含m个设备状态字典的列表
        w_masks = []#包含m个设备状态字典的列表
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #optimizer = torch.optim.SGD(net_glob.parameters(), lr=self.args.lr, momentum=self.args.momentum)


        flag = True
        flag2= True
        for idx in idxs_users:
            
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
            
            
            
            if args.sparse:
                global rand_indices_dic
                decay = CosineDecay(args.death_rate, len(train_loader)*(args.epochs*args.multiplier))        #稀疏率，训练总步数 计算衰减率
                mask = Masking(rand_indices_dic,optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                           redistribution_mode=args.redistribution, args=args,train_loader=train_loader)
                mask.add_module(net_local, sparse_init=args.sparse_init, density=args.density_local)
                
                # if densitys_get :
                #     densitys = mask.get_densitys()
                #     densitys_get = True
                    
            sys.stdout = original_stdout
            w_masks.append(copy.deepcopy(net_local.state_dict()))#剪完枝尚未训练的模型参数
            #get_densitys(net_local)
            if iter % 5 == 1 and flag:
                print("聚合前")
                #Total_density_B=get_densitys(net_local)#修复bug 
                #wandb.log({"Total_density_B":Total_density_B})
                #print(densitys)
                
                flag=False
            
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])

            w = local.train(net=net_local)#返回参数状态字典
            if iter % 5 == 1 and flag2:
                #get_densitys(net_local)
                #density_localtrain=get_densitys(net_local)
                #wandb.log({"density_ALocaltrain":density_localtrain})
                flag2=False
            w_locals.append(copy.deepcopy(w))   
            lens.append(len(dict_users[idx]))
            
        
        
        # update global weights   新方法
        if not args.aggmut:
            w_glob = Aggregation(w_locals, lens )
        else:
            w_glob = AggregationMut(w_locals, lens,w_masks)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        #item_acc = test(net_glob, dataset_test, args)

        if iter % 3 == 1:
            item_acc = test(net_glob, dataset_test, args)  #有修改
            acc.append(item_acc)
            
            print("聚合后")
            #Total_density_A=get_densitys(net_glob)
            wandb.log({"epoch":iter,"acc": item_acc,"density_local":args.density_local,"lr":args.lr,"density_fix":args.density_fix,})#"Total_density":Total_density_A
            #print(densitys)
            
    
    
    

    save_result(acc, 'test_acc', args)
    save_model(net_glob.state_dict(), 'test_model', args)
    wandb.finish()


def FedProx(net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()

    acc = []

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_FedProx(args=args, glob_model=net_glob, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc.append(test(net_glob, dataset_test, args))

    save_result(acc, 'test_acc', args)


from utils.clustering import *
from scipy.cluster.hierarchy import linkage
def ClusteredSampling(net_glob, dataset_train, dataset_test, dict_users):

    net_glob.to('cpu')

    n_samples = np.array([len(dict_users[idx]) for idx in dict_users.keys()])
    weights = n_samples / np.sum(n_samples)
    n_sampled = max(int(args.frac * args.num_users), 1)

    gradients = get_gradients('', net_glob, [net_glob] * len(dict_users))

    net_glob.train()

    # training
    acc = []

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        previous_global_model = copy.deepcopy(net_glob)
        clients_models = []
        sampled_clients_for_grad = []

        # GET THE CLIENTS' SIMILARITY MATRIX
        if iter == 0:
            sim_matrix = get_matrix_similarity_from_grads(
                gradients, distance_type=args.sim_type
            )

        # GET THE DENDROGRAM TREE ASSOCIATED
        linkage_matrix = linkage(sim_matrix, "ward")

        distri_clusters = get_clusters_with_alg2(
            linkage_matrix, n_sampled, weights
        )

        w_locals = []
        lens = []
        idxs_users = sample_clients(distri_clusters)
        for idx in idxs_users:
            local = LocalUpdate_ClientSampling(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local_model = local.train(net=copy.deepcopy(net_glob).to(args.device))
            local_model.to('cpu')

            w_locals.append(copy.deepcopy(local_model.state_dict()))
            lens.append(len(dict_users[idx]))

            clients_models.append(copy.deepcopy(local_model))
            sampled_clients_for_grad.append(idx)

            del local_model
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        gradients_i = get_gradients(
            '', previous_global_model, clients_models
        )
        for idx, gradient in zip(sampled_clients_for_grad, gradients_i):
            gradients[idx] = gradient

        sim_matrix = get_matrix_similarity_from_grads_new(
            gradients, distance_type=args.sim_type, idx=idxs_users, metric_matrix=sim_matrix
        )

        net_glob.to(args.device)
        acc.append(test(net_glob, dataset_test, args))
        net_glob.to('cpu')

        del clients_models

    save_result(acc, 'test_acc', args)

def test(net_glob, dataset_test, args):
    
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()

def test_with_loss(net_glob, dataset_test, args):
    
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item(), loss_test

if __name__ == '__main__':
    # parse args  
    
     
    
    args = args_parser()
    # 设置要使用的GPU索引
    device_index = args.gpu
    

    # 选择指定的GPU
    torch.cuda.set_device(device_index)
    # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.helf_sparse:
        #args.density_local= 2*args.density_local-1.0
        #args.density_local= 1.5*args.density_local-0.5
        print(args.density_local)

    set_random_seed(args.seed)

    dataset_train, dataset_test, dict_users = get_dataset(args)

    if 'cifar' in args.dataset and 'cnn' in args.model:##魔改剪枝    
        net_glob = CNNCifar(args)
    elif 'resnet18' in args.model:
        net_glob = ResNet18_cifar10(num_classes = args.num_classes)
        if args.finetuning :
            net_glob.load_state_dict(torch.load('./output/5/0.5/cifar10_FedAvg_resnet18_test_model_1000_lr_0.03_2024_04_11_07_31_31_frac_0.1.txt'))#非独立三分稀疏1000轮
            #net_glob.load_state_dict(torch.load('./output/5/0.5/cifar10_FedAvg_resnet18_test_model_250_lr_0.03_2024_04_11_04_41_24_frac_0.1.txt'))#非独立半稀疏750轮0.6
            #net_glob.load_state_dict(torch.load('./output/5/0.5/cifar10_FedAvg_resnet18_test_model_500_lr_0.03_2024_04_11_03_44_02_frac_0.1.txt'))#非独立半稀疏500轮0.6
            
            net_glob.load_state_dict(torch.load('./output/5/0.5/noniid_avgbase_cifar10_FedAvg_resnet18_test_model_250_lr_0.03_2024_04_09_18_03_12_frac_0.1.txt')) #非独立AvgBase
            #net_glob.load_state_dict(torch.load('./output/5/0.5/cifar10_FedAvg_resnet18_test_model_250_lr_0.05_2024_04_10_11_58_03_frac_0.1.txt'))    #非独立AvgBase微调250轮0.6
    elif 'resnet50' in args.model:
        net_glob = ResNet50_cifar10(num_classes = args.num_classes)
    elif 'lstm' in args.model:
        net_glob = CharLSTM()
    elif 'cifar' in args.dataset and args.model == 'vgg':
        net_glob = VGG16(args)
        if args.finetuning :
            net_glob.load_state_dict(torch.load('./output/5/0.5/cifar10_FedAvg_vgg_test_model_1000_lr_0.03437968813371699_2024_04_20_14_15_48_frac_0.1.txt'))
           
            #net_glob.load_state_dict(torch.load('./output/0/cifar10_FedAvg_vgg_test_model_250_lr_0.03_2024_04_07_21_26_18_frac_0.1.txt'))
            #net_glob.load_state_dict(torch.load('./output/0/cifar10_FedMut_vgg_test_model_300_lr_0.03_2024_04_05_23_36_23_frac_0.1.txt'))
        #elif args.finetuning == "MutBase":
    
    net_glob.to(args.device)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net_glob.parameters(), lr=args.lr)
    elif args.optimizer == 'adaBelief':
        optimizer = AdaBelief(net_glob.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
            
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        50,  #batchsize
        num_workers=8,
        pin_memory=True, shuffle=True)
    



    def remove_weight_partial_name(masks, partial_name):
        for name in list(masks.keys()):
            if partial_name in name:
                masks.pop(name)
    masksdic={}
    rand_indices_dic={}
    if args.sparse:
                # decay = CosineDecay(args.death_rate, len(train_loader)*(args.epochs*args.multiplier))        #稀疏率，训练总步数 计算衰减率
                # mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                #            redistribution_mode=args.redistribution, args=args,train_loader=train_loader)
                # mask.add_module(net_glob, sparse_init=args.sparse_init, density=args.density)
            masks={}
            for name, tensor in net_glob.named_parameters():
                if 'bias' not in name:
                    masksdic[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda(args.gpu)
            
            for name, module in net_glob.named_modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                    if name in masksdic:
                        masksdic.pop(name)
                        print('pop'+name)
                    elif name + '.weight' in masksdic:
                        masksdic.pop(name + '.weight')
                        print('pop'+name+'.weight')
            
            

            for k in masksdic.keys():
                print(k)
                num_elements = masksdic[k].numel()
                rand_indices_dic[k] = torch.randperm(num_elements)
 


    wandb.init(config=args,
               project='FedPrun',
               #entity='mastlab-t3s-org',
               group=args.model,
               name=args.exname,
               job_type="training",
               reinit=True,
               #mode="disabled"
               )
       
    
    
    print(net_glob)

    if args.algorithm == 'FedAvg':
        FedAvg(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedProx':
        FedProx(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'ClusteredSampling':
        ClusteredSampling(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedGen':
        FedGen(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedMut':
        FedMut(args, net_glob, dataset_train, dataset_test, dict_users)

