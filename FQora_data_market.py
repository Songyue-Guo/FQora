from Agents.agents import Seller, Buyer, Arbiter,SellerUtility,BuyerUtility,ArbiterUtility
from DataQuery.buyer_query import buyer_query
from tools.Config import Config
from DataQuery.data_process import split_dataset_by_given_seller_labels
from utils.mean_variance_frontier import mean, variance, covariance
import torch
import torch.nn as nn
import torchvision
import argparse
import json
import pandas as pd
import time
from tqdm import tqdm
from quality_function.q_function import count_feature_mean_hog, count_label_balance
from utils.util import find_data,get_query_dataset
from price_generation import MLPRegressor,align_score2anchorprice,price_anchor_point_cifar100,price_anchor_point_cifar10
from utils.util import gradient_normalizers,get_opt,get_parameters
from utils.min_norm_solvers import MinNormSolver
import os
from utils.methods import balancedGrad,BAGrad

parse  = argparse.ArgumentParser(description="FQora: Towards Fair quality-based data market")
parse.add_argument("-d","--dataset", type=str, default="CIFAR10")
parse.add_argument("-T",type=int, default=1)
parse.add_argument("--gpu", type=int, default=0)
parse.add_argument("-a","--alogrithm",type=str,default='FQora')
args = parse.parse_args()


device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')

#load config
config = Config()
print(args)
if args.dataset == "CIFAR10":
    config.set_dic({
    "num_seller": 5,
    "num_buyer": 5,
    "dataset": "cifar10",
    "seller_data_distribution": "split",
    "slabel": [],
    "opt":"Adam",
    "margin_rate": 0.2,
    "step_size": 0,
    "lr":0.05,
    "seller_labels": {0: [0,1], 1: [2,3], 2: [4,5], 3: [6,7], 4: [8,9]},
    "max_epoch": 100,
    "max_value": 100,
    "c": 0.05
    })
    cfg = config
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# train_dataset = torchvision.datasets.MNIST("../datasets/MNIST", train=True, download=True, transform=transform)  # 28 * 28
    seller_dataset = torchvision.datasets.CIFAR10("./datasets/CIFAR10_data", train=True, download=True, transform=transform)
elif args.dataset == "CIFAR100":
    config.set_dic({
    "num_seller": 5,
    "num_buyer": 5,
    "dataset": "cifar100",
    "seller_data_distribution": "split",
    "slabel": [],
    "opt":"Adam",
    "margin_rate": 0.2,
    "step_size": 0,
    "lr":0.03,
    "seller_labels": {0: [0,1,2,3], 1: [4,5,6,7], 2: [8,9,10,11], 3: [12,13,14,15], 4: [16,17,18,19]},
    "max_epoch": 100,
    "max_value": 100,
    "c": 0.5
    })
    cfg = config
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# train_dataset = torchvision.datasets.MNIST("../datasets/MNIST", train=True, download=True, transform=transform)  # 28 * 28
    seller_dataset = torchvision.datasets.CIFAR100("./datasets/CIFAR100_data", train=True, download=False, transform=transform)
elif args.dataset == "tinyimagenet":
    config.set_dic({
    "num_seller": 5,
    "num_buyer": 5,
    "dataset": "tinyimagenet",
    "seller_data_distribution": "split",
    "slabel": [],
    "opt":"Adam",
    "margin_rate": 0.2,
    "step_size": 0,
    "lr":0.005,
    "seller_labels": {0: [0,1], 1: [2,3], 2: [4,5], 3: [6,7], 4: [8,9]},
    "max_epoch": 30,
    "max_value": 100,
    "c": 0.01
    })
elif args.dataset == "MNIST":
    config.set_dic({
    "num_seller": 5,
    "num_buyer": 5,
    "dataset": "MNIST",
    "seller_data_distribution": "split",
    "slabel": [],
    "opt":"Adam",
    "margin_rate": 0.2,
    "step_size": 0,
    "lr":0.001,
    "seller_labels": {0: [0,1], 1: [2,3], 2: [4,5], 3: [6,7], 4: [8,9]},
    "max_epoch": 30,
    "max_value": 100,
    "c": 0.01
    })        
        
#load seller data


data = seller_dataset.data
targets = seller_dataset.targets

# 按标签分组数据, only run one time 
# data_groups = {}
# for i in range(len(data)):
#     label = targets[i]
#     if label not in data_groups:
#         data_groups[label] = []
#     data_groups[label].append(i)

# label distributiion for each dataset for seperating data to stimulate the data sellers, direclet distribution
if args.dataset == "CIFAR10":
    with open("./datasets/CIFAR10_data/train_data_labels.json", "r") as f:
        data_label_groups = json.load(f)
elif args.dataset == "CIFAR100":
    with open("./datasets/CIFAR100_data/train_data_labels.json", "r") as f:
        data_label_groups = json.load(f)
elif args.dataset == "MNIST":
    with open("./datasets/MNIST/train_data_labels.json", "r") as f:
        data_label_groups = json.load(f)
elif args.dataset == "tinyimagenet":
    with open("./datasets/TinyImageNet/train_data_labels.json", "r") as f:
        data_label_groups = json.load(f)        

seller_dataset_dic = {}
slabel = {} #slabel = {"0":[0,1]}

if cfg.get("slabel") == []:
    for s in range(cfg.get("num_seller")):
        slabel[s] = cfg.get("seller_labels")[s] #test
else:
    slabel = cfg.get("slabel")
#print(data_label_groups,slabel)
seller_dataset_dic = split_dataset_by_given_seller_labels(seller_dataset,data_label_groups,slabel,args)
length_slist = [len(seller_dataset_dic[i]) for i in seller_dataset_dic.keys()]
print(length_slist)

# for label, group_data in data_label_groups.items():
#     custom_datasets[label] = SellerDataset(group_data, transform=transform)


#initilize buyer, seller and arbiter
s_list = []
b_list = []

for s in range(cfg.get("num_seller")):
    s_list.append(Seller(seller_dataset,seller_dataset_dic[s], utility_history = []))

for b in range(cfg.get("num_buyer")):
    b_list.append(Buyer(buyer_query[args.dataset][str(b)], validation = buyer_query[args.dataset][str(b)]["val"] ,utility_history = []))

arbiter = Arbiter(cfg.get("margin_rate"), utility_history = [])
# initilize pricing function
print(len(s_list[0].data)) # s_list[i].data[n][0] data 3x32x32,  [1] label, all data are sort by label [0-4999]:0, [5000-10000]:1

#-------------------------- quality function test ----------------------------------
# feature_score,_,_,_ = count_feature_mean_hog(s_list[0].data)
# print("\nThe feature score of dataset is: " + str(feature_score))

# 1.2 using entropy to get f(x,y) distribution before training process
# bt_xy = cal_entropy_eva(s_list[0].data)
# print("\nThe joint score of dataset before training is: " + str(bt_xy))

# 1.3 using balance of label to get f(y) distribution before trainning process
# label_score,_ = count_label_balance(s_list[0].data)
# print("\nThe label score of dataset is: " + str(label_score))



b_utility = [0 * cfg.get("num_buyer")]
s_utility = [0 * cfg.get("num_seller")]
a_utility = 0

mean_b_utility_list = [] 
mean_s_utility_list = []
mean_a_utility_list = []
var_b_utility_list = []
var_s_utility_list = []
var_a_utility_list = []
pi_list = [[] * args.T]
agents = ['sellers','buyers','arbiter','anchor']
#agents = ['sellers','buyers','arbiter']
B = [1/3, 1/3, 1/3]
for t in range(args.T):
    print(f"----------------Iteration: {t}------------------")
    pi_list[t] =  MLPRegressor(input_size=1, hidden_size1= 64, hidden_size2= 128).to(device) 
    pi_list[t].load_state_dict(torch.load('init_price.pth'))
    opt, schedulers = get_opt(pi_list[t], cfg, device)
    output_weight = pd.DataFrame(columns=agents) # num_agents
    output_scale = pd.DataFrame(columns=agents)
    output_utility = pd.DataFrame(columns=agents)
    
    # locate data and compute quality score and initial price
    quality_score = []
    querys = {} 
    nums = {} 
    b_labels = {}
    b_label_s = {}
    b_data_s = {}
    values = {}
    w_list = []
    prices = []
    querytime = 0
    scoretime = 0
    start_time = time.time()
    feature_scores= []
    label_scores = []
    for b in range(cfg.get("num_buyer")):
        b_start_time = time.time()
        querys[b] = buyer_query[args.dataset][str(b)]["query"]
        b_labels[b] = buyer_query[args.dataset][str(b)]["label"]
        nums[b] = buyer_query[args.dataset][str(b)]["num"]
        values[b] = buyer_query[args.dataset][str(b)]["val"]
       
        b_label_s[b] = find_data(slabel,b_labels[b]) # return a dic, ex: {0: {0,4}, 1: {1}} {query_label:set{seller_id}}

        count = 0
        for label in b_labels[b]:
            seller_num = {}
            seller_list = b_label_s[b][label] 
            for s in seller_list:
                if (nums[b]%len(b_labels[b])) / len(seller_list) == 0 and count == 0:
                    num_s = int(nums[b]/len(b_labels[b])/len(seller_list))
                    seller_num[s] = num_s
                else:
                   count = (nums[b]%len(b_labels[b])) / len(seller_list)
                   num_s = int(nums[b]/len(b_labels[b])/len(seller_list))
                   seller_num[s] = num_s 
            #     print(f"{label}-{s}:{seller_num[s]}")   

                       
            # print(f"{label}-{s}:{seller_num}")       

            b_label_s[b][label] = seller_num   
        for label in b_labels[b]:
            seller_list = b_label_s[b][label]
            for s in seller_list:
                b_label_s[b][label][s] += int(count)  
                print(f"{label}-{s}:{b_label_s[b]}")  
                break
            break
        
        # input()    
        
        print(b_label_s[b]) 
          
        query_dataset = get_query_dataset(s_list,b_label_s[b],args)
        query_time = time.time()
        querytime += query_time-b_start_time
        
        #compute quality score
        # feature_score,_,_,_ = count_feature_mean_hog(query_dataset)
        label_score,_ = count_label_balance(query_dataset)
        #feature_scores.append(feature_score)
        label_scores.append(label_score)
        score_time = time.time()
        scoretime += score_time-query_time
        
        with torch.no_grad():
            quality_score.append((0.05 * label_score + 0.95 * nums[b]/len(seller_dataset))*100)   # 0.18779465834252318
            print(f"Buyer {b}: quality score is {quality_score[b]}, including feature score , label score {label_score}")    
    quality_score = torch.Tensor([quality_score]).to(device).reshape([len(quality_score),1]).requires_grad_()
    # if args.alogrithm == "FQora":
    #     quality_score = torch.Tensor([quality_score]).to(device).reshape([len(quality_score),1])
    #     prices = pi_list[t](quality_score)
    # elif args.alogrithm == "linear":
    #     prices = quality_score/100 * cfg.get("max_value")

    mid_time = time.time()
    pricingtime = 0
    for epoch in tqdm(range(cfg.get('max_epoch')),unit="epoch"):
        print(f"*******************epoch: {epoch}*******************")
        opt.zero_grad()
        t0 = time.time()
        pi_list[t].train()
        
        d = {agent: 0 for agent in agents}
        output_w = []
        output_s = []
        utility_l = []
        grads = {}
        utilities = {}
        utilities_d = {}
        # quality_score_c = quality_score.clone().requires_grad_()
        # if args.alogrithm == "FQora":
        #     prices_c = pi_list[t](quality_score_c)
        # elif args.alogrithm == "linear":
        #     prices_c = quality_score/100 * cfg.get("max_value")

        for agent in agents:
            if agent == "sellers":
                
                if args.alogrithm == "FQora":
                    
                    prices = pi_list[t](quality_score)
                    print(f"prices is {prices}")
                elif args.alogrithm == "linear":
                    prices = quality_score/100 * cfg.get("max_value")

                utility_function = SellerUtility()
                utilities[agent] = utility_function(prices,values,cfg)
                print(f"utility of {agent} is {utilities[agent]*-1000}")
                # opt.zero_grad()
                # utilities[agent].backward(retain_graph=True)
            elif agent == "buyers":
                if args.alogrithm == "FQora":
                    #quality_score = torch.Tensor([quality_score]).to(device).reshape([len(quality_score),1]).requires_grad_()
                    prices = pi_list[t](quality_score)
                elif args.alogrithm == "linear":
                    prices = quality_score/100 * cfg.get("max_value")
                utility_function = BuyerUtility()
                utilities[agent] = utility_function(prices,values,cfg)
                print(f"utility of {agent} is {utilities[agent]*-1000}")
                # opt.zero_grad()
                # utilities[agent].backward(retain_graph=True)
            elif agent == "arbiter":
                if args.alogrithm == "FQora":
                    #quality_score = torch.Tensor([quality_score]).to(device).reshape([len(quality_score),1]).requires_grad_()
                    prices = pi_list[t](quality_score)
                elif args.alogrithm == "linear":
                    prices = quality_score/100 * cfg.get("max_value")
                utility_function = ArbiterUtility()
                utilities[agent] = utility_function(prices,values,cfg)
                print(f"utility of {agent} is {utilities[agent]*(-100)}")
                # opt.zero_grad()
                # utilities[agent].backward(retain_graph=True)
            elif agent == "anchor":
                if args.alogrithm == "FQora":
                    #quality_score = torch.Tensor([qua1lity_score]).to(device).reshape([len(quality_score),1]).requires_grad_()
                    prices = pi_list[t](quality_score)
                elif args.alogrithm == "linear":
                    prices = quality_score/100 * cfg.get("max_value")
                utility_function = nn.SmoothL1Loss()
                if args.dataset == "CIFAR10":
                    refer_prices = align_score2anchorprice(quality_score,price_anchor_point_cifar10).to(device).reshape([5,1])
                elif args.dataset == "CIFAR100":
                    refer_prices = align_score2anchorprice(quality_score,price_anchor_point_cifar100).to(device).reshape([5,1])
                utilities[agent] = utility_function(prices,refer_prices).unsqueeze(0)/50
                print(f"utility of {agent} is {utilities[agent]*50}")
                # opt.zero_grad()
                # utilities[agent].backward(retain_graph=True)
            
            # with torch.no_grad():
            #     grads[agent] = [0.1 * quality_score.grad.data.clone().requires_grad_(False)]  #grad？
            #     print(f"{agent} grad:{grads[agent]}")  
        # for agent in agents:
        #     utilities_d[agent] = utilities[agent].data        
        # Normalize all gradients, this is optional and not included in the paper.
        # gn = gradient_normalizers(grads, utilities_d, 'loss+')
        # grads = {t: grads[t][0] / gn[t] for t in grads}
        # # Frank-Wolfe iteration to compute scales.
        # sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in agents])
        # scales = {t: sol[i] for i, t in enumerate(agents)}
        # # scales = {t : 1/len(agents)  for t in agents}       
        # print(f"scales:{scales}")
        # losses_ = {}
        # eva_losses_ = {}
        
        #opt.zero_grad() 
        # for agent in agents:
        #     utility = 0
        #     if epoch == 0:
        #         output_s.append(scales[agent])
        #     if agent == "sellers":
        #         if args.alogrithm == "FQora":
        #             #quality_score = torch.Tensor([quality_score]).to(device).reshape([len(quality_score),1]).requires_grad_()
        #             prices = pi_list[t](quality_score)
        #         elif args.alogrithm == "linear":
        #             prices = quality_score/100 * cfg.get("max_value")
        #         utility_function = SellerUtility()
        #         utility = utility_function(prices,values,cfg)
        #     elif agent == "buyers":
        #         if args.alogrithm == "FQora":
        #             #quality_score = torch.Tensor([quality_score]).to(device).reshape([len(quality_score),1]).requires_grad_()
        #             prices = pi_list[t](quality_score)
        #         elif args.alogrithm == "linear":
        #             prices = quality_score/100 * cfg.get("max_value")        
        #         utility_function = BuyerUtility()
        #         utility = utility_function(prices,values,cfg)
            # elif agent == "arbiter":
            #     if args.alogrithm == "FQora":
            #         #quality_score = torch.Tensor([quality_score]).to(device).reshape([len(quality_score),1]).requires_grad_()
            #         prices = pi_list[t](quality_score)
            #     elif args.alogrithm == "linear":
            #         prices = quality_score/100 * cfg.get("max_value")    
            #     utility_function = ArbiterUtility()
            #     utility = utility_function(prices,values,cfg)
            # elif agent == "anchor":
            #     if args.alogrithm == "FQora":
            #         #quality_score = torch.Tensor([quality_score]).to(device).reshape([len(quality_score),1]).requires_grad_()
            #         prices = pi_list[t](quality_score)
            #     elif args.alogrithm == "linear":
            #         prices = quality_score/100 * cfg.get("max_value")
            #     utility_function = nn.MSELoss()
            #     refer_prices = align_score2anchorprice(quality_score,price_anchor_point).to(device).reshape([5,1])
            #     utility = utility_function(prices,refer_prices).unsqueeze(0)    
            #losses_[agent] = utility


        ls_ = torch.stack([utilities[agent] for agent in agents])
        price_parameters = get_parameters(pi_list[t])
        #print(f"price parameters:{price_parameters}")
        # weight method
        bagrad = BAGrad(cfg, n_agents=len(agents),  device=device)

        w = bagrad.backward(
            losses=ls_,
            price_parameters=price_parameters,
            # task_specific_parameters=task_specific_parameters,
            # last_price_parameters=list(model.last_price_parameters()),
            # representation=features,
        )
    
        
        for i, agent in enumerate(agents):
            if i < len(agents) - 2:
                utility_l.append(utilities[agent].cpu().tolist()[0]*-1000)
            elif i == len(agents) - 2:
                utility_l.append(utilities[agent].cpu().tolist()[0]*-100)    
            else:
                utility_l.append(utilities[agent].cpu().tolist()[0]*50)    
        output_weight.loc[epoch] = w
        output_utility.loc[epoch] = utility_l
        opt.step()
        # print(output_weight)
        #output_weight.to_csv(os.path.join(f"./results/{args.dataset}/balanced_weight_{cfg.get('c')}_100.csv"))
        output_utility.to_csv(os.path.join(f"./results/{args.dataset}/balanced_utility_{cfg.get('c')}_100.csv"))
        # pi_list[t].eval()
        # for agent in agents:
        #     utility = 0
        #     #if epoch == 0:
        #         # output_s.append(scales[agent])
        #     if agent == "sellers":
        #         if args.alogrithm == "FQora":
        #             #quality_score = torch.Tensor([quality_score]).to(device).reshape([len(quality_score),1]).requires_grad_()
        #             prices = pi_list[t](quality_score)
        #         elif args.alogrithm == "linear":
        #             prices = quality_score/100 * cfg.get("max_value")
        #         utility_function = SellerUtility()
        #         utility = utility_function(prices,values,cfg)
        #     elif agent == "buyers":
        #         if args.alogrithm == "FQora":
        #             #quality_score = torch.Tensor([quality_score]).to(device).reshape([len(quality_score),1]).requires_grad_()
        #             prices = pi_list[t](quality_score)
        #         elif args.alogrithm == "linear":
        #             prices = quality_score/100 * cfg.get("max_value")        
        #         utility_function = BuyerUtility()
        #         utility = utility_function(prices,values,cfg)
            # elif agent == "arbiter":
            #     if args.alogrithm == "FQora":
            #         #quality_score = torch.Tensor([quality_score]).to(device).reshape([len(quality_score),1]).requires_grad_()
            #         prices = pi_list[t](quality_score)
            #     elif args.alogrithm == "linear":
            #         prices = quality_score/100 * cfg.get("max_value")    
            #     utility_function = ArbiterUtility()
            #     utility = utility_function(prices,values,cfg)
            # elif agent == "anchor":
            #     if args.alogrithm == "FQora":
            #         #quality_score = torch.Tensor([quality_score]).to(device).reshape([len(quality_score),1]).requires_grad_()

            #         prices = pi_list[t](quality_score)
            #     elif args.alogrithm == "linear":
            #         prices = quality_score/100 * cfg.get("max_value")
            #     utility_function = nn.MSELoss()
            #     refer_prices = align_score2anchorprice(quality_score,price_anchor_point).to(device).reshape([5,1])
            #     utility = utility_function(prices,refer_prices).unsqueeze(0)     
            # eva_losses_[agent] = utility
            
        # print(f"Losses: {eva_losses_}")
        
        

        output_w = list(w)
        print(f"Weights: {output_w}")
    end_time = time.time()
    pricingtime += end_time-mid_time    
    print(f"query time:{querytime}")
    print(f"score time:{scoretime}")
    print(f"pricing time:{pricingtime}")
# torch.save(pi_list[0],os.path.join(f'./results/{args.dataset}/pricing_function.pth'))        
        #compute mean and variance of utility    
        # for b in range(cfg.get("num_buyer")):
        #     b_utility[b] = b_list[b].utility_history[t]
        # print("Round {}: Buyer Mean utility is {}".format(t,mean(b_utility)))
        # mean_b_utility_list.append(mean(b_utility))
        # var_b_utility_list.append(variance(b_utility)) 
        
        
        # for s in range(cfg.get("num_seller")):
        #     s_utility[s] = s_list[s].utility_history[t]
        # print("Round {}: Seller Mean utility is {}".format(t,mean(s_utility)))
        # var_s_utility_list.append(variance(s_utility))
        
        # mean_s_utility_list.append(mean(s_utility))    
        # var_s_utility_list.append(variance(s_utility))
        
        
        # mean_b_utility = mean(b_utility)
        # var_b_utility = var_b_utility / cfg.get("num_buyer")
        
        # for s in range(cfg.get("num_seller")):
        #     s_utility[s] = s_list[s].utility_history[t]
        # mean_b_utility = mean()
    
    