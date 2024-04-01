from time import time
import numpy as np
from torch.utils.data import Subset,ConcatDataset
import numpy as np
import torch
# from models.text_cnn import TextR, TextO, u_TextO, TextD
# from models.text_rnn import TextRNNR, TextRNNO, u_TextRNNO, TextRNND
# from models.fudan import Fudan

def timing(f):
    def wrapper(*args, **kwargs):
        t = time()
        r = f(*args, **kwargs)
        print(f'# Time {f.__name__}: {time() - t}')
        return r
    return wrapper

def find_data(slabel,blabel):
    candidate_seller_dic = {i: set() for i in blabel}
    # slabel = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9]}, blabel = [0,1,3]
    
    for j in range(len(slabel)):
        for k in range(len(slabel[j])):
            if slabel[j][k] in blabel:
                candidate_seller_dic[slabel[j][k]].add(j)
    return candidate_seller_dic


def get_query_dataset(s_list,b_label_s,args):
    if args.dataset == 'CIFAR10':
        count_slices = 5000
    elif args.dataset == 'CIFAR100':
        count_slices = 2500    
    elif args.dataset == 'MNIST':
        count_slices = 2500   
    label_index = {s: 0 for s in range(len(s_list))} # determine if query this seller
    query_data = None
    # b_label_s = {label in query:{seller_id, num}}
    for label in b_label_s.keys():
        
        seller_label_num = b_label_s[label] #{seller_id, num}
        for s in seller_label_num.keys():
            query = Subset(s_list[s].data,range(count_slices * label_index[s], count_slices * label_index[s] + seller_label_num[s]))
            if query_data is None:
                query_data = query
            else:
                query_data = ConcatDataset([query_data,query])
            label_index[s] += 1
    return query_data

def get_model(cfg, word_embeddings):
    # if cfg.task == 'fudan':
    #     model = Fudan(cfg, word_embeddings)
    # elif cfg.model.name == 'textcnn':
    #     model = dict({})
    #     model['rep'] = TextR(cfg, word_embeddings)
    #     for t in cfg.data['tasks']:
    #         if cfg.task == 'uncertain':
    #             model[t] = u_TextO(cfg)
    #         else:
    #             model[t] = TextO(cfg)
    #     if 'dis' in cfg.exp and cfg.exp['dis']:
    #         model['dis'] = TextD(cfg)
    # return model
    pass

def get_opt(model,cfg,device):

    lr = cfg.get('lr') 
    
    model.to(device)
    
    params = model.parameters()

    if cfg.get('opt') == 'Adam':
        opt = torch.optim.Adam(params, lr=lr, weight_decay= 0)
    else:
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
        #schedulers.append(torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.exp['step'], gamma=0.9))

        
        #'''
    if cfg.get('step_size') == 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( opt, T_max=500, eta_min=0, last_epoch=-1
            )
        
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.get('step_size'),
                                gamma=0.8)
                            
            #'''
    return opt, scheduler

def scheduler_step(schedulers):
    for s in schedulers:
        s.step()


def get_parameters(model):
    # for param in model['rep'].parameters():
    #     if param.grad is not None:
    #         grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))

    shared_grads = []
    for p in model.parameters():
        if p.grad is not None:
                shared_grads.append(p)

    return shared_grads

def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().item() for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().item() for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn

