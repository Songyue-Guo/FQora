import os
import argparse
from tqdm import tqdm
import time
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.nn.functional as F
from utils.min_norm_solvers import MinNormSolver, gradient_normalizers
import numpy
from utils.util import timing, get_model
from tools.config import Config
import pandas as pd
from utils.methods import CAGrad, MCGrad, balancedGrad
from collections import defaultdict
import torchtext
import time

parser = argparse.ArgumentParser()
parser.add_argument('-d', default='sentiment')  
parser.add_argument('-m', default='textcnn')    
parser.add_argument('-t', default='bandit')     # algorithm
parser.add_argument('-g', default='0')          # GPU number
parser.add_argument('-c', default='')           # remark
parser.add_argument('-alpha', default='0.1')
parser.add_argument('-split_scale', default='0.1')
args = parser.parse_args()

#os.environ['CUDA_VISIBLE_DEVICES'] = args.g
device = torch.device('cuda:'+str(args.g) if torch.cuda.is_available() else 'cpu')
cfg = Config(data=args.d, model=args.m, agent=args.t)    # model config
agents = cfg.data['agents'] 
criterion = torch.nn.CrossEntropyLoss() 
print_result = pd.DataFrame(columns=agents+['avg_acc'])
print_loss = pd.DataFrame(columns=agents+['var'])

if __name__ == '__main__':
    from data.dataset import NewsDataset, SentimentDataset
    if args.d == '20news':
        loader = NewsDataset(cfg)
    elif args.d == 'sentiment':
        loader = SentimentDataset(cfg)

    max_epoch = cfg.exp['epochs']
    cfg_exp_str = '_'.join([k + str(cfg.exp[k]) for k in cfg.exp])
    exp_id = f'{time.strftime("%H%M", time.localtime())}_{cfg_exp_str}_{args.m}_{args.c}'
    print(f'# exp_id: {exp_id}')
    save_path = f'./runs/{args.d}/{args.t}/{time.strftime("%m%d")}/{exp_id}'
    writer = SummaryWriter(log_dir=save_path)           #tensorboard


@timing
def init_model():
    model = get_model(cfg, loader.word_embeddings)
    return model


def scheduler_step(schedulers):
    for s in schedulers:
        s.step()


def get_parameters(model):
    # for param in model['rep'].parameters():
    #     if param.grad is not None:
    #         grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))

    shared_grads = []
    for t, m in model.items():
        for p in m.parameters():
            if p.grad is not None:
                shared_grads.append(p)

    return shared_grads

# get optimizer
def get_opt(model, rep=False, dis=False):
    print(cfg.exp)
    #input()
    lr = cfg.exp['lr'] 
    params = {'all': []}
    for m_id, m in enumerate(model):
        model[m].to(device)
        if (m == 'rep' and rep) or (m == 'dis' and dis):
            params[m] = model[m].parameters()
        else:
            params['all'] += model[m].parameters() 
    opts = []
    schedulers = []
    for m in ['all', 'rep', 'dis']:
        if m in params:
            if 'opt' in cfg.exp and cfg.exp['opt'] == 'Adam':
                opt = torch.optim.Adam(params[m], lr=lr, weight_decay=cfg.exp['wd'] if 'wd' in cfg.exp else 0)
            else:
                opt = torch.optim.SGD(params[m], lr=lr, momentum=0.9)
                #schedulers.append(torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.exp['step'], gamma=0.9))
            opts.append(opt)
            
            #'''
            if 'step' in cfg.exp.keys() and cfg.exp['step'] == 0:
                schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=500, eta_min=0, last_epoch=-1
                    )
                )
            else:
                schedulers.append(
                    torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.exp['step'],
                                        gamma=0.8)
                                  )
            #'''
    return opts, schedulers


def balanced_multi_agent():
    n_batches = min([len(loader.train[t].dataset.examples) for t in agents]) // cfg.data['batch_size']
    model = init_model()
    [opt_c, opt_g], schedulers = get_opt(model, rep=True)
    output_weight = pd.DataFrame(columns=agents)
    output_scale = pd.DataFrame(columns=agents)
    t0 = time.time()
    for epoch in tqdm(range(max_epoch), unit='epoch'):
        # train
        for m in model:
            model[m].train()
        iters = {t: iter(loader.train[t]) for t in agents}
        d = {t: 0 for t in agents}
        output_w = []
        output_s = []
        for n in range(n_batches):
            grads = {}
            losses = {}
            xs = {}
            ys = {}
            opt_g.zero_grad()
            opt_c.zero_grad()
            for t in agents:
                batch = next(iters[t])
                d[t] += 1
                if batch.text.shape[1] != cfg.data['batch_size']:
                    batch = next(iters[t])
                    d[t] += 1
                x, y = batch.text.to(device), batch.label.to(device)
                xs[t] = x
                ys[t] = y

                # with torch.no_grad():
                #     rep = model['rep'](x)
                # rep = rep.clone().requires_grad_()
                y_ = model[t](model['rep'](x))
                loss_t = criterion(y_, y)
                losses[t] = loss_t
                # losses[t] = loss_t.data
                # opt_c.zero_grad()
                # loss.backward()
                # grads[t] = [rep.grad.data.clone().requires_grad_(False)]

            # Normalize all gradients, this is optional and not included in the paper.
            # gn = gradient_normalizers(grads, losses, 'loss+')
            # grads = {t: grads[t][0] / gn[t] for t in grads}
            # # Frank-Wolfe iteration to compute scales.
            # sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in agents])
            # scales = {t: sol[i] for i, t in enumerate(agents)}
            # losses_ = {}
            # opt_g.zero_grad()
            # opt_c.zero_grad()
            # for t in agents:
            #     if n == 0:
            #         output_s.append(scales[t])
            #     y_ = model[t](model['rep'](xs[t]))
            #     loss = criterion(y_, ys[t]) * scales[t]
            #     losses_[t] = loss
                # opt_g.zero_grad()
                # opt_c.zero_grad()
                # loss.backward()
                # opt_g.step()
                # opt_c.step()

            ls_ = torch.stack([losses[t] for t in agents])
            shared_parameters, agent_specific_parameters = get_parameters(model)

            # weight method
            mcgrad = MCGrad(cfg, n_agents=len(agents), device=device)

            w = mcgrad.backward(
                losses=ls_,
                shared_parameters=shared_parameters,
                # agent_specific_parameters=agent_specific_parameters,
                # last_shared_parameters=list(model.last_shared_parameters()),
                # representation=features,
            )
            # loss = None
            # for i, t in enumerate(agents):
            #     if n == 0:
            #         output_w = list(w)
            #     y_ = model[t](model['rep'](xs[t]))
            #     loss_t = criterion(y_, ys[t])
            #     if i > 0:
            #         loss += loss_t * w[i]
            #     else:
            #         loss = loss_t * w[i]
            # loss = loss / len(agents)
            # opt_g.zero_grad()
            # opt_c.zero_grad()
            # loss.backward()

            opt_g.step()
            opt_c.step()

            # opt_g.step()
            # opt_c.step()
            # schedulers[0].step()
            # schedulers[1].step()
            # print(f"w:{w}")
            # print(w[0])
            # print(w_i)

            if n == 0:
                output_w = list(w)

        output_weight.loc[epoch] = output_w
        # output_scale.loc[epoch] = output_s
        # output_scale.to_csv(os.path.join(save_path, "mgda_scale.csv"))
        output_weight.to_csv(os.path.join(save_path, "balanced_scale.csv"))
        # validation
        if epoch % 10 == 0:
            if epoch == 0:
                tot = time.time() - t0
                t1 = tot
            else:
                t1 = time.time() - t0 - tot
                tot = time.time() - t0
            print('-'*120)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                 'Time Taken: [{:.2f} sec]'
                 .format(t1))
            validate_all_agents(model, epoch)

        scheduler_step(schedulers)

    pass


def balancedmgda_multi_agent():
    n_batches = min([len(loader.train[t].dataset.examples) for t in agents]) // cfg.data['batch_size']
    model = init_model()
    [opt_c, opt_g], schedulers = get_opt(model, rep=True)
    output_weight = pd.DataFrame(columns=agents)
    output_scale = pd.DataFrame(columns=agents)
    t0 = time.time()
    for epoch in tqdm(range(max_epoch), unit='epoch'):
        # train
        for m in model:
            model[m].train()
        iters = {t: iter(loader.train[t]) for t in agents}
        d = {t: 0 for t in agents}
        output_w = []
        output_s = []
        for n in range(n_batches):
            grads = {}
            losses = {}
            xs = {}
            ys = {}
            for t in agents:
                batch = next(iters[t])
                d[t] += 1
                if batch.text.shape[1] != cfg.data['batch_size']:
                    batch = next(iters[t])
                    d[t] += 1
                x, y = batch.text.to(device), batch.label.to(device)
                xs[t] = x
                ys[t] = y

                with torch.no_grad():
                    rep = model['rep'](x)
                rep = rep.clone().requires_grad_()
                y_ = model[t](rep)
                loss = criterion(y_, y)
                losses[t] = loss.data
                opt_c.zero_grad()
                loss.backward()
                grads[t] = [rep.grad.data.clone().requires_grad_(False)]

            # Normalize all gradients, this is optional and not included in the paper.
            gn = gradient_normalizers(grads, losses, 'loss+')
            grads = {t: grads[t][0] / gn[t] for t in grads}
            # Frank-Wolfe iteration to compute scales.
            sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in agents])
            scales = {t: sol[i] for i, t in enumerate(agents)}
            losses_ = {}

            opt_g.zero_grad()
            opt_c.zero_grad()
            for t in agents:
                if n == 0:
                    output_s.append(scales[t])
                y_ = model[t](model['rep'](xs[t]))
                loss_t = criterion(y_, ys[t])
                losses_[t] = loss_t

            ls_ = torch.stack([losses_[t] for t in agents])
            shared_parameters, agent_specific_parameters = get_parameters(model)

            # weight method
            cagrad = balancedGrad(cfg, list(scales.values()), n_agents=len(agents), device=device)

            w = cagrad.backward(
                losses=ls_,
                shared_parameters=shared_parameters,
                # agent_specific_parameters=agent_specific_parameters,
                # last_shared_parameters=list(model.last_shared_parameters()),
                # representation=features,
            )
            opt_g.step()
            opt_c.step()

            if n  == 0:
                output_w = list(w)

        #output_scale.loc[epoch] = output_s
        #output_weight.loc[epoch] = output_w
        #output_scale.to_csv(os.path.join(save_path, "balancedmgda_scale.csv"))
        #output_weight.to_csv(os.path.join(save_path, "balancedmgda_weight.csv"))

        # validation
        if epoch % 10 == 0:
            if epoch == 0:
                tot = time.time() - t0
                t1 = tot
            else:
                t1 = time.time() - t0 - tot
                tot = time.time() - t0
            print('-'*120)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                 'Time Taken: [{:.2f} sec]'
                 .format(t1))
            #validate_all_agents(model, epoch)

        scheduler_step(schedulers)

    pass





def mgda_multi_agent():
    n_batches = min([len(loader.train[t].dataset.examples) for t in agents]) // cfg.data['batch_size']
    model = init_model()
    [opt_c, opt_g], schedulers = get_opt(model, rep=True)
    output_weight = pd.DataFrame(columns=agents)
    t0 = time.time()
    for epoch in tqdm(range(max_epoch), unit='epoch'):
        # train
        for m in model:
            model[m].train()
        iters = {t: iter(loader.train[t]) for t in agents}
        d = {t: 0 for t in agents}
        output_s = []
        output_w = []
        for n in range(n_batches): 
            grads = {}
            losses = {}
            xs = {}
            ys = {}
            for t in agents:
                batch = next(iters[t])
                d[t] += 1 
                if batch.text.shape[1] != cfg.data['batch_size']:
                    batch = next(iters[t])
                    d[t] += 1
                x, y = batch.text.to(device), batch.label.to(device)
                xs[t] = x
                ys[t] = y
                with torch.no_grad():
                    rep = model['rep'](x)
                rep = rep.clone().requires_grad_()
                y_ = model[t](rep)
                loss = criterion(y_, y)
                losses[t] = loss.data
                opt_c.zero_grad()
                loss.backward()
                grads[t] = [rep.grad.data.clone().requires_grad_(False)]
            # Normalize all gradients, this is optional and not included in the paper.
            gn = gradient_normalizers(grads, losses, 'loss+')
            grads = {t: grads[t][0] / gn[t] for t in grads}
            # Frank-Wolfe iteration to compute scales.
            sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in agents])
            scales = {t: sol[i] for i, t in enumerate(agents)}
            for t in agents:
                if n == 0:
                    output_s.append(scales[t])
                y_ = model[t](model['rep'](xs[t]))
                loss = criterion(y_, ys[t]) * scales[t]
                opt_g.zero_grad()
                opt_c.zero_grad()
                loss.backward()
                opt_g.step()
                opt_c.step()

        output_weight.loc[epoch] = output_s
        output_weight.to_csv(os.path.join(save_path,"mgda_scale.csv"))
        # validation
        if epoch % 10 == 0:
            if epoch == 0:
                tot = time.time() - t0
                t1 = tot
            else:
                t1 = time.time() - t0 - tot
                tot = time.time() - t0
            print('-'*120)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                 'Time Taken: [{:.2f} sec]'
                 .format(t1))
            validate_all_agents(model, epoch)

        scheduler_step(schedulers)



if __name__ == '__main__':
    elif args.t == 'mgda':
        mgda_multi_agent()
    elif args.t == 'mgdacag':
        mgdacag_multi_agent()
    elif args.t == 'balanced':
        balanced_multi_agent()
    elif args.t == 'balancedmgda':
        balancedmgda_multi_agent()


    print('exit')

