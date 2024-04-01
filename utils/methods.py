import copy
import random
from abc import abstractmethod
from typing import Dict, List, Tuple, Union


import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize


class WeightMethod:
    def __init__(self, n_agents: int, device: torch.device):
        super().__init__()
        self.n_agents = n_agents
        self.device = device

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        price_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        # last_price_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        # representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        price_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        # last_price_parameters: Union[
        #     List[torch.nn.parameter.Parameter], torch.Tensor
        # ] = None,
        # representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """
        Parameters
        ----------
        losses :
        price_parameters :
        task_specific_parameters :
        last_price_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :
        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            price_parameters=price_parameters,
            task_specific_parameters=task_specific_parameters,
            # last_price_parameters=last_price_parameters,
            # representation=representation,
            **kwargs,
        )
        loss.backward()
        return loss, extra_outputs

    def __call__(
        self,
        losses: torch.Tensor,
        price_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            price_parameters=price_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []

class BAGrad(WeightMethod):
    def __init__(self, cfg, n_agents, device: torch.device):
        super().__init__(n_agents, device=device)
        self.cfg = cfg
        #self.c = self.cfg.get['c']
        self.c = 0.5
    def get_weighted_loss(
            self,
            losses,
            price_parameters,
            **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        price_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in price_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_agents).to(self.device)

        for i in range(self.n_agents):
            if i < self.n_agents:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(price_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in price_parameters:
                p.grad = None

        g, w = self.mcgrad(grads, alpha=self.c, rescale=1)
        self.overwrite_grad(price_parameters, g, grad_dims)

        return w

    def mcgrad(self, grads, alpha=0.15, rescale=1):
        GtG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GtG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.n_agents) / self.n_agents
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        
        A = GtG.numpy()
        print(f"A: {A}; go_norm: {g0_norm}")
        b = x_start.copy()
        cg0 = (alpha * g0_norm + 1e-8).item()
        cg02 = (alpha * g0_norm * g0_norm + 1e-8).item()
        rho = (
                1 - np.square(alpha * g0_norm)
        ).item()

        def gwobjfn(x):
            return np.sqrt(
                x.reshape(1, self.n_agents).dot(A).dot(x.reshape(self.n_agents, 1)) + 1e-8
            )

        def objfn(x):
            print(x.reshape(1, self.n_agents).dot(A).dot(x.reshape(self.n_agents, 1))
                                + 1e-8)
            return (
                    np.square(
                        (cg0 + 1) * np.sqrt(
                                x.reshape(1, self.n_agents).dot(A).dot(x.reshape(self.n_agents, 1))
                                + 1e-8
                        )
                ) / (2 * alpha * np.sqrt(x.reshape(1, self.n_agents).dot(A).dot(x.reshape(self.n_agents, 1))
                                        + 1e-8) * rho)
                    - np.sqrt(x.reshape(1, self.n_agents).dot(A).dot(x.reshape(self.n_agents, 1)) + 1e-8) / 2 * alpha
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmda = (gw_norm + 1e-8) / cg02
        d = grads.mean(1) / rho + gw / rho * lmda
        # lmbda = cg0 / (gw_norm + 1e-8)
        # d = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return d, w_cpu
        elif rescale == 1:
            return d / (1 + alpha ** 2), w_cpu
        else:
            return d / (1 + alpha), w_cpu

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, price_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_agents  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in price_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
            self,
            losses: torch.Tensor,
            parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
            price_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,

            **kwargs,
    ):
        w = self.get_weighted_loss(losses, price_parameters)
        return w  # NOTE: to align with all other weight methods

 
class balancedGrad(WeightMethod):
    def __init__(self, cfg, scales, n_agents, device: torch.device):
        super().__init__(n_agents, device=device)
        self.cfg = cfg
        self.c = self.cfg.get('c')
        self.s = torch.Tensor(scales)

    def get_weighted_loss(
            self,
            losses,
            price_parameters,
            **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        price_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in price_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_agents).to(self.device)

        for i in range(self.n_agents):
            if i < self.n_agents:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(price_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in price_parameters:
                p.grad = None

        g, w = self.mcgrad(grads, alpha=self.c, rescale=1)
        self.overwrite_grad(price_parameters, g, grad_dims)

        return w

    def mcgrad(self, grads, alpha=0.15, rescale=1):
        GtG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GtG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.n_agents) / self.n_agents
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}

        A = GtG.numpy()
        # print(f"A:{A}")
        b = x_start.copy()
        cg0 = (alpha * g0_norm + 1e-8).item()
        cg02 = (alpha * g0_norm * g0_norm + 1e-8).item()
        rho = (
                1 - np.square(alpha * g0_norm)
        ).item()

        def gwobjfn(x):
            return np.sqrt(
                x.reshape(1, self.n_agents).dot(A).dot(x.reshape(self.n_agents, 1)) + 1e-8
            )

        def objfn(x):
            # print(x.reshape(1, self.n_agents).dot(A).dot(x.reshape(self.n_agents, 1))
            #                     + 1e-8
            #             )
            
            return (
                    np.square(
                        (cg0 + 1) * np.sqrt(
                                x.reshape(1, self.n_agents).dot(A).dot(x.reshape(self.n_agents, 1))
                                + 1e-8
                        )
                ) / (2 * alpha * np.sqrt(x.reshape(1, self.n_agents).dot(A).dot(x.reshape(self.n_agents, 1))
                                        + 1e-8) * rho)
                    - np.sqrt(x.reshape(1, self.n_agents).dot(A).dot(x.reshape(self.n_agents, 1)) + 1e-8) / 2 * alpha
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        wv = torch.Tensor(w_cpu) * self.s
        ww = torch.nn.functional.softmax(wv, dim=0)
        ww = torch.Tensor(ww).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmda = (gw_norm + 1e-8) / cg02
        d = grads.mean(1) / rho + gw / rho * lmda
        # lmbda = cg0 / (gw_norm + 1e-8)
        # d = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return d, w_cpu
        elif rescale == 1:
            return d / (1 + alpha ** 2), w_cpu
        else:
            return d / (1 + alpha), w_cpu

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, price_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_agents  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in price_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
            self,
            losses: torch.Tensor,
            parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
            price_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            **kwargs,
    ):
        w = self.get_weighted_loss(losses, price_parameters)
        return w  # NOTE: to align with all other weight methods

