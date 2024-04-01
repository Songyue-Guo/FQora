import torch.nn as nn 
from DataQuery.seller_dataset import SellerDataset
# seller
class Seller:
     def __init__(self, dataset, data_index, utility_history):
         self.data = SellerDataset(dataset,data_index)
         self.utility_history = utility_history
     def load_data(self, data):
        """
        :param data: 要加载的数据
        """
        self.data = data
     def update_utility_history(self, utility):
        """
        :param utility
        """
        self.utility_history.append(utility)
        
class SellerUtility(nn.Module):
    def __init__(self):
        super(SellerUtility,self).__init__()
        
    def forward(self,prices,values,cfg):
        utility = 0
        num_seller = cfg.get('num_seller')
        num_buyer = cfg.get('num_buyer')
        for i in range(num_buyer):
            if prices[i] <= values[i] :
                utility += prices[i]
        utility =  utility * (1 - cfg.get('margin_rate'))
        utility = - utility     
        return utility/1000
            
class SellerAllocationUtility(nn.Module):
    def __init__(self):
        super(SellerAllocationUtility,self).__init__()
        
    def forward(self,prices,values,cfg):
        utility = 0
        num_seller = cfg.get('num_seller')
        num_buyer = cfg.get('num_buyer')
        for i in range(num_buyer):
            if prices[i] <= values[i]:
                utility += prices[i]
        return - utility * (1 - cfg.get('margin_rate'))   
         
class ArbiterUtility(nn.Module):
    def __init__(self):
        super(ArbiterUtility,self).__init__()
        
    def forward(self,prices,values,cfg):
        utility = 0
        num_seller = cfg.get('num_seller')
        num_buyer = cfg.get('num_buyer')
        for i in range(num_buyer):
            if prices[i] <= values[i]:
                utility += prices[i]
        return - utility *  cfg.get('margin_rate')/100     
        

class Buyer:
     def __init__(self, query,validation, utility_history):
         self.validation = validation
         self.query = query
         self.utility_history = utility_history

     def update_query(self, query):
        """
        :param query
        """
        self.query = query     
     def update_utility_history(self, utility):
         self.utility_history.append(utility)
         
class BuyerUtility(nn.Module):
    def __init__(self):
        super(BuyerUtility,self).__init__()
        
    def forward(self,prices,values,cfg):
        utility = 0
        num_buyer = cfg.get('num_buyer')
        for i in range(num_buyer):
            if prices[i] <= values[i]:
                utility += values[i] - prices[i]
        return - utility/1000
    
class BuyerAllocationUtility(nn.Module):
    def __init__(self):
        super(BuyerAllocationUtility,self).__init__()
        
    def forward(self,prices,values):
        utility = 0

        for i in range(len(prices)):
            if prices[i] <= values[i]:
                utility += values[i] - prices[i]
        return - utility 
    
# Arbiter         
class Arbiter:
     def __init__(self, pricing_function, utility_history):
         self.pricing_function = pricing_function
         self.utility_history = utility_history
     def update_pf(self, pricing_function):
        """
        更新 pricing function 的方法
        :param pf
        """
        self.pricing_function = pricing_function     
     def update_utility_history(self, utility):
         self.utility_history.append(utility)
                           