import torchvision
from torch.utils.data import Dataset

class SellerDataset(Dataset):
    def __init__(self, dataset,data_index,transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])):
        """
        初始化数据集
        :param data_paths: 数据的路径列表
        :param transform: 应用于数据的转换操作（例如：标准化、数据增强等）
        """
        self.dataset = dataset
        self.data_index = data_index
        self.transform = transform
        

    def __len__(self):
        """返回数据集中的样本数"""
        return len(self.data_index)

    def __getitem__(self, idx):
        """
        根据索引 idx 返回一个样本
        :param idx: 样本的索引
        """

        # if self.transform:
        #     sample = self.transform(self.dataset[idx])
        # else:
        #     sample = self.dataset[idx]
        sample = self.data_index[idx]    
        return sample
    def __getitem__(self, slice):
        return self.data_index[slice]





    