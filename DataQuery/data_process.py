from torch.utils.data import Subset
from collections import defaultdict

def split_dataset_by_given_seller_labels(dataset,label_group,slabel,args):
    """
    按照数据的标签不同来划分不同的数据集。

    :param dataset: 要切分的 PyTorch 数据集，其中包含有标签的样本。
    :return: 一个字典，键为标签，值为对应标签的数据集子集。
    """
    if args.dataset == 'CIFAR10':
        s_label_indices = defaultdict(list)
        # 假设每个样本是一个元组，其中第二个元素是标签
        print(slabel)
        for s in range(len(slabel.keys())):
            for l in slabel[s]:
                s_label_indices[s].extend(label_group[str(l)])
        # image: label:subset[i][0], label:subset[i][1]
        subsets = {s: Subset(dataset, indices) for s, indices in s_label_indices.items()}
    elif args.dataset == 'CIFAR100':
        s_label_indices = defaultdict(list)
        # 假设每个样本是一个元组，其中第二个元素是标签
        print(slabel)
        for s in slabel.keys():
            for l in slabel[s]:
                for i in range(5):
                    s_label_indices[s].extend(label_group[str(l*5+i)])
        # image: label:subset[i][0], label:subset[i][1]
        subsets = {s: Subset(dataset, indices) for s, indices in s_label_indices.items()}
    elif args.dataset == 'tinyimagenet':
        s_label_indices = defaultdict(list)
        # 假设每个样本是一个元组，其中第二个元素是标签
        print(slabel)
        for s in slabel.keys():
            for l in slabel[s]:
                for i in range(5):
                    s_label_indices[s].extend(label_group[str(l+i)])
        # image: label:subset[i][0], label:subset[i][1]
        subsets = {s: Subset(dataset, indices) for s, indices in s_label_indices.items()}    
    return subsets


