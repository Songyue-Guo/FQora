import math
import cv2
from collections import defaultdict
import torch
import torchvision
import numpy as np
from skimage import feature
from scipy.stats import entropy
import math
from collections import Counter





def count_elements_2darray(arr):
    counts = defaultdict(int)
    for row in arr:
        for num in row:
            counts[str(int(num))] += 1
    return counts


class Hog_descriptor():
    def __init__(self, image_tensor, cell_size=8, bin_size=9):
        image_np = image_tensor.permute(1, 2, 0).mul(255).byte().numpy()  # 转换为NumPy数组并调整通道顺序和数据范围
        self.img = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).astype("uint8")
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size #对于360的迷惑

    def extract(self):
        height, width = self.img.shape
        #1.计算每个像素的梯度和方向
        gradient_magnitude, gradient_angle = self._global_gradient()
        gradient_magnitude = abs(gradient_magnitude)

        cell_gradient_vector = np.zeros((height // self.cell_size, width // self.cell_size, self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                #取一个cell中的梯度大小和方向
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]

                #得到每一个cell的梯度直方图;
                cell_gradient_vector[i][j] = self._cell_gradient(cell_magnitude, cell_angle)

        #得到HOG特征可视化图像，并转换为tensor
        hog_image = self._render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_image_tensor = torch.from_numpy(hog_image).unsqueeze(0)

        #HOG特征向量
        hog_vector = []
        #使用滑动窗口
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                #4个cell得到一个block
                block_vector = cell_gradient_vector[i:i+1][j:j+1].reshape(-1, 1)#串联
                #正则化
                block_vector = np.array([vector / (np.linalg.norm(vector)+1e-10) for vector in block_vector])
                hog_vector.append(block_vector)

        return hog_vector, hog_image_tensor

    def _global_gradient(self):
        #得到每个像素的梯度
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)#水平
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)#垂直
        gradient_magnitude = np.sqrt(gradient_values_x**2 + gradient_values_y**2)#总
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)#方向
        return gradient_magnitude, gradient_angle

    def _cell_gradient(self, cell_magnitude, cell_angle):
        #得到cell的梯度直方图
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self._get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def _get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        #return :起始bin索引，终止bin的索引，end索引对应bin所占权重
        return idx, (idx + 1) % self.bin_size, mod

    def _render_gradient(self, image, cell_gradient):
        #得到HOG特征图
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag #归一化
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    # 转换为弧度
                    angle_radian = math.radians(angle)
                    # 计算起始坐标和终点坐标，长度为幅值(归一化),幅值越大、绘制的线条越长、越亮
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image


def count_feature_mean_hog(dataset):
    
    """
    The dataset passed in here is an iterable of the tensor structure
    """
    
    hog_score, hog_maps = [], []
    total_samples = len(dataset)
    '这里的问题是index out of range (3-5)'
    for i in range(total_samples):
        # print(i)
        hog_extractor = Hog_descriptor(dataset[i][0], cell_size=8, bin_size=9)
        vector, hog_image = hog_extractor.extract()
        hog_image = hog_image.squeeze().numpy()
        # the type of hog_image is <class 'numpy.ndarray'>
        hog_count =  count_elements_2darray(hog_image)
        cover_map_score = 1 - hog_count["0"] / sum(hog_count.values())

        hog_score.append(cover_map_score)
        hog_maps.append(hog_image)
    
    max_hog_index = hog_score.index(max(hog_score))
    min_hog_index = hog_score.index(min(hog_score))

    total_hog = sum(hog_maps)
    total_hog_count =  count_elements_2darray(total_hog)
    cover_map_score = 1 - total_hog_count["0"] / sum(total_hog_count.values())

    # 寻找hog值的最大值和最小值
    # 1、hog高的图和hog低的图是有差异的，可以用于可视化，高低指标的sample在视觉上应该有差异；2、数值在数据集上的表现
    return cover_map_score ** 0.5, hog_score, [dataset[max_hog_index][0], hog_maps[max_hog_index]], [dataset[min_hog_index][0], hog_maps[min_hog_index]]

def cal_feature_info_entropy(dataset):
    '''
    Calculate the image entropy to express the richness of information for an image
    '''
    #create a dict, the value type is list
    hashmap = defaultdict(list)
    #derive the batch size
    batch_size = len(dataset)
    #iterate over the whole batch
    for i in range(batch_size):
        #labels[i] is tensor form, translate it to numpy form
        tmp_label = dataset[i][1]
        #Turn <class 'PIL.Image.Image'> to <class numpy>
        if dataset[0][0].size()[0] == 3:
            tmp = np.transpose(dataset[i][0], (1,2,0))
            gray_image =  np.dot(np.array(tmp), np.array([0.299, 0.587, 0.114]))
        else:
            gray_image = np.array(dataset[i][0])
        #summary the frequency of image
        _bins = 128
        hist, _ = np.histogram(gray_image.ravel(), bins=_bins, range=(0, _bins))
        prob_dist = hist / hist.sum()
        #use log2 base to calculate its entropy
        image_entropy = entropy(prob_dist, base=2)
        hashmap[tmp_label].append(image_entropy) #hashmap is a dict, its key is label and value is its image_entropy
    
    return hashmap

def cal_entropy_eva(dataset):
    '''
    Calculate the evaluation of entropy
    '''
    hashmap = cal_feature_info_entropy(dataset)
    entropy_mean = {}
    entropy_var = {}
    res = {}
    for key in hashmap:
        entropy_mean[key] = np.mean(np.array(hashmap[key]))
        entropy_var[key] = np.var(np.array(hashmap[key]))

    tp = [math.exp(value) for value in entropy_var.values()]
    sum_evar = sum(tp)
    tp = np.array([value for value in entropy_mean.values()])
    mean2 = np.mean(tp)
    mean_max = np.max(tp)
    mean_min = np.min(tp)
    for key in entropy_mean:
        res[key]  = (math.exp(entropy_var[key]) / (sum_evar + 1e-8)) * ((np.abs(entropy_mean[key]-mean2)) / (mean_max-mean_min+1e-8))

    tmp_labels = np.array([dataset[i][1] for i in range(len(dataset))])
    label_counts = dict(Counter(tmp_labels))
    size = sum(label_counts.values())
    for key in label_counts:
        label_counts[key] = label_counts[key]/size
    
    res2 = 0
    for key in res:
        res2 += res[key]*label_counts[key]

    return res2

def count_label_balance(dataset):
    """
    This is used to calculate labence of dataset labels.
    The size of y_ture is [batch_size, class_num].
    """
    y_true = [dataset[i][1] for i in range(len(dataset))]
    hashmap = Counter(y_true)
   
    total_num, cls_num,  = len(y_true), len(hashmap)
    print(f'hashmap: {hashmap},total_num: {total_num},label_num: {cls_num}')
    balance_rate = 1 / cls_num
    res = 0
    label_ratio = []
    for key in hashmap:
        cur_rate = hashmap[key] / total_num
        diff = cur_rate - balance_rate
        res += max(diff / (1 - balance_rate), 0)
        label_ratio.append(cur_rate)

    return (1 - res) ** 2, label_ratio # using square function to to increase sensitivity to high balance （可视化方法：显示比例）


if __name__ == "__main__":
    # 图像平均HOG值计算
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # train_dataset = torchvision.datasets.MNIST("../datasets/MNIST", train=True, download=True, transform=transform)  # 28 * 28
    train_dataset = torchvision.datasets.CIFAR10("../datasets/CIFAR10_data", train=True, download=True, transform=transform)
    # print(dataset[0][0].size()[0], type(dataset[0][0][0]), len(dataset))

    # 1.1 using HOG to get f(x) distribution before training process
    bt_x = count_feature_mean_hog(train_dataset)
    print("\nThe feature score of dataset before training is: " + str(bt_x))

    # 1.2 using entropy to get f(x,y) distribution before training process
    bt_xy = cal_entropy_eva(train_dataset)
    print("\nThe joint score of dataset before training is: " + str(bt_xy))

    # 1.3 using balance of label to get f(y) distribution before trainning process
    bt_y = count_label_balance(train_dataset)
    print("\nThe label score of dataset before training is: " + str(bt_y))