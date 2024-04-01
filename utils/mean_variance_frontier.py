def mean(values):
    
    return sum(values) / len(values) if values else 0

def variance(values):
    
    if not values:
        return 0

    mean_value = mean(values)
    return sum((x - mean_value) ** 2 for x in values) / len(values)

def covariance(list1, list2):
    
    if len(list1) != len(list2):
        raise ValueError("len of list is not equal")

    mean1 = mean(list1)
    mean2 = mean(list2)
    covar = sum((x - mean1) * (y - mean2) for x, y in zip(list1, list2)) / len(list1)
    return covar