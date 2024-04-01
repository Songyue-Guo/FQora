# pre-define buyer_query for each dataset

buyer_query = {
    'CIFAR10': {
    '0':{"query": "SELECT * FROM CIFAR10 WHERE (label = 0 OR label = 1) AND quality > 0.9 LIMIT 10000", "label" : [0,1,3] , "num": 10000, "quality": 0.9 ,"val": 40},
    '1':{"query": "SELECT * FROM CIFAR10 WHERE (label = 0 OR label = 2 OR label = 3 OR label = 5 OR label = 6) AND quality > 0.9 LIMIT 20000", "label" : [0,2,3,5,6] , "num": 20000, "quality": 0.9 ,"val": 100 },
    '2':{"query": "SELECT * FROM CIFAR10 WHERE (label = 0 OR label = 1 OR label = 2 OR label = 3 OR label = 4 OR label 5 OR label = 6) AND quality > 0.9 LIMIT 30000", "label" : [0,1,2,3,4,5,6] , "num": 30000, "quality": 0.9 ,"val": 130 },
    '3':{"query": "SELECT * FROM CIFAR10 WHERE (label = 1 OR label = 2 OR label = 3 OR label = 4 OR label 5 OR label = 6 OR label = 8 OR label = 9) AND quality > 0.9 LIMIT 35000", "label" : [1,2,3,4,5,6,8,9] , "num": 35000, "quality": 0.9 ,"val": 160 },
    '4':{"query": "SELECT * FROM CIFAR10 WHERE (label = 0 OR label = 1 OR label = 2 OR label = 3 OR label = 4 OR label 5 OR label = 6 OR label = 7 OR label = 8 OR label = 9) AND quality > 0.9 LIMIT 48000", "label" : [0,1,2,3,4,5,6,7,8,9] , "num": 40000, "quality": 0.9 ,"val": 170 },
    
    },
    # CIFAR100 labels coarse_labels = [
#     'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables','household electrical devices', 'household furniture', 'insects', 'large carnivores',
#     'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores','medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals',
#     'trees', 'vehicles 1', 'vehicles 2'
# ]
    'CIFAR100': { 
    '0':{"query": "SELECT * FROM CIFAR100 WHERE (label = 0 OR label = 1) AND quality > 0.9 LIMIT 5000", "label" : [0,4] , "num": 5000, "quality": 0.9 ,"val": 40},
    '1':{"query": "SELECT * FROM CIFAR100 WHERE (label = 0 OR label = 3) AND quality > 0.9 LIMIT 10000", "label" : [1,7,10,19] , "num": 10000, "quality": 0.9 ,"val": 88 },
    '2':{"query": "SELECT * FROM CIFAR100 WHERE (label = 0 OR label = 1) AND quality > 0.9 LIMIT 20000", "label" : [0, 2, 5, 8, 11, 14, 17, 19] , "num": 20000, "quality": 0.9 ,"val": 200 },
    '3':{"query": "SELECT * FROM CIFAR100 WHERE (label = 0 OR label = 1) AND quality > 0.9 LIMIT 35000", "label" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 18] , "num": 35000, "quality": 0.9 ,"val": 310 },
    '4':{"query": "SELECT * FROM CIFAR100 WHERE (label = 0 OR label = 1) AND quality > 0.9 LIMIT 45000", "label" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12, 13, 15, 16, 17, 18] , "num": 45000, "quality": 0.9 ,"val": 370 },
    },
    'tinyimagenet': {
    '0':{"query": "SELECT * FROM CIFAR10 WHERE (label = 0 OR label = 1) AND quality > 0.9 LIMIT 500", "label" : [0,1,3] , "num": 10000, "quality": 0.9 ,"val": 40},
    '1':{"query": "SELECT * FROM CIFAR10 WHERE (label = 0 OR label = 3) AND quality > 0.9 LIMIT 500", "label" : [0,2,3,5,6] , "num": 20000, "quality": 0.9 ,"val": 50 },
    '2':{"query": "SELECT * FROM CIFAR10 WHERE (label = 0 OR label = 1) AND quality > 0.9 LIMIT 500", "label" : [0,1,2,3,4,5,6] , "num": 30000, "quality": 0.9 ,"val": 60 },
    '3':{"query": "SELECT * FROM CIFAR10 WHERE (label = 0 OR label = 1) AND quality > 0.9 LIMIT 500", "label" : [1,2,3,4,5,6,8,9] , "num": 35000, "quality": 0.9 ,"val": 70 },
    '4':{"query": "SELECT * FROM CIFAR10 WHERE (label = 0 OR label = 1) AND quality > 0.9 LIMIT 48000", "label" : [0,1,2,3,4,5,6,7,8,9] , "num": 48000, "quality": 0.9 ,"val": 90 },
    },
    'MNIST': {
    '0':{"query": "SELECT * FROM CIFAR10 WHERE (label = 0 OR label = 1) AND quality > 0.9 LIMIT 500", "label" : [0,1,3] , "num": 10000, "quality": 0.9 ,"val": 40},
    '1':{"query": "SELECT * FROM CIFAR10 WHERE (label = 0 OR label = 3) AND quality > 0.9 LIMIT 500", "label" : [0,2,3,5,6] , "num": 30000, "quality": 0.9 ,"val": 110 },
    '2':{"query": "SELECT * FROM CIFAR10 WHERE (label = 0 OR label = 1) AND quality > 0.9 LIMIT 500", "label" : [0,1,2,3,4,5,6] , "num": 30000, "quality": 0.9 ,"val": 125},
    '3':{"query": "SELECT * FROM CIFAR10 WHERE (label = 0 OR label = 1) AND quality > 0.9 LIMIT 500", "label" : [1,2,3,4,5,6,8,9] , "num": 30000, "quality": 0.9 ,"val": 135 },
    '4':{"query": "SELECT * FROM CIFAR10 WHERE (label = 0 OR label = 1) AND quality > 0.9 LIMIT 48000", "label" : [0,1,2,3,4,5,6,7,8,9] , "num": 45000, "quality": 0.9 ,"val": 185 },
    }
}