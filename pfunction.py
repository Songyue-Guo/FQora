import random
def linear_price(quality_scores,alpha):
    prices = []
    for score in quality_scores:
        prices.append(score*alpha)
    return prices
def mid_price(quality_scores,max_value):
    prices = []
    for score in quality_scores:
        prices.append(max_value/2)

    return prices

def QIRANA(quality_scores,alpha):
    prices = []
    for score in quality_scores:
        if score != quality_scores[0] and score != quality_scores[-1]:
            prices.append(score*alpha-random.uniform(-5*alpha,5*alpha))
        else:
            prices.append(score*alpha)   
    return prices