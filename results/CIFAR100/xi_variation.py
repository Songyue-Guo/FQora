import sys
sys.path.append('../..')
from plot.hmean import harmonic_mean

s = 364.86
b = 551.9195199012756
utilities = [s,b,s]


print(harmonic_mean(utilities))
