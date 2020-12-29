"""
查看相邻帧时间戳 的差值
"""
import os
import sys
sys.path.append('/home/dlr/Project/ai-imu-dr/src')
import shutil
import numpy as np
from collections import namedtuple
import glob
import time
import datetime
import pickle
import torch
import matplotlib.pyplot as plt
from termcolor import cprint
from navpy import lla2ned
from collections import OrderedDict
from dataset import BaseDataset
from utils_torch_filter import TORCHIEKF
from utils_numpy_filter import NUMPYIEKF as IEKF
from utils import prepare_data
from train_torch_filter import train_filter
# from utils_plot import results_filter

def difftime(pathtime):
    rt = np.loadtxt(pathtime)
    N = rt.shape[0]
    dt = np.zeros(N-1)
    for i in range(N-1):
        dt[i] = rt[i+1] - rt[i]
    
    #绘制图  0.0078~0.0117 之间
    x = np.arange(0, N)
    l = plt.plot(x,rt,marker='o', markerfacecolor='none', label='T')
    plt.xlabel('frame')
    plt.ylabel('s')
    plt.legend()
    plt.show(block=True)
    # plt.show()

# https://www.google.com/search?newwindow=1&sxsrf=ALeKk00k7G5Jl0hGlUlNuPqmqZkX4bxisw%3A1608993895769&ei=Z0znX-a-LpWpoATg6Kto&q=python+find+index++same+value+another+array&oq=python+find+index++same+value+another+array&gs_lcp=CgZwc3ktYWIQAzoECCMQJzoFCAAQywE6AggAOgQIABAeOgYIABAIEB46BwghEAoQoAE6BQghEKABUPDkAVjQ1gJggtkCaAJwAHgBgAH0A4gBjkWSAQ0wLjIyLjEzLjIuMS4xmAEAoAEBqgEHZ3dzLXdpesABAQ&sclient=psy-ab&ved=0ahUKEwimjums8evtAhWVFIgKHWD0Cg0Q4dUDCA0&uact=5
# https://stackoverflow.com/a/8251757/9641752
# import numpy as np
# filepath = "/media/kitti/dataset/rawdata/2011_09_30/2011_09_30_drive_0016_"
# ntextractpath = filepath + "extract/oxts/ntimes.txt"
# ntsyncpath = filepath + "sync/oxts/ntimes.txt"
# print("100hz " + ntextractpath)
# print("10hz " + ntsyncpath)
# x = np.loadtxt(ntextractpath)#np.array([0.03,0.05,0.07,0.1,0.9,0.18,0.26,0.27])
# y = np.loadtxt(ntsyncpath)#np.array([0.05,0.1,0.26])

# index = np.argsort(x)
# sorted_x = x[index]
# sorted_index = np.searchsorted(sorted_x, y)

# yindex = np.take(index, sorted_index, mode="clip")
# mask = x[yindex] != y

# result = np.ma.array(yindex, mask=mask)
# # print(result)
# indsave = filepath + "sync/oxts/ind_extract.txt"
# print("ind save to " + indsave)
# np.savetxt(indsave, result, fmt='%d')



if __name__ == '__main__':
    timepath = 'results/2011_09_30_drive_0016_extract/ntimes.txt'
    difftime(timepath)

