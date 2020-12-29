"""
使用ind_extract或者 直接间隔下采样 对已有的100hz数据处理 得到10hz的数据 
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
from main_kitti import KITTIDataset
from changerefFrame import *



# 对原始各类数据进行转化 5个
def todometrylike(resroot):
    seq_dirs = os.listdir(resroot) # results 下的 _extract
    for n_iter, date_dir in enumerate(seq_dirs):
        # get access to each sequence
        path1 = os.path.join(resroot, date_dir) #results/2011_09_30_drive_0016_extract
        if not os.path.isdir(path1):
            continue
        seqid = KITTIDataset.odometry_seqid[date_dir]
        print("-> " + path1 + ", " + seqid)

        # 保存 到 sync 10hz 对应的文件
        indextractfile = os.path.join(path1, 'ind_extract.txt')
        print("load ind_extract " + indextractfile)
        index_extract = np.loadtxt(indextractfile, dtype=int)

        # 按照odometry 和 sync 之间的起始关系 截取index_extract 对应部分
        syncdir = date_dir[:-7] + "sync"
        index_slice = KITTIDataset.odometry_benchmark_sync[syncdir] # 起始 闭区间 [start,end]
        print("only use " + str(index_slice))
        start = index_slice[0]
        end = index_slice[1] + 1 # 前闭后开
        # https://www.zmonster.me/2016/03/09/numpy-slicing-and-indexing.html
        index_odometry = index_extract[start : end]
        
        rawgtwcfile = os.path.join(path1, 'rawgtwc.txt')
        rawestwcfile = os.path.join(path1, 'rawestwc.txt')

        rawgtwc = loadtraj(rawgtwcfile) # n,4,4
        rawestwc = loadtraj(rawestwcfile) # 之前这里搞成rawgt路径了。。
        if rawgtwc.shape[0] != rawestwc.shape[0]:
            print("two traj shape not equal !")
        # https://www.runoob.com/numpy/numpy-advanced-indexing.html
        # 使用index_odometry进行索引 新的轨迹
        odogtwc = rawgtwc[index_odometry]
        odoestwc = rawestwc[index_odometry]
        odogtwcfile = os.path.join(path1, 'odogtwc.txt')
        odoestwcfile = os.path.join(path1, 'odoestwc.txt')

        # 保存odo 轨迹
        print("save odogtwc " + odogtwcfile)
        savetrajKITTI(odogtwc, odogtwcfile)
        print("save odoestwc " + odoestwcfile)
        savetrajKITTI(odoestwc, odoestwcfile)

        path2 = "match10hz"
        odogtwcfile1 = os.path.join(path2, 'oxtsgt', seqid + '.txt')
        odoestwcfile1 = os.path.join(path2, seqid + '.txt')
        savetrajKITTI(odogtwc, odogtwcfile1)
        savetrajKITTI(odoestwc, odoestwcfile1)

        cov_ypath = os.path.join(path1, "covsy.txt")
        carimupath = os.path.join(path1, "car_imu.txt") # Timu_body
        P6path = os.path.join(path1, "Psave.txt") #状态协方差 N 6 6

        cov_y = np.loadtxt(cov_ypath)
        carimu = np.loadtxt(carimupath)
        P6 = np.loadtxt(P6path)

        odocovy = cov_y[index_odometry]
        odocarimu = carimu[index_odometry]
        odoP6 = P6[index_odometry]

        odocov_ypath = os.path.join(path1, "odocovsy.txt")
        odocarimupath = os.path.join(path1, "odocar_imu.txt") # Timu_body
        odoP6path = os.path.join(path1, "odoPsave.txt") #状态协方差 N 6 6

        print("save odocov_y " + odocov_ypath)
        np.savetxt(odocov_ypath, odocovy)
        np.savetxt(odocarimupath, odocarimu)
        print("save odoP6 " + odoP6path)
        np.savetxt(odoP6path, odoP6)



        

        


if __name__ == '__main__':
    # odometryroot = '/media/kitti/dataset/sequences'
    resultroot = 'results'
    # getTimucamr(odometryroot)
    # 对100hz进行处理
    todometrylike(resultroot)