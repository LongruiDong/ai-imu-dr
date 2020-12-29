"""
对原本的位姿 变换参考系 Twimu -> KITTI odometry benchmark Twc
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

# 保存 原始数据 和 odometry中seqid 的对应
odometry_extract = OrderedDict()
odometry_extract["00"] = "2011_10_03_drive_0027_extract"
odometry_extract["01"] = "2011_10_03_drive_0042_extract"
odometry_extract["02"] = "2011_10_03_drive_0034_extract"
odometry_extract["03"] = "2011_09_26_drive_0067_extract"
odometry_extract["04"] = "2011_09_30_drive_0016_extract"
odometry_extract["05"] = "2011_09_30_drive_0018_extract"
odometry_extract["06"] = "2011_09_30_drive_0020_extract"
odometry_extract["07"] = "2011_09_30_drive_0027_extract"
odometry_extract["08"] = "2011_09_30_drive_0028_extract"
odometry_extract["09"] = "2011_09_30_drive_0033_extract"
odometry_extract["10"] = "2011_09_30_drive_0034_extract"

# extract 和 odometry benchmark seq id 的映射
odometry_seqid = OrderedDict()
odometry_seqid["2011_10_03_drive_0027_extract"] = "00"
odometry_seqid["2011_10_03_drive_0042_extract"] = "01"
odometry_seqid["2011_10_03_drive_0034_extract"] = "02"
odometry_seqid["2011_09_26_drive_0067_extract"] = "03"
odometry_seqid["2011_09_30_drive_0016_extract"] = "04"
odometry_seqid["2011_09_30_drive_0018_extract"] = "05"
odometry_seqid["2011_09_30_drive_0020_extract"] = "06"
odometry_seqid["2011_09_30_drive_0027_extract"] = "07"
odometry_seqid["2011_09_30_drive_0028_extract"] = "08"
odometry_seqid["2011_09_30_drive_0033_extract"] = "09"
odometry_seqid["2011_09_30_drive_0034_extract"] = "10"

# 读取KITTI格式的轨迹 返回序列的变换矩阵
def loadtraj(filename):
    file = open(filename)
    file_lines = file.readlines()
    numberOfLines = len(file_lines)
    dataArray = np.zeros((numberOfLines, 4, 4))
    index = 0
    for line in file_lines:
        line = line.strip() # 参数为空时，默认删除开头、结尾处空白符（包括'\n', '\r',  '\t',  ' ')
        formLine = line.split( ) #按空格切分？
        dataArray[index] = np.eye(4) 
        dataArray[index, 0, 0:4] = formLine[0:4]
        dataArray[index, 1, 0:4] = formLine[4:8]
        dataArray[index, 2, 0:4] = formLine[8:12]
        index += 1
    
    return dataArray #

# copy from global jointopt  resultprocess.py
# 读取 calib_imu_to_velo.txt  和 Tr.txt 分别表示 Tvelo_imu Tcamrect_velo
# 上面txt中 以一行保存 即KITTI 格式
def loadcalib(filename):
    file = open(filename)
    file_lines = file.readlines()
    numberOfLines = len(file_lines)
    dataArray = np.zeros((numberOfLines, 4, 4))
    #labels = []
    index = 0
    for line in file_lines:
        line = line.strip() # 参数为空时，默认删除开头、结尾处空白符（包括'\n', '\r',  '\t',  ' ')
        formLine = line.split( ) #按空格切分？
        dataArray[index] = np.eye(4) 
        dataArray[index, 0, 0:4] = formLine[0:4]
        dataArray[index, 1, 0:4] = formLine[4:8]
        dataArray[index, 2, 0:4] = formLine[8:12]
        #labels.append((formLine[-1]))
        index += 1
    
    return dataArray[0] #因为实际就一行 返回变换矩阵

# 计算并保存每个序列 imu 到 rect相机0 的变换
def getTimucamr(odometryroot):
    seq_dirs = os.listdir(odometryroot) # /media/kitti/dataset/sequences下的 00~10
    for n_iter, date_dir in enumerate(seq_dirs):
        # get access to each sequence
        path1 = os.path.join(odometryroot, date_dir) #/sequences/00
        if not os.path.isdir(path1):
            continue
        
        imu2velofile = os.path.join(path1, 'calib_imu_to_velo.txt')
        velo2camfile = os.path.join(path1, 'Tr.txt')

        #分别读取
        Tvelo_imu = loadcalib(imu2velofile)
        Tcam_velo = loadcalib(velo2camfile)

        Tcam_imu = np.matmul(Tcam_velo, Tvelo_imu)

        imu2camfile = os.path.join(path1, 'imu_to_camr.txt')
        np.savetxt(imu2camfile, Tcam_imu) # , fmt='%.6f'

        # if path1[-2:] == "03":
        #     continue
        hereroot = "results"
        drivedir = odometry_extract[path1[-2:]] # "00"
        if not os.path.exists(os.path.join(hereroot, drivedir)):
            print("dir Not exits: " + os.path.join(hereroot, drivedir))
            continue
        # 保存到对应100hz 文件夹下
        herefile = os.path.join(hereroot, drivedir, 'imu_to_camr.txt')
        np.savetxt(herefile, Tcam_imu)

# 对输入的某个Tw_imu 轨迹 使用 对应的外参 Timu_cam 转化为Twc
def changeref(intraj, Timu_cam):
    nframe = intraj.shape[0] # 总帧数
    outraj = np.zeros((nframe, 4, 4))
    for i in range(nframe):
        Tw_imu = intraj[i]# .copy()
        Twc = np.matmul(Tw_imu, Timu_cam)
        outraj[i] = np.eye(4)
        outraj[i] = Twc
    
    # 转化为以初始帧为参考
    T0 = outraj[0].copy() #注意这里 否则会变化
    T0inv = np.linalg.inv(T0)
    for i in range(nframe):
        Ti = outraj[i].copy()
        outraj[i] = np.matmul(T0inv, Ti)
    
    return outraj
    



# 保存位姿 以KITTI格式
def savetrajKITTI(intraj,savefile):
    nframe = intraj.shape[0] # 总帧数
    # 转化为以初始帧为参考
    T0 = intraj[0].copy() #注意这里 否则会变化
    T0inv = np.linalg.inv(T0)
    for i in range(nframe):
        Ti = intraj[i].copy()
        intraj[i] = np.matmul(T0inv, Ti)
    
    
    #写入文件 
    f = open(savefile , "w") 
    for j in range(nframe):
        slidt = [intraj[j,0,0], intraj[j,0,1], intraj[j,0,2], intraj[j,0,3],
                 intraj[j,1,0], intraj[j,1,1], intraj[j,1,2], intraj[j,1,3],
                 intraj[j,2,0], intraj[j,2,1], intraj[j,2,2], intraj[j,2,3]]
        k = 0
        for v in slidt:
            #if type(v) == 'torch.float32':
            #    v = v.numpy() 
            f.write(str(v))
            if(k<11): # 最后一个无空格，直接换行
                f.write(' ')
            k = k + 1
        f.write('\n')
    f.close()


# 对wimu的结果进行变换
def wimutowc(resroot):
    seq_dirs = os.listdir(resroot) # results 下的 _extract
    for n_iter, date_dir in enumerate(seq_dirs):
        # get access to each sequence
        path1 = os.path.join(resroot, date_dir) #results/2011_09_30_drive_0016_extract
        if not os.path.isdir(path1):
            continue

        seqid = odometry_seqid[date_dir]
        print("-> " + path1 + ", " + seqid)
        imugtfile = os.path.join(path1, 'imugt.txt')
        imuestfile = os.path.join(path1, 'imuest.txt')

        imugt = loadtraj(imugtfile) # n,4,4
        imuest = loadtraj(imuestfile)

        imu2camfile = os.path.join(path1, 'imu_to_camr.txt')
        Tcam_imu = np.loadtxt(imu2camfile)
        Timu_cam = np.linalg.inv(Tcam_imu)

        rawgtwc = changeref(imugt, Timu_cam)
        rawestwc = changeref(imuest, Timu_cam)

        rawgtwcfile = os.path.join(path1, 'rawgtwc.txt')
        rawestwcfile = os.path.join(path1, 'rawestwc.txt')

        # 保存 Twc
        savetrajKITTI(rawgtwc, rawgtwcfile)
        savetrajKITTI(rawestwc, rawestwcfile)
        print("save Twc rawgt : " + rawgtwcfile)
        print("save Twc rawest : " + rawestwcfile)




if __name__ == '__main__':
    # odometryroot = '/media/kitti/dataset/sequences'
    resultroot = 'results'
    # getTimucamr(odometryroot) #从 calib 中计算Imu 和 相机的外参 一次就行
    # 对Tw_imu 变换 到 Tw_c
    wimutowc(resultroot)
