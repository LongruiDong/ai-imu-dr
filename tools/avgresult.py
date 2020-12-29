#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
"""
import argparse
import sys
import os
#当编写的Python文件即当前代码及注释中含有中文或者非英文字符时，需要声明编码格式为utf-8
import os   #读取txt文件所需要的包
import linecache #读取指定行函数linecache.getline(file_ob, line_num)所在的包
import numpy as np
import math


def report(pathtoresult):
    #
    root = pathtoresult + 'seqavgstats_'#读取的批量txt所在的文件夹的路径 /home/dlr/Project/ORB_SLAM2/result/lidarstereo/seqavgstats_04.txt

    file_ob_list = []   #定义一个列表，用来存放刚才读取的520个txt文件名
    for i in range(11):  #循环得到它的具体路径
        fileob = root + "%02d"%i + '.txt'
        if not (os.path.exists(fileob)):
            print("Warning: seqavgstats_%02d doesn't exist !"%(i))
            continue
        else:
            print("find " + fileob)  #43

        
        file_ob_list.append(fileob) #将路径追加到列表中存储
    

    #储存两种误差的列表
    te = []
    re = []

    line_num = 1  #从txt的第一行开始读入
    total_line = len(open(file_ob_list[0]).readlines()) #计算一个txt中有多少行
    # print('total_line =',total_line)
    print("\nSeq\tTranslation(%)\tRotation(°/100m)\n")
    while line_num <= total_line:        #只有读完的行数小于等于总行数时才再读下一行，否则结束读取
        for file_ob in file_ob_list:    #按顺序循环读取所有文件
                if not (os.path.exists(file_ob)):
                    print("Warning: %s doesn't exist !"%(file_ob)) #55
                    continue 
                
                seqid = file_ob[-6:-4]   #  
                line = linecache.getline(file_ob, line_num)#读取这个文件的第line_num行
                line = line.strip() #去掉这一行最后一个字符\n 即换行符
                if line is None or len(line) ==0 :
                    break
                
                fields = line.split( )  
                terror = float(fields[0])           # fields[0]是'ENSG00000242268.2'   fields[1]是'0.0'
                rerror = float(fields[1])           #rads
                print("%s  \t %.5f    \t %.5f"%( seqid, terror*100, math.degrees(rerror)*100 ))
                te.append(terror)           
                re.append(rerror)         
    
        line_num = line_num + 1     #行数加1，好接着读取每一个文件的第二行

    #计算平均值
    t_avg = np.mean(te) * 100 #11个序列的相对平移误差 %
    r_avg = math.degrees(np.mean(re)) * 100 #旋转误差 °/100m

    #输出结果
    print("\nAverage\t%.5f\t%.5f"%(t_avg, r_avg))


if __name__ == '__main__':
    
    # parser command lines
    parser = argparse.ArgumentParser(description='''
    This script report Localization error on KITTI 00-10, should run "./evakit/evaluate_odometry path/toresult/"  
    ''') 
    parser.add_argument('result_dir', help='预测的轨迹所在文件夹',default="/home/dlr/Project/ORB_posegraph/result/lidarstereo/") #

    args = parser.parse_args()
    # 读取参数
    estdir = args.result_dir
    
    print("report result...\n")
    report(estdir)