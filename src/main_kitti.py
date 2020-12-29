import os
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
from utils_plot import results_filter
from utils_plot import results_save

# 总的调度 args:KITTIArgs
def launch(args):
    if args.read_data: #读取raw data 100hz
        args.dataset_class.read_data(args)
        # args.dataset_class.process_time(args) #装换为统一时间戳格式
        # args.dataset_class.findtimematch(args) #寻找对应关系
    dataset = args.dataset_class(args) #KITTIDataset类

    if args.train_filter: #训练
        train_filter(args, dataset)

    if args.test_filter: # 进行测试  没有03结果 因为03没有对应的raw data
        test_filter(args, dataset) #保存结果到results

    if args.results_filter:
        # results_filter(args, dataset)
        results_save(args, dataset) # 保存处理数据


class KITTIParameters(IEKF.Parameters):
    # gravity vector
    g = np.array([0, 0, -9.80655])

    cov_omega = 2e-4
    cov_acc = 1e-3
    cov_b_omega = 1e-8
    cov_b_acc = 1e-6
    cov_Rot_c_i = 1e-8
    cov_t_c_i = 1e-8
    cov_Rot0 = 1e-6
    cov_v0 = 1e-1
    cov_b_omega0 = 1e-8
    cov_b_acc0 = 1e-3
    cov_Rot_c_i0 = 1e-5
    cov_t_c_i0 = 1e-2
    cov_lat = 1
    cov_up = 10

    def __init__(self, **kwargs):
        super(KITTIParameters, self).__init__(**kwargs)
        self.set_param_attr()

    def set_param_attr(self):
        attr_list = [a for a in dir(KITTIParameters) if
                     not a.startswith('__') and not callable(getattr(KITTIParameters, a))]
        for attr in attr_list:
            setattr(self, attr, getattr(KITTIParameters, attr))


# KITTI raw data 的信息
class KITTIDataset(BaseDataset):
    OxtsPacket = namedtuple('OxtsPacket',
                            'lat, lon, alt, ' + 'roll, pitch, yaw, ' + 'vn, ve, vf, vl, vu, '
                                                                       '' + 'ax, ay, az, af, al, '
                                                                            'au, ' + 'wx, wy, wz, '
                                                                                     'wf, wl, wu, '
                                                                                     '' +
                            'pos_accuracy, vel_accuracy, ' + 'navstat, numsats, ' + 'posmode, '
                                                                                  'velmode, '
                                                                                  'orimode')

    # Bundle into an easy-to-access structure
    OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')
    min_seq_dim = 25 * 100  # 60 s
    # 这三个用来做什么
    datasets_fake = ['2011_09_26_drive_0093_extract', '2011_09_28_drive_0039_extract',
                     '2011_09_28_drive_0002_extract']
    """
    '2011_09_30_drive_0028_extract' has trouble at N = [6000, 14000] -> test data  seq08
    '2011_10_03_drive_0027_extract' has trouble at N = 29481                       seq00 to correct  index_extract 问题最多
    '2011_10_03_drive_0034_extract' has trouble at N = [33500, 34000]              seq02 to correct
    my:
    '09_30_drive_0018_extract'     2-3                                             seq05
    '09_30_drive_0020_extract'     0-1                                             seq06
    '09_30_drive_0027_extract'     7663-7665  时间增长为负 但没影响到index match 频率降低时正好跳过了     seq07                            
    '09_30_drive_0028_extract'     0-1                                             seq08
    """

    # training set to the raw data of the KITTI dataset.
    # The following dict lists the name and end frame of each sequence that
    # has been used to extract the visual odometry / SLAM training set      00-10
    odometry_benchmark = OrderedDict()
    odometry_benchmark["2011_10_03_drive_0027_extract"] = [0, 45692]
    odometry_benchmark["2011_10_03_drive_0042_extract"] = [0, 12180]
    odometry_benchmark["2011_10_03_drive_0034_extract"] = [0, 47935]
    odometry_benchmark["2011_09_26_drive_0067_extract"] = [0, 8000]
    odometry_benchmark["2011_09_30_drive_0016_extract"] = [0, 2950]
    odometry_benchmark["2011_09_30_drive_0018_extract"] = [0, 28659]
    odometry_benchmark["2011_09_30_drive_0020_extract"] = [0, 11347]
    odometry_benchmark["2011_09_30_drive_0027_extract"] = [0, 11545]
    odometry_benchmark["2011_09_30_drive_0028_extract"] = [11231, 53650]
    odometry_benchmark["2011_09_30_drive_0033_extract"] = [0, 16589]
    odometry_benchmark["2011_09_30_drive_0034_extract"] = [0, 12744]

    # 和上面不同在哪里 index为何变化 貌似没用到  img 指图像?  但这也不是真实的对应起始关系
    odometry_benchmark_img = OrderedDict()
    odometry_benchmark_img["2011_10_03_drive_0027_extract"] = [0, 45400]
    odometry_benchmark_img["2011_10_03_drive_0042_extract"] = [0, 11000]
    odometry_benchmark_img["2011_10_03_drive_0034_extract"] = [0, 46600]
    odometry_benchmark_img["2011_09_26_drive_0067_extract"] = [0, 8000]
    odometry_benchmark_img["2011_09_30_drive_0016_extract"] = [0, 2700]
    odometry_benchmark_img["2011_09_30_drive_0018_extract"] = [0, 27600]
    odometry_benchmark_img["2011_09_30_drive_0020_extract"] = [0, 11000]
    odometry_benchmark_img["2011_09_30_drive_0027_extract"] = [0, 11000]
    odometry_benchmark_img["2011_09_30_drive_0028_extract"] = [11000, 51700]
    odometry_benchmark_img["2011_09_30_drive_0033_extract"] = [0, 15900]
    odometry_benchmark_img["2011_09_30_drive_0034_extract"] = [0, 12000]

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

    # from odometry benchmark中的 /media/kitti/dataset/devkit 
    # 表示图像的时间在raw data sync（10hz） 中的起始对应
    odometry_benchmark_sync = OrderedDict()
    odometry_benchmark_sync["2011_10_03_drive_0027_sync"] = [0, 4540]
    odometry_benchmark_sync["2011_10_03_drive_0042_sync"] = [0, 1100]
    odometry_benchmark_sync["2011_10_03_drive_0034_sync"] = [0, 4660]
    odometry_benchmark_sync["2011_09_26_drive_0067_sync"] = [0, 800]
    odometry_benchmark_sync["2011_09_30_drive_0016_sync"] = [0, 270]
    odometry_benchmark_sync["2011_09_30_drive_0018_sync"] = [0, 2760]
    odometry_benchmark_sync["2011_09_30_drive_0020_sync"] = [0, 1100]
    odometry_benchmark_sync["2011_09_30_drive_0027_sync"] = [0, 1100]
    odometry_benchmark_sync["2011_09_30_drive_0028_sync"] = [1100, 5170]
    odometry_benchmark_sync["2011_09_30_drive_0033_sync"] = [0, 1590]
    odometry_benchmark_sync["2011_09_30_drive_0034_sync"] = [0, 1200] 

    def __init__(self, args):
        super(KITTIDataset, self).__init__(args)
        # 04怎么没用？
        self.datasets_validatation_filter['2011_09_30_drive_0028_extract'] = [11231, 53650] #8
        self.datasets_train_filter["2011_10_03_drive_0042_extract"] = [0, None] #01
        self.datasets_train_filter["2011_09_30_drive_0018_extract"] = [0, 15000] #05
        self.datasets_train_filter["2011_09_30_drive_0020_extract"] = [0, None] #06
        self.datasets_train_filter["2011_09_30_drive_0027_extract"] = [0, None] #07
        self.datasets_train_filter["2011_09_30_drive_0033_extract"] = [0, None] #09
        self.datasets_train_filter["2011_10_03_drive_0027_extract"] = [0, 18000] #00
        self.datasets_train_filter["2011_10_03_drive_0034_extract"] = [0, 31000] #02
        self.datasets_train_filter["2011_09_30_drive_0034_extract"] = [0, None] #10

        for dataset_fake in KITTIDataset.datasets_fake:
            if dataset_fake in self.datasets:
                self.datasets.remove(dataset_fake)
            if dataset_fake in self.datasets_train:
                self.datasets_train.remove(dataset_fake)

    @staticmethod
    def read_data(args):
        """
        Read the data from the KITTI dataset

        :param args:
        :return:
        """

        print("Start read_data")
        t_tot = 0  # sum of times for the all dataset
        date_dirs = os.listdir(args.path_data_base) # rawdata/  2011_10_03 2011_09_30
        for n_iter, date_dir in enumerate(date_dirs):
            # get access to each sequence
            path1 = os.path.join(args.path_data_base, date_dir) #/rawdata/2011_09_30
            if not os.path.isdir(path1):
                continue
            date_dirs2 = os.listdir(path1) # path1/下所有目录及文件

            for date_dir2 in date_dirs2:
                path2 = os.path.join(path1, date_dir2)
                if not os.path.isdir(path2): #只要目录  '/media/kitti/dataset/rawdata/2011_10_03/2011_10_03_drive_0042_sync'
                    continue
                # read data
                oxts_files = sorted(glob.glob(os.path.join(path2, 'oxts', 'data', '*.txt')))
                oxts = KITTIDataset.load_oxts_packets_and_poses(oxts_files)

                """ Note on difference between ground truth and oxts solution:
                    - orientation is the same
                    - north and east axis are inverted
                    - position are closed to but different
                    => oxts solution is not loaded
                """

                print("\n Sequence name : " + date_dir2)
                if len(oxts) < KITTIDataset.min_seq_dim:  #  sequence shorter than 30 s are rejected
                    cprint("Dataset is too short ({:.2f} s)".format(len(oxts) / 100), 'yellow')
                    continue
                lat_oxts = np.zeros(len(oxts))
                lon_oxts = np.zeros(len(oxts))
                alt_oxts = np.zeros(len(oxts))
                roll_oxts = np.zeros(len(oxts))
                pitch_oxts = np.zeros(len(oxts))
                yaw_oxts = np.zeros(len(oxts))
                roll_gt = np.zeros(len(oxts))
                pitch_gt = np.zeros(len(oxts))
                yaw_gt = np.zeros(len(oxts))
                t = KITTIDataset.load_timestamps(path2) # 读年月日时分秒
                acc = np.zeros((len(oxts), 3))
                acc_bis = np.zeros((len(oxts), 3))
                gyro = np.zeros((len(oxts), 3))
                gyro_bis = np.zeros((len(oxts), 3))
                p_gt = np.zeros((len(oxts), 3))
                v_gt = np.zeros((len(oxts), 3))
                v_rob_gt = np.zeros((len(oxts), 3))

                k_max = len(oxts)
                for k in range(k_max):
                    oxts_k = oxts[k]
                    t[k] = 3600 * t[k].hour + 60 * t[k].minute + t[k].second + t[
                        k].microsecond / 1e6
                    lat_oxts[k] = oxts_k[0].lat
                    lon_oxts[k] = oxts_k[0].lon
                    alt_oxts[k] = oxts_k[0].alt
                    acc[k, 0] = oxts_k[0].af
                    acc[k, 1] = oxts_k[0].al
                    acc[k, 2] = oxts_k[0].au
                    acc_bis[k, 0] = oxts_k[0].ax # imu frame 实际使用的 但为啥取名叫_bias??
                    acc_bis[k, 1] = oxts_k[0].ay
                    acc_bis[k, 2] = oxts_k[0].az
                    gyro[k, 0] = oxts_k[0].wf
                    gyro[k, 1] = oxts_k[0].wl
                    gyro[k, 2] = oxts_k[0].wu
                    gyro_bis[k, 0] = oxts_k[0].wx
                    gyro_bis[k, 1] = oxts_k[0].wy
                    gyro_bis[k, 2] = oxts_k[0].wz
                    roll_oxts[k] = oxts_k[0].roll
                    pitch_oxts[k] = oxts_k[0].pitch
                    yaw_oxts[k] = oxts_k[0].yaw
                    v_gt[k, 0] = oxts_k[0].ve # 真值速度
                    v_gt[k, 1] = oxts_k[0].vn
                    v_gt[k, 2] = oxts_k[0].vu
                    # 社么意思 vn 和 vf 不应该才是对应的吗
                    v_rob_gt[k, 0] = oxts_k[0].vf
                    v_rob_gt[k, 1] = oxts_k[0].vl
                    v_rob_gt[k, 2] = oxts_k[0].vu
                    # oxts_k[1] T_w_imu gt
                    p_gt[k] = oxts_k[1][:3, 3]
                    Rot_gt_k = oxts_k[1][:3, :3]
                    roll_gt[k], pitch_gt[k], yaw_gt[k] = IEKF.to_rpy(Rot_gt_k)

                t0 = t[0]
                t = np.array(t) - t[0] # 这里t已经减掉起始了
                # some data can have gps out
                if np.max(t[:-1] - t[1:]) > 0.1:
                    cprint(date_dir2 + " has time problem", 'yellow')
                ang_gt = np.zeros((roll_gt.shape[0], 3))
                ang_gt[:, 0] = roll_gt
                ang_gt[:, 1] = pitch_gt
                ang_gt[:, 2] = yaw_gt

                p_oxts = lla2ned(lat_oxts, lon_oxts, alt_oxts, lat_oxts[0], lon_oxts[0],
                                 alt_oxts[0], latlon_unit='deg', alt_unit='m', model='wgs84')
                p_oxts[:, [0, 1]] = p_oxts[:, [1, 0]]  # see note

                # take correct imu measurements 注意没用 f-l-u 参考系 而是imu系
                u = np.concatenate((gyro_bis, acc_bis), -1)
                # convert from numpy
                t = torch.from_numpy(t)
                p_gt = torch.from_numpy(p_gt)
                v_gt = torch.from_numpy(v_gt)
                ang_gt = torch.from_numpy(ang_gt)
                u = torch.from_numpy(u)

                # convert to float
                t = t.float()
                u = u.float()
                p_gt = p_gt.float()
                ang_gt = ang_gt.float()
                v_gt = v_gt.float()

                mondict = {
                    't': t, 'p_gt': p_gt, 'ang_gt': ang_gt, 'v_gt': v_gt,
                    'u': u, 'name': date_dir2, 't0': t0 #还好这里也保存了起始时间点 可以恢复出原时间
                    }

                t_tot += t[-1] - t[0]
                KITTIDataset.dump(mondict, args.path_data_save, date_dir2)
        print("\n Total dataset duration : {:.2f} s".format(t_tot))

    @staticmethod
    def process_time(args):
        """
        convert the timestamp from the KITTI dataset to unix 格式

        :param args:
        :return:
        """

        print("Start convert_time")
        
        date_dirs = os.listdir(args.path_data_base) # rawdata  2011_10_03 2011_09_30
        for n_iter, date_dir in enumerate(date_dirs):
            # get access to each sequence
            path1 = os.path.join(args.path_data_base, date_dir) #/rawdata/2011_09_30
            if not os.path.isdir(path1):
                continue
            date_dirs2 = os.listdir(path1) # path1/下所有目录及文件

            for date_dir2 in date_dirs2:
                path2 = os.path.join(path1, date_dir2)
                if not os.path.isdir(path2): #只要目录  '/media/kitti/dataset/rawdata/2011_10_03/2011_10_03_drive_0042_sync'
                    continue
                # convert time
                if path2[-4:] == "ract":
                    continue
                # timestamp_file = os.path.join(path2, 'oxts', 'timestamps.txt')
                # print("file: " + timestamp_file)
                # # Read and parse the timestamps
                # ntimes = [] 
                # with open(timestamp_file, 'r') as f:
                #     for line in f.readlines():
                #         # NB: datetime only supports microseconds, but KITTI timestamps
                #         # give nanoseconds, so need to truncate last 4 characters to
                #         # get rid of \n (counts as 1) and extra 3 digits
                #         t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                #         nt =t.timestamp() # 转为unix时间
                #         ntimes.append(nt)
                # nts = np.array(ntimes)

                ntspath = os.path.join(path2, 'oxts', 'ntimes.txt')
                # print("save to: " + ntspath)
                # np.savetxt(ntspath, nts, fmt= '%.6f') #原地保存时间戳
                
                #绘制图  看sync的时间是否正常
                nts = np.loadtxt(ntspath)
                N = nts.shape[0]
                x = np.arange(0, N)
                l = plt.plot(x,nts,marker='o', markerfacecolor='none', label='s')
                plt.title(ntspath)
                plt.xlabel('index')
                plt.ylabel('timestamp')
                plt.legend()
                plt.show(block=True)

    @staticmethod
    def findtimematch(args):
        """
        从extract/oxts/ntimes.txt 中找到 sync/oxts/ntimes.txt 中时间相等的id
        保存一个数组index 大小和sync/oxts/ntimes.txt一样， extract[index[i]] = sync[i] 
        :param args:
        :return:
        """
        print("Find index in extract/oxts/ntimes of sync...")
        copypath = "results"
        date_dirs = os.listdir(args.path_data_base) # rawdata/  2011_10_03 2011_09_30
        for n_iter, date_dir in enumerate(date_dirs):
            # get access to each sequence
            path1 = os.path.join(args.path_data_base, date_dir) #/rawdata/2011_09_30
            if not os.path.isdir(path1):
                continue
            date_dirs2 = os.listdir(path1) # path1/下所有目录及文件

            for date_dir2 in date_dirs2:
                path2 = os.path.join(path1, date_dir2)
                if not os.path.isdir(path2): #只要目录  '/media/kitti/dataset/rawdata/2011_10_03/2011_10_03_drive_0042_sync'
                    continue
                if path2[-4:] == "ract":
                    continue
                print("path2[-4:] = " + path2[-4:])
                driveid = path2[:-4] #/media/kitti/dataset/rawdata/2011_10_03/2011_10_03_drive_0042_
                print("drive : " + driveid)
                drivesync = path2 #driveid + "sync"
                driveextract = driveid + "extract"
                # 不同帧率时间戳路径
                timesync_file = os.path.join(drivesync, 'oxts', 'ntimes.txt')
                timeextract_file = os.path.join(driveextract, 'oxts', 'ntimes.txt')
                
                # Read and parse the timestamps
                ntextract = np.loadtxt(timeextract_file)
                ntsync = np.loadtxt(timesync_file)
                
                print("100hz " + timeextract_file)
                print("10hz " + timesync_file)
                x = ntextract.copy()#np.array([0.03,0.05,0.07,0.1,0.9,0.18,0.26,0.27])
                y = ntsync.copy()#np.array([0.05,0.1,0.26])

                index = np.argsort(x)
                sorted_x = x[index]
                sorted_index = np.searchsorted(sorted_x, y)

                yindex = np.take(index, sorted_index, mode="clip")
                mask = x[yindex] != y

                indresult = np.ma.array(yindex, mask=mask)
                indsave = os.path.join(drivesync, 'oxts', 'ind_extract.txt')
                print("ind save to " + indsave)
                np.savetxt(indsave, indresult, fmt='%d')
                herepath = os.path.join(copypath, date_dir2[:-4] + 'extract', 'ind_extract.txt')
                print("ind copy to " + herepath)
                np.savetxt(herepath, indresult, fmt='%d')

                

    @staticmethod
    def prune_unused_data(args):
        """
        Deleting image and velodyne
        Returns:

        """

        unused_list = ['image_00', 'image_01', 'image_02', 'image_03', 'velodyne_points']
        date_dirs = ['2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']

        for date_dir in date_dirs:
            path1 = os.path.join(args.path_data_base, date_dir)
            if not os.path.isdir(path1):
                continue
            date_dirs2 = os.listdir(path1)

            for date_dir2 in date_dirs2:
                path2 = os.path.join(path1, date_dir2)
                if not os.path.isdir(path2):
                    continue
                print(path2)
                for folder in unused_list:
                    path3 = os.path.join(path2, folder)
                    if os.path.isdir(path3):
                        print(path3)
                        shutil.rmtree(path3)

    @staticmethod
    def subselect_files(files, indices):
        try:
            files = [files[i] for i in indices]
        except:
            pass
        return files

    @staticmethod
    def rotx(t):
        """Rotation about the x-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def roty(t):
        """Rotation about the y-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def rotz(t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    @staticmethod
    def pose_from_oxts_packet(packet, scale):
        """Helper method to compute a SE(3) pose matrix from an OXTS packet.
        """
        er = 6378137.  # earth radius (approx.) in meters

        # Use a Mercator projection to get the translation vector 和kitti devkit 一致
        tx = scale * packet.lon * np.pi * er / 180.
        ty = scale * er * np.log(np.tan((90. + packet.lat) * np.pi / 360.))
        tz = packet.alt
        t = np.array([tx, ty, tz])

        # Use the Euler angles to get the rotation matrix
        Rx = KITTIDataset.rotx(packet.roll)
        Ry = KITTIDataset.roty(packet.pitch)
        Rz = KITTIDataset.rotz(packet.yaw)
        R = Rz.dot(Ry.dot(Rx))

        # Combine the translation and rotation into a homogeneous transform
        return R, t

    @staticmethod
    def transform_from_rot_trans(R, t):
        """Transformation matrix from rotation matrix and translation vector."""
        R = R.reshape(3, 3)
        t = t.reshape(3, 1)
        return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    @staticmethod
    def load_oxts_packets_and_poses(oxts_files):
        """Generator to read OXTS ground truth data.  得到100hz真值
           Poses are given in an East-North-Up coordinate system
           whose origin is the first GPS position.
        """
        # Scale for Mercator projection (from first lat value)
        scale = None
        # Origin of the global coordinate system (first GPS position)
        origin = None

        oxts = []

        for filename in oxts_files:
            with open(filename, 'r') as f:
                for line in f.readlines():
                    line = line.split()
                    # Last five entries are flags and counts
                    line[:-5] = [float(x) for x in line[:-5]]
                    line[-5:] = [int(float(x)) for x in line[-5:]]
                    # 所有imu数据
                    packet = KITTIDataset.OxtsPacket(*line)

                    if scale is None:
                        scale = np.cos(packet.lat * np.pi / 180.)

                    R, t = KITTIDataset.pose_from_oxts_packet(packet, scale)

                    if origin is None:
                        origin = t

                    T_w_imu = KITTIDataset.transform_from_rot_trans(R, t - origin)

                    oxts.append(KITTIDataset.OxtsData(packet, T_w_imu))
        return oxts

    @staticmethod
    def load_timestamps(data_path):
        """Load timestamps from file."""
        timestamp_file = os.path.join(data_path, 'oxts', 'timestamps.txt')

        # Read and parse the timestamps
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(t)
        return timestamps

    def load_timestamps_img(data_path):
        """Load timestamps from file."""
        timestamp_file = os.path.join(data_path, 'image_00', 'timestamps.txt')

        # Read and parse the timestamps
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(t)
        return timestamps

# 在除了03 其余10个序列上测试
def test_filter(args, dataset):
    iekf = IEKF()
    torch_iekf = TORCHIEKF()

    # put Kitti parameters
    iekf.filter_parameters = KITTIParameters()
    iekf.set_param_attr()
    torch_iekf.filter_parameters = KITTIParameters()
    torch_iekf.set_param_attr()

    torch_iekf.load(args, dataset)
    iekf.set_learned_covariance(torch_iekf)

    for i in range(0, len(dataset.datasets)):
        dataset_name = dataset.dataset_name(i)
        if dataset_name not in dataset.odometry_benchmark.keys(): # 限于benchmark
            continue
        print("Test filter on sequence: " + dataset_name)
        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, i,
                                                       to_numpy=True)
        N = None
        u_t = torch.from_numpy(u).double() # 输入的imu测量
        measurements_covs = torch_iekf.forward_nets(u_t) #这是伪测量的协方差N=cov(y) ! 直接从模型预测得到每帧的协方差
        measurements_covs = measurements_covs.detach().numpy()
        start_time = time.time()
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, Pbuffer = iekf.run(t, u, measurements_covs,
                                                                   v_gt, p_gt, N,
                                                                   ang_gt[0])
        diff_time = time.time() - start_time
        print("Execution time: {:.2f} s (sequence time: {:.2f} s)".format(diff_time,
                                                                          t[-1] - t[0]))
        # 测试输出的x N  需要找出P
        mondict = {
            't': t, 'Rot': Rot, 'v': v, 'p': p, 'b_omega': b_omega, 'b_acc': b_acc,
            'Rot_c_i': Rot_c_i, 't_c_i': t_c_i, 'Pbuffer': Pbuffer, # 增加保存状态协方差
            'measurements_covs': measurements_covs,
            } # TIMU_body
        # 结果保存为.p文件
        dataset.dump(mondict, args.path_results, dataset_name + "_filter.p")


# 一些主要的参数
class KITTIArgs():
        path_data_base = "/media/kitti/dataset/rawdata" # kitti raw data 里面既有 100hz , 10hz
        path_data_save = "data" # 如果再生成数据的话 data1
        path_results = "results"
        path_temp = "temp" # 模型文件

        epochs = 400
        seq_dim = 6000 #？

        # training, cross-validation and test dataset
        cross_validation_sequences = ['2011_09_30_drive_0028_extract']
        test_sequences = ['2011_09_30_drive_0028_extract']
        continue_training = True

        # choose what to do launch() 中调用
        read_data = 0 #1表示从raw data 中重新生成pickle文件
        train_filter = 0
        test_filter = 0 #1
        results_filter = 1 # 是否读取预测结果
        dataset_class = KITTIDataset #类
        parameter_class = KITTIParameters #类


if __name__ == '__main__':
    args = KITTIArgs()
    dataset = KITTIDataset(args)
    launch(KITTIArgs)

