import matplotlib
import os
import sys
sys.path.append('/home/dlr/Project/ai-imu-dr/tools')
from termcolor import cprint
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from utils import *
from utils_torch_filter import TORCHIEKF
from difftime import *
# from main_kitti import KITTIDataset


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

# 从.p中读取结果 绘图 
def results_filter(args, dataset):

    for i in range(0, len(dataset.datasets)):
        plt.close('all')
        dataset_name = dataset.dataset_name(i)
        file_name = os.path.join(dataset.path_results, dataset_name + "_filter.p")
        if not os.path.exists(file_name): # data/中不是每个序列都有结果 目前只有10个序列的结果
            print('No result for ' + dataset_name)
            continue

        seqid = odometry_seqid[dataset_name]
        print("\nResults for: " + dataset_name + ", " + seqid)

        # 得到估计的结果 增加协方差
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, measurements_covs, Pbuffer = dataset.get_estimates(
            dataset_name)

        # get data 有时间戳 真值位姿 u是imu输入： 加速度 角速度
        t, ang_gt, p_gt, v_gt, u, t0 = dataset.get_data(dataset_name)
        # get data for nets
        u_normalized = dataset.normalize(u).numpy()
        # shift for better viewing
        u_normalized[:, [0, 3]] += 5
        u_normalized[:, [2, 5]] -= 5

        # tsave = t.numpy().copy() #深拷贝时间戳
        # tsave = (tsave + t0) # 恢复原始时间 .numpy()
        # print("look t[0] = {:f}, raw t0 = {:f}".format(t[0], t0))
        t = (t - t[0]).numpy() #其实t[0]已经是0了 上面输出已经做验证
        u = u.cpu().numpy()
        ang_gt = ang_gt.cpu().numpy()
        v_gt = v_gt.cpu().numpy()
        p_gt = (p_gt - p_gt[0]).cpu().numpy()
        print("Total sequence time: {:.2f} s".format(t[-1]))

        ang = np.zeros((Rot.shape[0], 3))
        Rot_gt = torch.zeros((Rot.shape[0], 3, 3))
        for j in range(Rot.shape[0]):
            roll, pitch, yaw = TORCHIEKF.to_rpy(torch.from_numpy(Rot[j]))
            ang[j, 0] = roll.numpy()
            ang[j, 0] = pitch.numpy()
            ang[j, 0] = yaw.numpy()
        # unwrap
            Rot_gt[j] = TORCHIEKF.from_rpy(torch.Tensor([ang_gt[j, 0]]),
                                        torch.Tensor([ang_gt[j, 1]]),
                                        torch.Tensor([ang_gt[j, 2]]))
            roll, pitch, yaw = TORCHIEKF.to_rpy(Rot_gt[j])
            ang_gt[j, 0] = roll.numpy()
            ang_gt[j, 0] = pitch.numpy()
            ang_gt[j, 0] = yaw.numpy()

        Rot_align, t_align, _ = umeyama_alignment(p_gt[:, :3].T, p[:, :3].T)
        p_align = (Rot_align.T.dot(p[:, :3].T)).T - Rot_align.T.dot(t_align) #只用于绘图 不评估
        v_norm = np.sqrt(np.sum(v_gt ** 2, 1))
        v_norm /= np.max(v_norm)

        # plot and save plot
        folder_path = os.path.join(args.path_results, dataset_name)
        create_folder(folder_path)

        # 保存 真值位姿Tw_imu 时间戳 伪测量y协方差N(measurements_covs)
        # timepath = os.path.join(folder_path, "ntimes.txt")

        # np.savetxt(timepath, tsave, fmt= '%.6f')
        
        # Compute various errors
        error_p = np.abs(p_gt - p)
        # MATE
        mate_xy = np.mean(error_p[:, :2], 1)
        mate_z = error_p[:, 2]

        # CATE
        cate_xy = np.cumsum(mate_xy)
        cate_z = np.cumsum(mate_z)

        # RMSE
        rmse_xy = 1 / 2 * np.sqrt(error_p[:, 0] ** 2 + error_p[:, 1] ** 2)
        rmse_z = error_p[:, 2]

        RotT = torch.from_numpy(Rot).float().transpose(-1, -2)

        v_r = (RotT.matmul(torch.from_numpy(v).float().unsqueeze(-1)).squeeze()).numpy()
        v_r_gt = (Rot_gt.transpose(-1, -2).matmul(
            torch.from_numpy(v_gt).float().unsqueeze(-1)).squeeze()).numpy()

        p_r = (RotT.matmul(torch.from_numpy(p).float().unsqueeze(-1)).squeeze()).numpy()
        p_bis = (Rot_gt.matmul(torch.from_numpy(p_r).float().unsqueeze(-1)).squeeze()).numpy()
        error_p = p_gt - p_bis

        # position, velocity and velocity in body frame
        fig1, axs1 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
        # orientation, bias gyro and bias accelerometer
        fig2, axs2 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
        # position in plan
        fig3, ax3 = plt.subplots(figsize=(20, 10))
        # position in plan after alignment
        fig4, ax4 = plt.subplots(figsize=(20, 10))
        #  Measurement covariance in log scale and normalized inputs
        fig5, axs5 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
        # input: gyro, accelerometer
        fig6, axs6 = plt.subplots(2, 1, sharex=True, figsize=(20, 10))
        # errors: MATE, CATE  RMSE
        fig7, axs7 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))

        axs1[0].plot(t, p_gt)
        axs1[0].plot(t, p)
        axs1[1].plot(t, v_gt)
        axs1[1].plot(t, v)
        axs1[2].plot(t, v_r_gt) # 实际是imu坐标系下
        axs1[2].plot(t, v_r)

        axs2[0].plot(t, ang_gt)
        axs2[0].plot(t, ang)
        axs2[1].plot(t, b_omega)
        axs2[2].plot(t, b_acc)

        ax3.plot(p_gt[:, 0], p_gt[:, 1])
        ax3.plot(p[:, 0], p[:, 1])
        ax3.axis('equal')
        ax4.plot(p_gt[:, 0], p_gt[:, 1])
        ax4.plot(p_align[:, 0], p_align[:, 1])
        ax4.axis('equal')

        axs5[0].plot(t, np.log10(measurements_covs))
        axs5[1].plot(t, u_normalized[:, :3]) #归一化后的角速度
        axs5[2].plot(t, u_normalized[:, 3:]) #归一化后的加速度

        axs6[0].plot(t, u[:, :3]) #原始的角速度
        axs6[1].plot(t, u[:, 3:6]) #原始的加速度

        axs7[0].plot(t, mate_xy)
        axs7[0].plot(t, mate_z)
        axs7[0].plot(t, rmse_xy)
        axs7[0].plot(t, rmse_z)
        axs7[1].plot(t, cate_xy)
        axs7[1].plot(t, cate_z)
        axs7[2].plot(t, error_p)

        axs1[0].set(xlabel='time (s)', ylabel='$\mathbf{p}_n$ (m)', title="Position")
        axs1[1].set(xlabel='time (s)', ylabel='$\mathbf{v}_n$ (m/s)', title="Velocity")
        axs1[2].set(xlabel='time (s)', ylabel='$\mathbf{R}_n^T \mathbf{v}_n$ (m/s)',
                    title="Velocity in body frame")
        axs2[0].set(xlabel='time (s)', ylabel=r'$\phi_n, \theta_n, \psi_n$ (rad)',
                    title="Orientation")
        axs2[1].set(xlabel='time (s)', ylabel=r'$\mathbf{b}_{n}^{\mathbf{\omega}}$ (rad/s)',
                    title="Bias gyro")
        axs2[2].set(xlabel='time (s)', ylabel=r'$\mathbf{b}_{n}^{\mathbf{a}}$ (m/$\mathrm{s}^2$)',
                    title="Bias accelerometer")
        ax3.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Position on $xy$")
        ax4.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Aligned position on $xy$")
        axs5[0].set(xlabel='time (s)', ylabel=r' $\mathrm{cov}(\mathbf{y}_{n})$ (log scale)',
                     title="Covariance on the zero lateral and vertical velocity measurements (log "
                           "scale)")
        axs5[1].set(xlabel='time (s)', ylabel=r'Normalized gyro measurements',
                     title="Normalized gyro measurements")
        axs5[2].set(xlabel='time (s)', ylabel=r'Normalized accelerometer measurements',
                   title="Normalized accelerometer measurements")
        axs6[0].set(xlabel='time (s)', ylabel=r'$\omega^x_n, \omega^y_n, \omega^z_n$ (rad/s)',
                    title="Gyrometer")
        axs6[1].set(xlabel='time (s)', ylabel=r'$a^x_n, a^y_n, a^z_n$ (m/$\mathrm{s}^2$)',
                    title="Accelerometer")
        axs7[0].set(xlabel='time (s)', ylabel=r'$|| \mathbf{p}_{n}-\hat{\mathbf{p}}_{n} ||$ (m)',
                    title="Mean Absolute Trajectory Error (MATE) and Root Mean Square Error (RMSE)")
        axs7[1].set(xlabel='time (s)',
                    ylabel=r'$\Sigma_{i=0}^{n} || \mathbf{p}_{i}-\hat{\mathbf{p}}_{i} ||$ (m)',
                    title="Cumulative Absolute Trajectory Error (CATE)")
        axs7[2].set(xlabel='time (s)', ylabel=r' $\mathbf{\xi}_{n}^{\mathbf{p}}$',
                    title="$SE(3)$ error on position")

        for ax in chain(axs1, axs2, axs5, axs6, axs7):
            ax.grid()
        ax3.grid()
        ax4.grid()
        axs1[0].legend(
            ['$p_n^x$', '$p_n^y$', '$p_n^z$', '$\hat{p}_n^x$', '$\hat{p}_n^y$', '$\hat{p}_n^z$'])
        axs1[1].legend(
            ['$v_n^x$', '$v_n^y$', '$v_n^z$', '$\hat{v}_n^x$', '$\hat{v}_n^y$', '$\hat{v}_n^z$'])
        axs1[2].legend(
            ['$v_n^x$', '$v_n^y$', '$v_n^z$', '$\hat{v}_n^x$', '$\hat{v}_n^y$', '$\hat{v}_n^z$'])
        axs2[0].legend([r'$\phi_n^x$', r'$\theta_n^y$', r'$\psi_n^z$', r'$\hat{\phi}_n^x$',
                        r'$\hat{\theta}_n^y$', r'$\hat{\psi}_n^z$'])
        axs2[1].legend(
            ['$b_n^x$', '$b_n^y$', '$b_n^z$', '$\hat{b}_n^x$', '$\hat{b}_n^y$', '$\hat{b}_n^z$'])
        axs2[2].legend(
            ['$b_n^x$', '$b_n^y$', '$b_n^z$', '$\hat{b}_n^x$', '$\hat{b}_n^y$', '$\hat{b}_n^z$'])
        ax3.legend(['ground-truth trajectory', 'proposed'])
        ax4.legend(['ground-truth trajectory', 'proposed'])
        axs5[0].legend(['zero lateral velocity', 'zero vertical velocity'])
        axs6[0].legend(['$\omega_n^x$', '$\omega_n^y$', '$\omega_n^z$'])
        axs6[1].legend(['$a_n^x$', '$a_n^y$', '$a_n^z$'])
        if u.shape[1] > 6:
            axs6[2].legend(['$m_n^x$', '$m_n^y$', '$m_n^z$'])
        axs7[0].legend(['MATE xy', 'MATE z', 'RMSE xy', 'RMSE z'])
        axs7[1].legend(['CATE xy', 'CATE z'])

        # save figures
        figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, ]
        figs_name = ["position_velocity", "orientation_bias", "position_xy", "position_xy_aligned",
                     "measurements_covs", "imu", "errors", "errors2"]
        for l, fig in enumerate(figs):
            fig_name = figs_name[l]
            fig.savefig(os.path.join(folder_path, fig_name + ".png"))

        # plt.show(block=True)
        

# 保存位姿 以KITTI格式
def saveposeKITTI(R,p, savefile, flag):
    nframe = R.shape[0] # 总帧数
    SE3 = np.ones((nframe, 4, 4))

    for i in range(nframe):
        SE3[i, 0:3, 0:3] = R[i]
        SE3[i, 0:3, 3] = p[i]
    
    if flag: # 是轨迹时才 转化为以初始帧为参考
        T0 = SE3[0].copy() #注意这里 否则会变化
        T0inv = np.linalg.inv(T0)
        for i in range(nframe):
            Ti = SE3[i].copy()
            SE3[i] = np.matmul(T0inv, Ti)
    else:
        print("Not trajectory, No T0inv")
    
    
    #写入文件 
    f = open(savefile , "w") 
    for j in range(nframe):
        slidt = [SE3[j,0,0], SE3[j,0,1], SE3[j,0,2], SE3[j,0,3],
                    SE3[j,1,0], SE3[j,1,1], SE3[j,1,2], SE3[j,1,3],
                    SE3[j,2,0], SE3[j,2,1], SE3[j,2,2], SE3[j,2,3]]
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


# 保存IEKF输出的协方差 注意已经是6*6
def saveP66(P6, savefile):
    nframe = P6.shape[0] # 总帧数
    #写入文件 
    f = open(savefile , "w") 
    for j in range(nframe):
        slidt = [P6[j,0,0], P6[j,0,1], P6[j,0,2], P6[j,0,3], P6[j,0,4], P6[j,0,5],
                 P6[j,1,0], P6[j,1,1], P6[j,1,2], P6[j,1,3], P6[j,1,4], P6[j,1,5],
                 P6[j,2,0], P6[j,2,1], P6[j,2,2], P6[j,2,3], P6[j,2,4], P6[j,2,5],
                 P6[j,3,0], P6[j,3,1], P6[j,3,2], P6[j,3,3], P6[j,3,4], P6[j,3,5],
                 P6[j,4,0], P6[j,4,1], P6[j,4,2], P6[j,4,3], P6[j,4,4], P6[j,4,5],
                 P6[j,5,0], P6[j,5,1], P6[j,5,2], P6[j,5,3], P6[j,5,4], P6[j,5,5],
                ]
        k = 0
        for v in slidt:
            f.write(str(v))
            if(k<35): # 最后一个无空格，直接换行
                f.write(' ')
            k = k + 1
        f.write('\n')
    f.close()

# 从结果中保存需要的内容
def results_save(args, dataset):

    for i in range(0, len(dataset.datasets)):
        plt.close('all')
        dataset_name = dataset.dataset_name(i)
        file_name = os.path.join(dataset.path_results, dataset_name + "_filter.p")
        if not os.path.exists(file_name): # data/中不是每个序列都有结果 目前只有10个序列的结果
            print('No result for ' + dataset_name)
            continue
        
        seqid = odometry_seqid[dataset_name]
        print("\n for: " + dataset_name + ", " + seqid)

        # 得到估计的结果 增加协方差
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, measurements_covs, Pbuffer = dataset.get_estimates(
            dataset_name)

        # get data 有时间戳 真值位姿 u是imu输入： 加速度 角速度
        t, ang_gt, p_gt, v_gt, u, t0 = dataset.get_data(dataset_name)
        
        ang_gt = ang_gt.cpu().numpy()
        # v_gt = v_gt.cpu().numpy()
        # p_gt = (p_gt - p_gt[0]).cpu().numpy()
        p_gt = p_gt.cpu().numpy()
        print("Total sequence time: {:.2f} s".format(t[-1]))

        ang = np.zeros((Rot.shape[0], 3))
        Rot_gt = torch.zeros((Rot.shape[0], 3, 3))
        for j in range(Rot.shape[0]):
            roll, pitch, yaw = TORCHIEKF.to_rpy(torch.from_numpy(Rot[j]))
            ang[j, 0] = roll.numpy()
            ang[j, 0] = pitch.numpy()
            ang[j, 0] = yaw.numpy()
        # unwrap
            Rot_gt[j] = TORCHIEKF.from_rpy(torch.Tensor([ang_gt[j, 0]]),
                                        torch.Tensor([ang_gt[j, 1]]),
                                        torch.Tensor([ang_gt[j, 2]]))
            roll, pitch, yaw = TORCHIEKF.to_rpy(Rot_gt[j])
            ang_gt[j, 0] = roll.numpy()
            ang_gt[j, 0] = pitch.numpy()
            ang_gt[j, 0] = yaw.numpy()

        Rot_gt = Rot_gt.cpu().numpy()
        # Rot_align, t_align, _ = umeyama_alignment(p_gt[:, :3].T, p[:, :3].T)
        # p_align = (Rot_align.T.dot(p[:, :3].T)).T - Rot_align.T.dot(t_align)
        # v_norm = np.sqrt(np.sum(v_gt ** 2, 1))
        # v_norm /= np.max(v_norm)

        # plot and save plot
        folder_path = os.path.join(args.path_results, dataset_name)
        create_folder(folder_path)

        # 保存 真值位姿Tw_imu, 估计的位姿 时间戳 伪测量y协方差N(measurements_covs) imu到车体的变换
        # timepath = os.path.join(folder_path, "ntimes.txt")
        gtposepath = os.path.join(folder_path, "imugt.txt")
        estposepath = os.path.join(folder_path, "imuest.txt") # 用来评估
        cov_ypath = os.path.join(folder_path, "covsy.txt")
        carimu = os.path.join(folder_path, "car_imu.txt") # Timu_body
        P6path = os.path.join(folder_path, "Psave.txt") #状态协方差 N 21,21->N 6 6

        # np.savetxt(timepath, tsave, fmt= '%.6f')
        
        # difftime(timepath) # 查看100hz时间是否异常 

        # indpath = os.path.join(folder_path, "ind_extract.txt")
        
        # #绘制图  查看 对应索引是否正常 0 2 5 6 8 都有异常
        # indextract = np.loadtxt(indpath)
        # N = indextract.shape[0]
        # x = np.arange(0, N)
        # l = plt.plot(x,indextract,marker='o', markerfacecolor='none', label='id')
        # plt.title(indpath)
        # plt.xlabel('frame in 10 hz')
        # plt.ylabel('index in 100hz')
        # plt.legend()
        # plt.show(block=True)
        print("save gt Twimu "+gtposepath)
        saveposeKITTI(Rot_gt, p_gt, gtposepath, True)

        saveposeKITTI(Rot, p, estposepath, True)

        print("save T_c_i "+carimu)
        saveposeKITTI(Rot_c_i, t_c_i, carimu, False)

        np.savetxt(cov_ypath, measurements_covs)

        N = Rot.shape[0]
        Psave = np.zeros((N, 6, 6)) # 只保存与位姿相关的协方差矩阵的值 R 和 p
        for i in range(N):
            bigP = Pbuffer[i].copy() # 21,21
            Psave[i, 0:3, 0:3] = bigP[0:3, 0:3]
            Psave[i, 0:3, 3:6] = bigP[0:3, 6:9]
            Psave[i, 3:6, 0:3] = bigP[6:9, 0:3]
            Psave[i, 3:6, 3:6] = bigP[6:9, 6:9]
        
        print("save P6 "+P6path)
        saveP66(Psave, P6path)


