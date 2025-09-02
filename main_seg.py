from __future__ import print_function
from csv import writer
import math
import os
import argparse
import torch
import glob
import h5py
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ModelNet, load_my_data
from model import DGCNN, PointNetSeg, PointNet2Seg, PointCNN, RandLABridgeNet, ElevationWeightedDGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream, compute_class_weights, compute_class_weights1
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import time

global room_seg
room_seg = []
global room_pred
room_pred = []
global visual_warning
visual_warning = True

def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    os.system('cp main_semseg_s3dis.py outputs'+'/'+args.exp_name+'/'+'main_semseg_s3dis.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')


def calculate_sem_IoU(pred_np, seg_np, visual=False):
    """
    计算语义分割的交并比（IoU）。

    参数:
    pred_np -- 预测的标签数组，形状为 (134, 1024)
    seg_np -- 真实的标签数组，形状为 (134, 1024)
    visual -- 是否可视化结果，目前此参数未使用

    返回:
    IoU -- 每个类别的交并比，形状为 (6,)
    """
    num_classes = 6
    I_all = np.zeros(num_classes)
    U_all = np.zeros(num_classes)

    # 保证shape一致
    if seg_np.shape != pred_np.shape:
        seg_np = seg_np.reshape(pred_np.shape)
    # 遍历每个类别
    for sem in range(num_classes):
        # 计算交集 I
        I = np.sum(np.logical_and(pred_np == sem, seg_np == sem))
        # 计算并集 U
        U = np.sum(np.logical_or(pred_np == sem, seg_np == sem))
        # 避免除以零
        if U == 0:
            I_all[sem] = 0
            U_all[sem] = 1
        else:
            I_all[sem] = I
            U_all[sem] = U

    # 计算 IoU
    IoU = I_all / U_all
    return IoU 

class HybridAlphaScheduler:
    def __init__(self, total_epochs, val_loader, init_alpha=0.3, 
                 max_alpha=1.0, patience=3, delta=0.01):
        """
        混合α调度器
        :param total_epochs: 总训练轮数
        :param val_loader: 验证集数据加载器
        :param init_alpha: 初始α值
        :param max_alpha: 最大α值
        :param patience: 性能停滞容忍轮数
        :param delta: 显著改进阈值
        """
        self.total_epochs = total_epochs
        self.val_loader = val_loader
        self.current_alpha = init_alpha
        self.max_alpha = max_alpha
        self.min_alpha = 0.1
        self.patience = patience
        self.delta = delta
        self.best_miou = 0.0
        self.counter = 0
        self.stage = 0
        
        # 动态调度参数
        self.warmup_epochs = max(1, int(0.2 * total_epochs))
        self.plateau_epochs = max(1, int(0.6 * total_epochs))
        
    def adjust_alpha(self, model, epoch, current_miou=None):
        """动态调整α值"""
        # 阶段1：预热期(前20% epochs)
        if epoch < self.warmup_epochs:
            # 线性增长：0.3 → 0.6
            new_alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * (epoch / self.warmup_epochs) * 0.6
            
        # 阶段2：高原期(中间60% epochs)
        elif epoch < self.warmup_epochs + self.plateau_epochs:
            if current_miou is not None:
                if current_miou > self.best_miou + self.delta:
                    # 显著改进：奖励性增强
                    self.best_miou = current_miou
                    self.counter = 0
                    new_alpha = min(self.current_alpha * 1.15, self.max_alpha)
                elif current_miou > self.best_miou - self.delta:
                    # 平台期：维持当前α
                    self.counter += 1
                    new_alpha = self.current_alpha
                else:
                    # 性能下降：惩罚性减弱
                    self.counter = 0
                    new_alpha = max(self.current_alpha * 0.85, self.min_alpha)
                
                # 检查停滞计数器
                if self.counter >= self.patience:
                    new_alpha = min(new_alpha * 1.25, self.max_alpha)
                    self.counter = 0  # 重置计数器
            else:
                # 非验证轮次保持当前α
                new_alpha = self.current_alpha
                
        # 阶段3：收敛期(最后20% epochs)
        else:
            # 余弦衰减稳定收敛
            progress = (epoch - self.warmup_epochs - self.plateau_epochs) / \
                      (self.total_epochs - self.warmup_epochs - self.plateau_epochs)
            decay_factor = 0.5 * (1 + math.cos(math.pi * progress))
            new_alpha = self.current_alpha * decay_factor
        
        # 确保α在合理范围内
        new_alpha = max(self.min_alpha, min(new_alpha, self.max_alpha))
        
        # 更新当前α值
        self.current_alpha = new_alpha
        return self.current_alpha
    
def evaluate_validation(model, val_loader, device):
    """专门用于混合调度器的验证评估"""
    model.eval()
    total_iou = 0.0
    total_samples = 0

    
    with torch.no_grad():
        for data, seg in val_loader:
            seg = seg - 1
            #data = data.reshape(-1,6)
            seg = seg.reshape(-1)
            data, seg = torch.FloatTensor(data).to(device),  torch.LongTensor(seg).to(device)
            data = data.permute(0, 2, 1)
            
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            seg_np = seg.cpu().numpy()
            pred = seg_pred.max(dim=2)[1]
            pred_np = pred.detach().cpu().numpy()
            
            # 计算当前批次的mIoU
            batch_iou = calculate_sem_IoU(pred_np, seg_np)
            total_iou += np.mean(batch_iou) * data.size(0)
            total_samples += data.size(0)
    
    model.train()
    return total_iou / total_samples
        
def train(args, io):
    train_loader = DataLoader(ModelNet(partition='train', num_points=args.num_points, exp_name=args.exp_name), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet(partition='val', num_points=args.num_points, exp_name=args.exp_name), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")
    
    all_data, labels = load_my_data(partition='train', exp_name=args.exp_name)
    # 根据整个数据集计算类别权重
    class_weights = compute_class_weights(labels).to(device)
    
    # 添加混合α调度器初始化
    scheduler_alpha = HybridAlphaScheduler(
        total_epochs=args.epochs,
        val_loader=test_loader,  # 使用test_loader作为验证集
        init_alpha=0.3,
        max_alpha=1.0,
        patience=3,
        delta=0.005
    )
    all_fold_iou = []
    best_test_iou = 0
    best_fold_iou = 0
    # 初始化空列表，用于存储每个 epoch 的指标
    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []
    train_iou_list, test_iou_list = [], []

    #Try to load models
    if args.model == 'pointnet':
        model = PointNetSeg(6).to(device)
        criterion = cal_loss
    elif args.model == 'dgcnn':
        #model = DGCNN(args).to(device)
        model = ElevationWeightedDGCNN(args).to(device)
        criterion = cal_loss
    elif args.model == 'pointnet2':
        model = PointNet2Seg(6).to(device)
        criterion = cal_loss
    elif args.model == 'pointcnn':
        model = PointCNN().to(device)
        criterion = cal_loss
    elif args.model == 'randla-bridgenet':
        model = RandLABridgeNet().to(device)
        criterion = compute_class_weights1
    else:
        raise Exception("Not implemented")

    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # Load previous model state if resuming training
    if args.resume:# 从哪个epoch开始加载模型 每次加载需要修改路径！！！
        model.load_state_dict(torch.load(os.path.join(args.model_root, 'epoch199_model.t7')))  # Load the saved model state

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args.epochs)

    # 记录α值的列表
    alpha_values = []
    #criterion = my_cal_loss
    for epoch in range(args.epochs):
        # 在关键阶段执行验证 (预热结束/高原期开始/收敛期开始)
        # if epoch % 5 == 0 or epoch in [scheduler_alpha.warmup_epochs, 
        #                               scheduler_alpha.warmup_epochs + scheduler_alpha.plateau_epochs]:
        #     # 执行验证获取当前mIoU
        #     val_miou = evaluate_validation(model, test_loader, device)
        # else:
        #     val_miou = None  # 非验证轮次使用None
        
        # # 调整α值
        # current_alpha = scheduler_alpha.adjust_alpha(model, epoch, val_miou)
        current_alpha = args.alpha  # 使用命令行参数指定的α值
        # 更新模型α值 (注意DataParallel包装)
        model.module.alpha = current_alpha
        alpha_values.append(current_alpha)
        
        io.cprint(f"Epoch {epoch}: Current alpha = {current_alpha:.4f}")
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        class_correct = {}
        class_total = {}
        class_accuracy ={}
        for data, seg in train_loader:
            # 将标签减去1，使其范围在[0, 5]之间
            seg = seg - 1
            #data = data.reshape(-1,6)
            seg = seg.reshape(-1)
            data, seg = torch.FloatTensor(data).to(device),  torch.LongTensor(seg).to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            seg_np = seg.cpu().numpy()
            #loss = criterion(seg_pred.view(-1, 6), seg.view(-1,1).squeeze(), data, 0.001, 0.01, class_weights)
            loss = criterion(seg_pred.view(-1, 6), seg.view(-1,1).squeeze(), class_weights)
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            for label in range(6):
                pred_np = pred_np.reshape(-1)
                seg_np = seg_np.reshape(-1)
                class_correct[label] = class_correct.get(label, 0) + (pred_np[seg_np == label] == label).sum()
                class_total[label] = class_total.get(label, 0) + (seg_np == label).sum()
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        train_loss_list.append(train_loss*1.0/count)
        train_acc_list.append(train_acc)
        train_iou_list.append(np.mean(train_ious))
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
                                                                                                train_loss*1.0/count,
                                                                                                train_acc,
                                                                                                avg_per_class_acc,
                                                                                                np.mean(train_ious))
        io.cprint(outstr)
        
        # 计算每个类别的准确率
        for label in class_total:
            if class_total[label] > 0:
                class_accuracy[label] = class_correct[label] / class_total[label]
                outstr = 'Class {} accuracy: {:.6f}'.format(label, class_accuracy[label])
                io.cprint(outstr)
        
        # outstr = f'Epoch {epoch}: alpha={current_alpha}'
        # io.cprint(outstr)
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        class_correct = {}
        class_total = {}
        class_accuracy ={}
        for data, seg in test_loader:
            seg = seg - 1
            #data = data.reshape(-1,6)
            seg = seg.reshape(-1)
            # data, seg = data.to(device), seg.to(device)
            data, seg = torch.FloatTensor(data).to(device),  torch.LongTensor(seg).to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            seg_np = seg.cpu().numpy()
                
            loss = criterion(seg_pred.view(-1, 6), seg.view(-1,1).squeeze(), class_weights)
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            for label in range(6):
                pred_np = pred_np.reshape(-1)
                seg_np = seg_np.reshape(-1)
                class_correct[label] = class_correct.get(label, 0) + (pred_np[seg_np == label] == label).sum()
                class_total[label] = class_total.get(label, 0) + (seg_np == label).sum()
        
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        all_fold_iou.append(np.mean(test_ious))
        test_loss_list.append(test_loss*1.0/count)
        test_acc_list.append(test_acc)
        test_iou_list.append(np.mean(test_ious))
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                            test_loss*1.0/count,
                                                                                            test_acc,
                                                                                            avg_per_class_acc,
                                                                                            np.mean(test_ious))
        io.cprint(outstr)
        # 计算每个类别的准确率
        # 绘制并保存损失图
        plt.figure()
        plt.plot(range(epoch + 1), train_loss_list, 'bo-', label='Train Loss')
        plt.plot(range(epoch + 1), test_loss_list, 'ro-', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()
        plt.savefig(os.path.join(args.model_root, 'loss_over_epochs.png'))
        plt.close()

        # 绘制并保存精度图
        plt.figure()
        plt.plot(range(epoch + 1), train_acc_list, 'bo-', label='Train Accuracy')
        plt.plot(range(epoch + 1), test_acc_list, 'ro-', label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()
        plt.savefig(os.path.join(args.model_root, 'accuracy_over_epochs.png'))
        plt.close()

        # 绘制并保存 IoU 图
        plt.figure()
        plt.plot(range(epoch + 1), train_iou_list, 'bo-', label='Train IoU')
        plt.plot(range(epoch + 1), test_iou_list, 'ro-', label='Validation IoU')
        plt.xlabel('Epochs')
        plt.ylabel('IoU')
        plt.title('IoU Over Epochs')
        plt.legend()
        plt.savefig(os.path.join(args.model_root, 'iou_over_epochs.png'))
        plt.close()

        # 添加α值与性能关联分析
        if epoch > 10 and len(test_iou_list) > 10:
            corr_coef = np.corrcoef(alpha_values[-10:], test_iou_list[-10:])[0, 1]
            io.cprint(f"Alpha-IoU Correlation (last 10 epochs): {corr_coef:.4f}")
            
        for label in class_total:
            if class_total[label] > 0:
                class_accuracy[label] = class_correct[label] / class_total[label]
                class_total[label] = class_total.get(label, 0) + (seg_np == label).sum()
                outstr = 'Class {} accuracy: {:.6f}'.format(label, class_accuracy[label])
                io.cprint(outstr)
        torch.save(model.state_dict(), os.path.join(args.model_root, 'epoch%d_model.t7') %epoch)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(model.state_dict(), os.path.join(args.model_root, 'best_model_overall.t7'))
            outstr = 'best test iou: {:.6f}, epoch {:d}'.format(best_test_iou, epoch)
            io.cprint(outstr)

    # 训练结束后保存α值曲线
    plt.figure(figsize=(12, 8))
    plt.plot(range(args.epochs), alpha_values, 'g-', linewidth=2, label='Alpha Value')
    plt.plot(range(args.epochs), test_iou_list, 'b-', linewidth=2, label='Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Dynamic Alpha Scheduling and Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.model_root, 'final_alpha_performance.png'))
    plt.close()
 
def test(args, io):
    # 获取所有测试文件路径
    test_files = glob.glob(os.path.join(args.data_dir, '*.h5'))

    device = torch.device("cuda" if args.cuda else "cpu")
    
    # Try to load models
    if args.model == 'pointnet':
        model = PointNetSeg(6).to(device)
    elif args.model == 'dgcnn':
        #model = DGCNN(args).to(device)
        model = ElevationWeightedDGCNN(args).to(device)
    elif args.model == 'pointnet2':
        model = PointNet2Seg(6).to(device)
    elif args.model == 'pointcnn':
        model = PointCNN(6).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    start_time_model = time.time()
    #model.load_state_dict(torch.load(os.path.join(args.model_root, 'epoch317_model.t7')))
    model.load_state_dict(torch.load(os.path.join(args.model_root, 'best_model_overall.t7')))
    print(f"Model Load Time: {time.time() - start_time_model}")
    
    model = model.eval()
    test_ious_list = []
    corrected_ious_list = []
    test_acc_list = []
    corrected_acc_list = []
    test_mRecall_list = []  # 用于存储每个文件的平均召回率
    corrected_mRecall_list = []
    class_acc_list = []  # 用于存储每个文件中每个类别的准确率
    corrected_class_acc_list = []  # 用于存储标签矫正后的每个类别的准确率
    test_mAcc_list = []  # 用于存储每个文件的mAcc
    corrected_mAcc_list = []  # 用于存储矫正后的每个文件的mAcc

    for test_file in test_files:
        start_time = time.time()
        # 加载测试数据
        f = h5py.File(test_file, 'r+')
        all_data = f['data'][:].astype('float32')
        all_labels = f['label'][:].astype('int64')
        f.close()
        all_labels = all_labels - 1
        # 预加载数据到 GPU
        all_data_tensor = torch.Tensor(all_data).to(device)
        all_label_tensor = torch.LongTensor(all_labels).to(device)

        # 创建DataLoader
        test_dataset = torch.utils.data.TensorDataset(all_data_tensor, all_label_tensor)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
        print(f"Data Loading Time: {time.time() - start_time}")
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        class_correct = {}
        class_total = {}
        class_accuracy = {}
        all_time = 0
        # 统计算法预测时间
        start_time = time.time()  # 统计整个测试文件的处理时间
        for data, seg in test_loader:
            data = data.permute(0, 2, 1)
            
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1)
            pred = seg_pred.argmax(dim=2)  # 使用argmax而不是max
            
            test_true_cls.append(seg.cpu().numpy().reshape(-1))
            test_pred_cls.append(pred.detach().cpu().numpy().reshape(-1))
            # 真实标签和预测标签
            test_true_seg.append(seg.cpu().numpy())
            test_pred_seg.append(pred.detach().cpu().numpy())
        end_time = time.time() 
        all_time += end_time - start_time   
        outstr = 'File: {} :: 运行时间: {:.6f}'.format(
                os.path.basename(test_file), end_time - start_time)
        io.cprint(outstr)

        # 计算整体准确率
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        test_ious_list.append(np.mean(test_ious))
        test_acc_list.append(avg_per_class_acc)
        
        # 计算每个类别的召回率 (Recall) 并求平均
        per_class_recall = metrics.recall_score(test_true_cls, test_pred_cls, average=None, labels=np.unique(test_true_cls))
        avg_recall = np.mean(per_class_recall)  # 计算该文件的平均召回率
        test_mRecall_list.append(avg_recall)
        
        # 计算每个类别的准确率
        per_class_accuracy = metrics.precision_score(test_true_cls, test_pred_cls, average=None, labels=np.unique(test_true_cls))
        class_acc_list.append(per_class_accuracy)  # 保存每个类别的准确率
        
        # 计算mAcc
        mAcc = np.mean(per_class_accuracy)  # 计算mAcc
        test_mAcc_list.append(mAcc)  # 保存mAcc

        outstr = 'File: {} :: test acc: {:.6f}, test avg acc: {:.6f}, test iou: {:.6f}, mRecall: {:.6f}, mAcc: {:.6f}'.format(
            os.path.basename(test_file), test_acc, avg_per_class_acc, np.mean(test_ious), avg_recall, mAcc)
        io.cprint(outstr)
        
        # 保存每个文件的可视化结果
        save_visualization(all_data, test_pred_seg, args.output_dir, os.path.basename(test_file))
        
        # 标签矫正
        corrected_labels = correct_point_cloud_labels(all_data, test_pred_seg, test_file, k=50)
        corrected_labels = correct_labels(all_data, corrected_labels, k=70)
        
        corrected_ious = calculate_sem_IoU(corrected_labels, test_true_seg)
        corrected_ious_list.append(np.mean(corrected_ious))
        corrected_acc = metrics.accuracy_score(test_true_cls, corrected_labels.reshape(-1))
        
        corrected_avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, corrected_labels.reshape(-1))
        corrected_acc_list.append(corrected_avg_per_class_acc)
        per_class_recall = metrics.recall_score(test_true_cls, corrected_labels.reshape(-1), average=None, labels=np.unique(test_true_cls))
        avg_recall = np.mean(per_class_recall)  # 计算该文件的平均召回率
        corrected_mRecall_list.append(avg_recall) 
        corrected_per_class_accuracy = metrics.precision_score(test_true_cls, corrected_labels.reshape(-1), average=None, labels=np.unique(test_true_cls))
        corrected_class_acc_list.append(corrected_per_class_accuracy)  # 保存矫正后的每个类别的准确率
        # 计算矫正后的mAcc
        corrected_mAcc = np.mean(corrected_per_class_accuracy)  # 计算mAcc
        corrected_mAcc_list.append(corrected_mAcc)  # 保存矫正后的mAcc
        
        outstr_corrected = 'File: {} :: corrected test acc: {:.6f}, corrected test avg acc: {:.6f}, corrected test iou: {:.6f}, mRecall: {:.6f}, corrected mAcc: {:.6f}'.format(
            os.path.basename(test_file), corrected_acc, corrected_avg_per_class_acc, np.mean(corrected_ious), avg_recall, corrected_mAcc)
        io.cprint(outstr_corrected)
        save_visualization(all_data, corrected_labels, args.output_dir + '/K1=50/', os.path.basename(test_file) + '_corrected')
    
    outstr = 'mean test acc: {:.6f}'.format(np.mean(test_acc_list))
    io.cprint(outstr)    
    outstr = 'mean test iou: {:.6f}'.format(np.mean(test_ious_list))
    io.cprint(outstr)
    outstr = 'mean mRecall: {:.6f}'.format(np.mean(test_mRecall_list))  # 输出平均召回率
    io.cprint(outstr)
    outstr = 'mean corrected acc: {:.6f}'.format(np.mean(corrected_acc_list))    
    io.cprint(outstr)
    outstr = 'mean corrected test iou: {:.6f}'.format(np.mean(corrected_ious_list))
    io.cprint(outstr)
    outstr = 'mean corrected mRecall: {:.6f}'.format(np.mean(corrected_mRecall_list))  # 输出平均召回率
    io.cprint(outstr)
    
    # 输出mAcc
    outstr = 'mean mAcc: {:.6f}'.format(np.mean(test_mAcc_list))  # 输出mAcc
    io.cprint(outstr)
    outstr = 'mean corrected mAcc: {:.6f}'.format(np.mean(corrected_mAcc_list))  # 输出矫正后的mAcc
    io.cprint(outstr)
    
    # 输出每个类别的准确率的平均值
    class_acc_avg = np.mean(np.array(class_acc_list), axis=0)  # 计算每个类别准确率的平均值
    corrected_class_acc_avg = np.mean(np.array(corrected_class_acc_list), axis=0)  # 矫正后的类别准确率的平均值

    for i, class_acc in enumerate(class_acc_avg):
        outstr = 'Average class accuracy for class {}: {:.6f}'.format(i, class_acc)
        io.cprint(outstr)

    for i, corrected_class_acc in enumerate(corrected_class_acc_avg):
        outstr = 'Average corrected class accuracy for class {}: {:.6f}'.format(i, corrected_class_acc)
        io.cprint(outstr)

def correct_labels(batch_points, batch_pred_labels, k=100):
    
    # xyz用于最近点搜索
    n, num_points, snum_features   = batch_points.shape
    
    batch_points = batch_points[:,:,[0, 2]]
    #corrected_labels = np.zeros_like(batch_pred_labels)
    corrected_labels = batch_pred_labels
    start_time = time.time()
    for i in range(n):
        points = batch_points[i]
        #pred_labels = batch_pred_labels[i]
        pred_labels = corrected_labels[i]
        # 使用 NearestNeighbors 搜索最近邻点
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        for j in range(num_points):
            # 获取第 j 个点的最近邻点的索引（排除自身）
            neighbor_indices = indices[j, 1:]
            # 获取最近邻点的预测标签
            neighbor_labels = pred_labels[neighbor_indices]
            valid_labels = (neighbor_labels < pred_labels[j] + 3)
            # 应用过滤条件，保留有效的标签
            neighbor_labels = neighbor_labels[valid_labels]
            if len(neighbor_labels) > 0:
                # 选择最频繁出现的标签作为矫正后的标签
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                corrected_labels[i, j] = unique_labels[np.argmax(counts)]
    print("time:%.4f", time.time() - start_time)
    return corrected_labels

def get_deviation_from_log(las_file_path, log_file_path = 'E:/lcy/dgcnn/output/log.txt'):
    # 读取日志文件并提取偏差值
    with open(log_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    deviations = {}
    las_file_name = las_file_path.split("\\")[-1].split(".")[0]  # 获取.las文件的名称（不含扩展名）
    for line in lines:
        # 查找包含"正在处理文件"的行，以获取日志中对应的.las文件名
        if "正在处理文件" in line:
            parts = line.split(": ")
            if len(parts) > 1:
                # 提取日志中的文件路径，并获取文件名（不含扩展名）
                log_file_name_with_path = parts[-1].strip()
                log_file_name = log_file_name_with_path.split("/")[-1].split(".")[0]
                # 检查日志行中的文件名是否与提供的文件名匹配
                if las_file_name == log_file_name:
                    # 找到匹配的文件，现在查找对应的偏差值
                    for subsequent_line in lines[lines.index(line)+1:]:
                        if "坐标偏差值" in subsequent_line:
                            parts = subsequent_line.split("：")
                            if len(parts) > 1:
                                deviation = float(parts[1].strip())
                                deviations[las_file_path] = deviation / 2  # 将偏差值除以2并存储
                            break  # 找到偏差值后跳出循环
    return deviations

def correct_point_cloud_labels(points, labels, las_file_path, k=5):
     # 展平点云数据和标签
    num_samples, num_points, num_features = points.shape
    points = points[:, :, :3].reshape(-1, 3)  # 使用前三个维度作为坐标
    labels = labels.reshape(-1)
    # 轨枕外侧不可能出现 3, 4, 5 类
    invalid_outer = {3, 4, 5}
    # 轨枕内侧不可能出现 0, 1, 2 类
    invalid_inner = {0, 1, 2}
     # 使用 k-NN 算法找到最近的 k 个点
    start_time = time.time()
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    corrected_labels = labels.copy()
    deviation = get_deviation_from_log(las_file_path, 'E:/lcy/dgcnn/output/log.txt').get(las_file_path, 0)
    for i in range(len(points)):
        point = points[i]
        label = labels[i]
        
        # 判断点在轨枕内侧还是外侧
        is_outer = point[2] > 1.2 + deviation or point[2] < -1.2 + deviation
        is_invalid = (is_outer and label in invalid_outer) or (not is_outer and label in invalid_inner)
        
        if is_invalid:
            # 获取 k 个最近的点，排除当前点自身
            nearest_indices = indices[i][1:]
            nearest_labels = corrected_labels[nearest_indices]
            
            # 统计 k 个最近点的标签频次
            valid_labels = []
            for j, neighbor_idx in enumerate(nearest_indices):
                neighbor_label = nearest_labels[j]
                neighbor_point = points[neighbor_idx]
                neighbor_is_outer = neighbor_point[2] > 1.2 + deviation or point[2] < -1.2 + deviation
                neighbor_is_invalid = (neighbor_is_outer and neighbor_label in invalid_outer) or (not neighbor_is_outer and neighbor_label in invalid_inner)
                
                if not neighbor_is_invalid:
                     valid_labels.append(neighbor_label)
            
            if valid_labels:
                # 选择出现次数最多的有效标签
                corrected_label = max(set(valid_labels), key=valid_labels.count)
                corrected_labels[i] = corrected_label
            # else:
            #     print(f"Warning: No valid labels found for point {i}. Keeping original label.")
        if not is_outer:
            if point[2] > 1 + deviation:
                if corrected_labels[i] != 5 and corrected_labels[i] != 4:
                     # 获取k个最近的邻居，排除当前点自身
                    nearest_indices = indices[i][1:]  # 排除当前点自身
                    nearest_labels = corrected_labels[nearest_indices]
                    # 获取邻域标签，排除当前点自身标签
                    valid_labels = [neighbor_label for neighbor_label in nearest_labels if neighbor_label == 4 or neighbor_label == 5]
                    
                    if valid_labels:
                        # 选择出现次数最多的标签（4 或 5）
                        corrected_label = max(set(valid_labels), key=valid_labels.count)
                        corrected_labels[i] = corrected_label
    print("time:%.4f", time.time() - start_time)
    # 返回label形状与输入保持一致
    return corrected_labels.reshape(-1, 1024)

def save_visualization(data, pred, output_dir, filename):
    """
    保存点云分割结果的可视化文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保 pred 和 data 已经是 numpy.ndarray
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # 如果 data 是列表，先转换为 numpy 数组
    if isinstance(data, list):
        data = np.concatenate(data, axis=0)  # 将 data 列表合并为一个 numpy 数组
    
    # 确保 pred 是 numpy 数组
    if isinstance(pred, list):
        pred = np.concatenate(pred, axis=0)  # 将 pred 列表合并为一个 numpy 数组
    
    pred_colors = label_to_color(pred)

    # 合并数据 (xyz + pred_colors + label)
    result = np.concatenate([data[:, :, :3], pred_colors, pred[:, :, None]], axis=2)
    
    # 将 result 变为二维数组
    result = result.reshape(-1, 7)
    # 将 .h5 后缀修改为 .txt 后缀
    if filename.endswith('.h5'):
        filename = filename[:-3] + '.txt'
    elif not filename.endswith('.txt'):
        filename += '.txt'
    # 保存为 TXT 文件
    output_file = os.path.join(output_dir, filename)
    np.savetxt(output_file, result, fmt='%f %f %f %d %d %d %d')
def label_to_color(pred):
    color_map = {
        0: [255, 0, 0],    # 红色
        1: [0, 255, 0],    # 绿色
        2: [0, 0, 255],    # 蓝色
        3: [255, 255, 0],  # 黄色
        4: [0, 255, 255],  # 青色
        5: [255, 0, 255],  # 品红
        6: [128, 0, 0],    # 深红
    }
    pred_colors = np.array([color_map[label] for label in pred.flatten()])
    return pred_colors.reshape(pred.shape[0], pred.shape[1], 3)
def count_class_points(data_loader):
    # 初始化类别计数字典
    class_counts = {}

    for data, seg in data_loader:
        # 遍历每个批次
        for label in seg.unique():
            count = (seg == label).sum().item()
            if label.item() in class_counts:
                class_counts[label.item()] += count
            else:
                class_counts[label.item()] = count

    return class_counts
if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    #parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
    parser.add_argument('--exp_name', type=str, default='seg_1024_1', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--alpha', type=float, default='0.75', 
                        help='加强权重系数')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',#下次运行代码记得改保存模型的那行！！
                        help='number of episode to train ')
    parser.add_argument('--resume', type=str, default=False,
                        help='Whether to load the model')
    parser.add_argument('--eval', type=bool,  default=True,
                        help='evaluate the model')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                        choices=['S3DIS'])
    parser.add_argument('--test_area', type=str, default=None, metavar='N',
                        choices=['1', '2', '3', '4', '5', '6', 'all'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=2048, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_root', type=str, default='E:/lcy/dgcnn/outputs/coordinate_correction/dgcnn/seg_1024_rgbrmuda/models/alpha0.75/', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--visu', type=str, default='',
                        help='visualize the model')
    parser.add_argument('--visu_format', type=str, default='ply',
                        help='file format of visualization')
    parser.add_argument('--data_dir', type=str, default='E:/lcy/dgcnn/data/coordinate_correction/seg_1024_1/test/',
                        help='file of test_data')
    parser.add_argument('--output_dir', type=str, default='E:/lcy/dgcnn/data/coordinate_correction/test_result/dgcnn/seg_1024_rgbrmuda/alpha0.75/',
                        help='test result')
    args = parser.parse_args()

    _init_()

    #io = IOStream('outputs/' + args.exp_name + '/run.log')
    io = IOStream(args.model_root + '/run.log')

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        io = IOStream(args.output_dir + '/run.log')
        test(args, io)

