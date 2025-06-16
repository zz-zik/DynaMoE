# -*- coding: utf-8 -*-
"""
@Project : DynaMoE
@FileName: trainer.py
@Time    : 2025/6/12 上午11:57
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 建筑物变化检测训练引擎
@Usage   :
"""
import pandas as pd
import numpy as np
import datetime
import random
import time
import warnings

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from tensorboardX import SummaryWriter
from tqdm import tqdm
from dataloader import build_dataset
from models import build_model
from utils import *

warnings.filterwarnings('ignore')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class Trainer:
    """建筑物变化检测训练引擎"""

    def __init__(self, cfg):
        """
        初始化训练引擎

        Args:
            cfg: 配置对象
        """
        self.cfg = cfg
        self.device = cfg.device

        self.classes = cfg.data.classes
        self.threshold = cfg.test.threshold

        # 初始化输出目录和日志
        self.output_dir = get_output_dir(cfg.output_dir, cfg.name)
        self.logger = setup_logging(cfg, self.output_dir)

        # 设置随机种子
        self._setup_seed()

        # 初始化模型和相关组件
        self._setup_model()
        self._setup_optimizer()
        self._setup_dataloader()

        # 初始化训练状态
        self._setup_training_state()

        # 初始化记录工具
        self._setup_logging_tools()

        # 恢复训练状态（如果需要）
        self._resume_if_needed()

    def _setup_seed(self):
        """设置随机种子"""
        seed = self.cfg.seed + get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _setup_model(self):
        """初始化模型和损失函数"""
        self.logger.info('------------------------ model params ------------------------')
        self.model, self.criterion = build_model(self.cfg, training=True)

        # 移动到GPU
        self.model.to(self.device)
        if isinstance(self.criterion, tuple) and len(self.criterion) == 2:
            for loss in self.criterion:
                loss.to(self.device)
        else:
            self.criterion.to(self.device)

        self.model_without_ddp = self.model
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info('number of params: %d', n_parameters)

    def _setup_optimizer(self):
        """设置优化器和学习率调度器"""
        # 对模型的不同部分使用不同的优化参数
        param_dicts = [{
            "params": [p for n, p in self.model_without_ddp.named_parameters()
                       if "backbone" not in n and p.requires_grad],
            "lr": self.cfg.training.lr
        }, {
            "params": [p for n, p in self.model_without_ddp.named_parameters()
                       if "backbone" in n and p.requires_grad],
            "lr": self.cfg.training.lr_backbone,
        }]

        self.optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay
        )

        # 配置学习率调度器
        self.lr_scheduler = None
        if self.cfg.training.scheduler == 'step':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.training.lr_drop,
                gamma=0.1
            )
        elif self.cfg.training.scheduler == 'plateau':
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=10,
                verbose=True
            )

        # 打印优化器信息
        self._log_optimizer_info()

    def _log_optimizer_info(self):
        """记录优化器信息"""
        optimizer_info = f"optimizer: Adam(lr={self.cfg.training.lr})"
        optimizer_info += " with parameter groups "
        for i, param_group in enumerate(self.optimizer.param_groups):
            optimizer_info += f"{len(param_group['params'])} weight(decay={param_group['weight_decay']}), "
        optimizer_info = optimizer_info.rstrip(', ')
        self.logger.info(optimizer_info)

    def _setup_dataloader(self):
        """设置数据加载器"""
        self.dataloader_train, self.dataloader_val = build_dataset(cfg=self.cfg)

    def _setup_training_state(self):
        """初始化训练状态"""
        self.start_epoch = self.cfg.training.start_epoch
        self.step = 0

        # 保存训练期间的指标（每个类）
        self.precision_list = []
        self.recall_list = []
        self.f1_dict_list = []
        self.iou_dict_list = []
        self.f1_avg_list = []
        self.iou_avg_list = []
        self.accuracy_list = []

    def _setup_logging_tools(self):
        """设置日志记录工具"""
        # 创建tensorboard
        tensorboard_dir = os.path.join(str(self.output_dir), 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_dir)

        # 初始化 CSV 文件
        self.csv_file_path = os.path.join(str(self.output_dir), 'result.csv')
        self.results_df = pd.DataFrame(columns=[
            'epoch', 'train_loss', 'train_oa',
            'val_loss', 'val_precision', 'val_recall', 'val_f1', 'val_iou', 'val_oa'
        ])

        # 创建检查点目录
        self.ckpt_dir = os.path.join(str(self.output_dir), 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _resume_if_needed(self):
        """如果需要，恢复训练状态"""
        if self.cfg.resume:
            self.logger.info('------------------------ Continue training ------------------------')
            logging.warning(f"loading from {self.cfg.resume}")
            checkpoint = torch.load(self.cfg.resume, map_location='cpu')
            self.model_without_ddp.load_state_dict(checkpoint['model'])

            if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.start_epoch = checkpoint['epoch'] + 1

    def _log_training_info(self):
        """记录训练开始信息"""
        self.logger.info('Train Log %s' % time.strftime("%c"))
        env_info = get_environment_info()
        self.logger.info(env_info)
        self.logger.info('Running with config:')
        self.logger.info(pprint.pformat(self.cfg.__dict__))

    def _train_one_epoch(self, epoch):
        """训练一个epoch"""
        t1 = time.time()
        stat = self.train(epoch)
        time.sleep(1)  # 避免tensorboard卡顿
        t2 = time.time()

        # 记录训练损失和OA
        if self.writer is not None:
            self.logger.info("[ep %d][lr %.7f][%.2fs] loss: %.4f, oa: %.4f",
                             epoch, self.optimizer.param_groups[0]['lr'], t2 - t1,
                             stat['loss'], stat['oa'])
            self.writer.add_scalar('loss/loss', stat['loss'], epoch)
            self.writer.add_scalar('metric/o_a', stat['oa'], epoch)

        # 更新训练指标
        self.results_df.loc[epoch] = {
            'epoch': epoch,
            'train_loss': stat['loss'],
            'train_oa': stat['oa'],
            'val_loss': '',
            'val_precision': '',
            'val_recall': '',
            'val_f1': '',
            'val_iou': '',
            'val_oa': ''
        }

        return stat

    def _adjust_learning_rate(self, metrics=None):
        """调整学习率"""
        if self.cfg.training.scheduler == 'step':
            self.lr_scheduler.step()
        elif self.cfg.training.scheduler == 'plateau' and metrics is not None:
            self.lr_scheduler.step(metrics['loss'])

    def _save_checkpoint(self, epoch, stat, checkpoint_type='latest'):
        """保存检查点"""
        checkpoint_data = {
            'epoch': epoch,
            'step': self.step,
            'model': self.model_without_ddp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'loss': stat['loss'],
        }

        if checkpoint_type == 'latest':
            checkpoint_path = os.path.join(self.ckpt_dir, 'latest.pth')
        elif checkpoint_type == 'best_f1':
            checkpoint_path = os.path.join(self.ckpt_dir, 'best_f1.pth')
        elif checkpoint_type == 'best_iou':
            checkpoint_path = os.path.join(self.ckpt_dir, 'best_iou.pth')
        else:
            checkpoint_path = os.path.join(self.ckpt_dir, f'{checkpoint_type}.pth')

        torch.save(checkpoint_data, checkpoint_path)

    def _evaluate_one_epoch(self, epoch):
        """评估一个epoch"""
        t1 = time.time()
        metrics = self.evaluate(epoch)
        t2 = time.time()

        # 更新指标列表
        for cls in self.classes:
            if cls not in self.f1_dict_list:
                self.f1_dict_list.append({})
            if cls not in self.iou_dict_list:
                self.iou_dict_list.append({})
            if cls not in self.precision_list:
                self.precision_list.append({})
            if cls not in self.recall_list:
                self.recall_list.append({})
            if cls not in self.accuracy_list:
                self.accuracy_list.append({})

            self.precision_list[-1][cls] = metrics['precision'][cls]
            self.recall_list[-1][cls] = metrics['recall'][cls]
            self.f1_dict_list[-1][cls] = metrics['f1'][cls]
            self.iou_dict_list[-1][cls] = metrics['iou'][cls]
            self.accuracy_list[-1][cls] = metrics['oa'][cls]

        fps = len(self.dataloader_val.dataset) / (t2 - t1)

        # 构造日志字符串
        metric_str = ", ".join(
            [f"{cls}: F1={metrics['f1'][cls]:.4f}, IoU={metrics['iou'][cls]:.4f}" for cls in self.classes])
        self.logger.info(
            "[ep %d][%.3fs][%.5ffps] loss: %.4f | %s ---- @best f1: %s" %
            (
                epoch, t2 - t1, fps, metrics['loss'],
                metric_str,
                ", ".join([f"{cls}: {np.max([m[cls] for m in self.f1_dict_list if cls in m])}" for cls in self.classes])
            )
        )

        # 记录到tensorboard
        if self.writer is not None:
            self.writer.add_scalar('metric/val_loss', metrics['loss'], self.step)
            for cls in self.classes:
                self.writer.add_scalar(f'metric/{cls}_precision', metrics['precision'][cls], self.step)
                self.writer.add_scalar(f'metric/{cls}_recall', metrics['recall'][cls], self.step)
                self.writer.add_scalar(f'metric/{cls}_f1', metrics['f1'][cls], self.step)
                self.writer.add_scalar(f'metric/{cls}_iou', metrics['iou'][cls], self.step)
                self.writer.add_scalar(f'metric/{cls}_oa', metrics['oa'][cls], self.step)
            self.step += 1

        return metrics

    def _save_best_models(self, epoch, stat, metrics):
        """保存最佳模型"""
        # 提取所有类别的 F1 和 IoU 值
        f1_values = np.array(list(metrics['f1'].values()))
        iou_values = np.array(list(metrics['iou'].values()))

        # 计算类别平均 F1 和 IoU
        avg_f1 = np.mean(f1_values)
        avg_iou = np.mean(iou_values)

        # 更新 F1 列表和 IoU 列表
        self.f1_avg_list.append(avg_f1)
        self.iou_avg_list.append(avg_iou)

        # 保存最好的 F1 模型
        if avg_f1 == np.max(self.f1_avg_list):
            self._save_checkpoint(epoch, stat, 'best_f1')

        # 保存最好的 IoU 模型
        if avg_iou == np.max(self.iou_avg_list):
            self._save_checkpoint(epoch, stat, 'best_iou')

    def _save_results_csv(self):
        """保存结果到CSV文件"""
        result_list = []
        for idx in range(len(self.results_df)):
            row = self.results_df.iloc[idx].to_dict()
            for cls in self.classes:
                row[f'{cls}_precision'] = self.precision_list[idx].get(cls, '')
                row[f'{cls}_recall'] = self.recall_list[idx].get(cls, '')
                row[f'{cls}_f1'] = self.f1_dict_list[idx].get(cls, '')
                row[f'{cls}_iou'] = self.iou_dict_list[idx].get(cls, '')
                row[f'{cls}_oa'] = self.accuracy_list[idx].get(cls, '')
            result_list.append(row)

        results_df = pd.DataFrame(result_list)
        results_df.to_csv(self.csv_file_path, index=False)

    def _log_final_summary(self, total_time):
        """记录最终总结"""
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        env_info = get_environment_info()

        self.logger.info('Summary of results')
        self.logger.info(env_info)
        self.logger.info('Training time {}'.format(total_time_str))

        # 打印最好的指标
        self.logger.info("Best F1: %.4f, Best IoU: %.4f, Best Accuracy: %.4f" % (
            np.max(self.f1_dict_list), np.max(self.iou_dict_list), np.max(self.accuracy_list)
        ))
        self.logger.info('Results saved to {}'.format(self.cfg.output_dir))

    def run(self):
        """执行训练过程"""
        # 记录训练开始信息
        self._log_training_info()

        self.logger.info("------------------------ Start training ------------------------")
        start_time = time.time()

        # 训练循环
        for epoch in range(self.start_epoch, self.cfg.training.epochs):
            # 训练一个epoch
            train_stat = self._train_one_epoch(epoch)

            # 调整学习率
            self._adjust_learning_rate()

            # 保存最新检查点
            self._save_checkpoint(epoch, train_stat, 'latest')

            # 评估
            if epoch % self.cfg.training.eval_freq == 0 and epoch >= self.cfg.training.start_eval:
                eval_metrics = self._evaluate_one_epoch(epoch)

                # 根据验证损失调整学习率（如果使用plateau调度器）
                self._adjust_learning_rate(eval_metrics)

                # 保存最佳模型
                self._save_best_models(epoch, train_stat, eval_metrics)

            # 保存结果到CSV
            self._save_results_csv()

        # 训练完成后的总结
        total_time = time.time() - start_time
        self._log_final_summary(total_time)

    def train(self, epoch):
        self.model.train()
        self.criterion.train()
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        total_pixels = 0

        with tqdm(self.dataloader_train, desc=f'Epoch {epoch} [Training]') as pbar:
            for images_a, images_b, labels in pbar:
                images_a = images_a.to(self.device)
                images_b = images_b.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images_a, images_b)
                losses = self.criterion(outputs, labels)
                loss = losses['total_loss']
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * images_a.size(0)
                total_samples += images_a.size(0)

                # Calculate accuracy
                preds = (torch.sigmoid(outputs['prediction']) > self.cfg.training.threshold).float().squeeze(1)
                correct = (preds == labels).sum().item()
                total_correct += correct
                total_pixels += labels.numel()

                pbar.set_postfix({
                    'loss': total_loss / total_samples,
                    'oa': total_correct / total_pixels
                })

        epoch_loss = total_loss / total_samples
        epoch_oa = total_correct / total_pixels

        return {'loss': epoch_loss, 'oa': epoch_oa}

    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        class_preds = {cls: [] for cls in self.classes}
        class_labels = {cls: [] for cls in self.classes}

        with torch.no_grad(), tqdm(self.dataloader_val, desc=f'Epoch {epoch} [Validation]') as pbar:
            for images_a, images_b, labels in pbar:
                images_a = images_a.to(self.device)
                images_b = images_b.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images_a, images_b)
                losses = self.criterion(outputs, labels)
                loss = losses['total_loss']

                total_loss += loss.item() * images_a.size(0)
                total_samples += images_a.size(0)

                # 获取预测输出
                if isinstance(outputs, dict):
                    predictions = outputs['prediction']  # shape: [B, N, H, W]
                else:
                    predictions = outputs  # 如果没有 MOE，则直接是张量

                for idx, cls in enumerate(self.classes):
                    pred_channel = predictions[:, idx, :, :]  # 提取当前类别的预测通道
                    pred = (torch.sigmoid(pred_channel) > self.cfg.training.threshold).float().cpu().numpy().flatten()
                    label = labels[:, idx, :, :].cpu().numpy().flatten()  # 提取当前类别的标签

                    class_preds[cls].extend(pred)
                    class_labels[cls].extend(label)

                pbar.set_postfix({'loss': total_loss / total_samples})

        metrics = {
            'loss': total_loss / total_samples,
            'precision': {},
            'recall': {},
            'f1': {},
            'iou': {},
            'oa': {}
        }

        for cls in self.classes:
            y_true = np.array(class_labels[cls])
            y_pred = np.array(class_preds[cls])

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            iou = jaccard_score(y_true, y_pred, zero_division=0)
            accuracy = np.mean(y_true == y_pred)

            metrics['precision'][cls] = precision
            metrics['recall'][cls] = recall
            metrics['f1'][cls] = f1
            metrics['iou'][cls] = iou
            metrics['oa'][cls] = accuracy

        return metrics
