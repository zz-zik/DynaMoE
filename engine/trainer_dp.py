# -*- coding: utf-8 -*-
"""
@Project : DynaMoE
@FileName: trainer_dp.py
@Time    : 2025/6/20 下午5:50
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : 建筑物变化检测分布式训练引擎
@Usage   : 
"""
import pandas as pd
import numpy as np
import datetime
import random
import time
import warnings
from torch.nn.parallel import DistributedDataParallel as DDP

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score
from tensorboardX import SummaryWriter
from tqdm import tqdm
from dataloader import build_dataset
from models import build_model
from utils import *

warnings.filterwarnings('ignore')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class Trainer:
    """建筑物变化检测训练引擎 - 支持分布式训练"""

    def __init__(self, cfg):
        """
        初始化训练引擎

        Args:
            cfg: 配置对象
        """
        self.cfg = cfg

        self.device_info = parse_device(cfg.device)

        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.is_main_process = self.local_rank == 0

        if self.device_info['distributed']:
            # 在分布式模式下，使用LOCAL_RANK对应的实际设备ID
            actual_device_id = self.device_info['device_ids'][self.local_rank]
            self.device = torch.device(f"cuda:{actual_device_id}")
        else:
            self.device = self.device_info['main_device']

        self.distributed = self.device_info['distributed']

        self.classes = cfg.data.classes
        self.new_classes = getattr(cfg.model, 'new_classes', None) or []
        self.weights = cfg.model.weights
        self.threshold = cfg.test.threshold

        # 初始化输出目录和日志（仅主进程）
        if self.is_main_process:
            self.output_dir = get_output_dir(cfg.output_dir, cfg.name)
            self.logger = setup_logging(cfg, self.output_dir)
        else:
            self.logger = None

        # 设置随机种子
        self._setup_seed()

        # 初始化模型和相关组件
        self._setup_model()
        self._setup_optimizer()
        self._setup_dataloader()

        # 初始化训练状态
        self._setup_training_state()

        # 初始化记录工具
        if self.is_main_process:
            self._setup_logging_tools()

        # 恢复训练状态
        self._resume_if_needed()

    def _log_info(self, message, *args):
        """安全的日志记录"""
        if self.logger is not None:
            self.logger.info(message, *args)

    def _setup_seed(self):
        """设置随机种子"""
        seed = self.cfg.seed + self.local_rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _setup_model(self):
        """初始化模型和损失函数"""
        if self.is_main_process:
            self._log_info('------------------------ model params ------------------------')

        self.model, self.criterion = build_model(self.cfg, training=True)

        self.model.to(self.device)
        if isinstance(self.criterion, tuple) and len(self.criterion) == 2:
            for loss in self.criterion:
                loss.to(self.device)
        else:
            self.criterion.to(self.device)

        # 增量式训练：如果new_classes不为空，加载预训练权重
        if self.new_classes:
            if self.is_main_process:
                self._log_info('------------------------ Incremental Learning Mode ------------------------')
                self._log_info(f'New classes: {self.new_classes}')
                self._log_info(f'Loading pretrained weights from: {self.weights}')

            if self.weights and os.path.exists(self.weights):
                # 加载预训练权重
                if self.is_main_process:
                    self._log_info(f'Loading model weights from {self.weights}')

                # 根据是否分布式确定map_location
                if self.distributed:
                    map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
                else:
                    map_location = self.device

                try:
                    checkpoint = torch.load(self.weights, map_location=map_location)

                    # 提取模型权重
                    if 'model' in checkpoint:
                        pretrained_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        pretrained_dict = checkpoint['state_dict']
                    else:
                        pretrained_dict = checkpoint

                    # 加载权重（允许部分匹配）
                    model_dict = self.model.state_dict()

                    # 过滤掉不匹配的权重
                    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                       if k in model_dict and v.shape == model_dict[k].shape}

                    # 更新模型权重
                    model_dict.update(pretrained_dict)
                    self.model.load_state_dict(model_dict, strict=False)

                    if self.is_main_process:
                        self._log_info(f'Successfully loaded {len(pretrained_dict)} pretrained parameters')
                        self._log_info('Model is ready for incremental training with frozen MoE gates')

                except Exception as e:
                    if self.is_main_process:
                        self._log_info(f'Warning: Failed to load pretrained weights: {str(e)}')
                        self._log_info('Continuing with random initialization')
            else:
                if self.is_main_process:
                    if not self.weights:
                        self._log_info('Warning: No pretrained weights path provided for incremental learning')
                    else:
                        self._log_info(f'Warning: Pretrained weights file not found: {self.weights}')
                    self._log_info('Continuing with random initialization')

        # 设置分布式训练
        if self.distributed:
            actual_device_id = self.device_info['device_ids'][self.local_rank]
            self.model = DDP(
                self.model,
                device_ids=[actual_device_id],
                output_device=actual_device_id,
                find_unused_parameters=True
            )
            self.model_without_ddp = self.model.module
        else:
            self.model_without_ddp = self.model

        if self.is_main_process:
            n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self._log_info('number of params: %d', n_parameters)

            # 如果是增量式训练，打印冻结参数信息
            if self.new_classes:
                frozen_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                self._log_info(f'Frozen parameters: {frozen_params}')
                self._log_info(f'Trainable parameters: {trainable_params}')
                self._log_info(f'Total parameters: {frozen_params + trainable_params}')

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
        if self.is_main_process:
            self._log_optimizer_info()

    def _log_optimizer_info(self):
        """记录优化器信息"""
        optimizer_info = f"optimizer: Adam(lr={self.cfg.training.lr})"
        optimizer_info += " with parameter groups "
        for i, param_group in enumerate(self.optimizer.param_groups):
            optimizer_info += f"{len(param_group['params'])} weight(decay={param_group['weight_decay']}), "
        optimizer_info = optimizer_info.rstrip(', ')
        self._log_info(optimizer_info)

    def _setup_dataloader(self):
        """设置数据加载器"""
        self.dataloader_train, self.dataloader_val = build_dataset(
            cfg=self.cfg,
            distributed=self.distributed,
            world_size=self.world_size,
            rank=self.local_rank
        )

    def _setup_training_state(self):
        """初始化训练状态"""
        self.start_epoch = self.cfg.training.start_epoch
        self.step = 0

        # 保存训练期间的指标
        self.precision_list = []
        self.recall_list = []
        self.f1_avg_list = []
        self.iou_avg_list = []
        self.oa_avg_list = []

    def _setup_logging_tools(self):
        """设置日志记录工具（仅主进程）"""
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
            if self.is_main_process:
                self._log_info('------------------------ Continue training ------------------------')
                logging.warning(f"loading from {self.cfg.resume}")

            # 所有进程都需要加载模型
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
            checkpoint = torch.load(self.cfg.resume, map_location=map_location)
            self.model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

            if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.start_epoch = checkpoint['epoch'] + 1

    def _log_training_info(self):
        """记录训练开始信息"""
        if not self.is_main_process:
            return

        self._log_info('Train Log %s' % time.strftime("%c"))
        env_info = get_environment_info()
        self._log_info(env_info)
        self._log_info('Running with config:')
        self._log_info(pprint.pformat(self.cfg.__dict__))

        # 记录分布式训练信息
        if self.distributed:
            self._log_info(f'Distributed training enabled: {self.world_size} GPUs')
            self._log_info(f'Device IDs: {self.device_info["device_ids"]}')

    def _train_one_epoch(self, epoch):
        """训练一个epoch"""
        t1 = time.time()

        # 设置epoch用于分布式采样器
        if self.distributed and hasattr(self.dataloader_train.sampler, 'set_epoch'):
            self.dataloader_train.sampler.set_epoch(epoch)

        stat = self.train(epoch)

        # 同步所有进程
        if self.distributed:
            dist.barrier()

        time.sleep(1)  # 避免tensorboard卡顿
        t2 = time.time()

        # 记录训练损失和OA（仅主进程）
        if self.is_main_process:
            if self.writer is not None:
                self._log_info("[ep %d][lr %.7f][%.2fs] loss: %.4f, oa: %.4f",
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
        """保存检查点（仅主进程）"""
        if not self.is_main_process:
            return

        checkpoint_data = {
            'epoch': epoch,
            'step': self.step,
            'model': self.model_without_ddp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
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

        # 同步所有进程的评估结果
        if self.distributed:
            # 收集所有进程的指标并在主进程上计算平均值
            metrics = self._sync_metrics(metrics)

        if not self.is_main_process:
            return metrics

        # 更新指标列表
        avg_metrics = get_avg_metrics(metrics)
        self.precision_list.append(avg_metrics['precision'])
        self.recall_list.append(avg_metrics['recall'])
        self.f1_avg_list.append(avg_metrics['f1'])
        self.iou_avg_list.append(avg_metrics['iou'])
        self.oa_avg_list.append(avg_metrics['oa'])

        fps = len(self.dataloader_val.dataset) / (t2 - t1)

        # 构造日志字符串
        metric_str = ", ".join(
            [f"{cls}: F1={metrics['f1'][cls]:.4f}, IoU={metrics['iou'][cls]:.4f}, OA={metrics['oa'][cls]:.4f}" for cls
             in self.classes])
        self._log_info(
            "[ep %d][%.3fs][%.5ffps] loss: %.4f | %s ---- @best f1: %s" %
            (
                epoch, t2 - t1, fps, metrics['loss'],
                metric_str,
                ", ".join([
                              f"avg_f1: {np.max(self.f1_avg_list):.4f}, avg_IoU: {np.max(self.iou_avg_list):.4f}, avg_OA: {np.max(self.oa_avg_list):.4f}"])
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

        # 更新验证指标
        self.results_df.loc[epoch, [
            'val_loss', 'val_precision', 'val_recall', 'val_f1', 'val_iou', 'val_oa'
        ]] = [avg_metrics['loss'], avg_metrics['precision'], avg_metrics['recall'],
              avg_metrics['f1'], avg_metrics['iou'], avg_metrics['oa']]
        return avg_metrics

    def _sync_metrics(self, metrics):
        """
        同步分布式训练中的评估指标
        """
        if not self.distributed:
            return metrics

        device = self.device

        loss_tensor = torch.tensor(metrics['loss']).to(device)
        dist.barrier()
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / dist.get_world_size()

        synced_metrics = {
            'loss': avg_loss,
            'precision': {},
            'recall': {},
            'f1': {},
            'iou': {},
            'oa': {}
        }

        # 对每个类别分别同步指标
        for cls in self.classes:
            for metric_name in ['precision', 'recall', 'f1', 'iou', 'oa']:
                val = metrics[metric_name].get(cls, 0.0)
                tensor = torch.tensor(val).to(device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                synced_metrics[metric_name][cls] = tensor.item() / dist.get_world_size()

        return synced_metrics

    def _save_best_models(self, epoch, stat, metrics):
        """保存最佳模型（仅主进程）"""
        if not self.is_main_process:
            return

        # 保存最好的F1模型
        if metrics['f1'] == np.max(self.f1_avg_list):
            self._save_checkpoint(epoch, stat, 'best_f1')

        # 保存最好的IoU模型
        if metrics['iou'] == np.max(self.iou_avg_list):
            self._save_checkpoint(epoch, stat, 'best_iou')

    def _save_results_csv(self):
        """保存结果到CSV文件（仅主进程）"""
        if self.is_main_process:
            self.results_df.to_csv(self.csv_file_path, index=False)

    def _log_final_summary(self, total_time):
        """记录最终总结（仅主进程）"""
        if not self.is_main_process:
            return

        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        env_info = get_environment_info()

        self._log_info('Summary of results')
        self._log_info(env_info)
        self._log_info('Training time {}'.format(total_time_str))

        # 打印最好的指标
        self._log_info("Best F1: %.4f, Best IoU: %.4f, Best Accuracy: %.4f" % (
            np.max(self.f1_avg_list), np.max(self.iou_avg_list), np.max(self.oa_avg_list)
        ))
        self._log_info('Results saved to {}'.format(self.cfg.output_dir))

    def run(self):
        """执行训练过程"""
        # 记录训练开始信息
        self._log_training_info()

        if self.is_main_process:
            self._log_info("------------------------ Start training ------------------------")
        start_time = time.time()

        # 训练循环
        for epoch in range(self.start_epoch, self.cfg.training.epochs):
            if hasattr(self.criterion, 'set_current_epoch'):
                self.criterion.set_current_epoch(epoch)  # 更新当前训练轮次

            # 训练一个epoch
            train_stat = self._train_one_epoch(epoch)

            # 调整学习率
            self._adjust_learning_rate()

            # 保存最新检查点（仅主进程）
            self._save_checkpoint(epoch, train_stat, 'latest')

            # 评估
            if epoch % self.cfg.training.eval_freq == 0 and epoch >= self.cfg.training.start_eval:
                eval_metrics = self._evaluate_one_epoch(epoch)

                # 根据验证损失调整学习率（如果使用plateau调度器）
                if self.is_main_process:
                    self._adjust_learning_rate(eval_metrics)

                # 保存最佳模型（仅主进程）
                self._save_best_models(epoch, train_stat, eval_metrics)

            # 保存结果到CSV（仅主进程）
            self._save_results_csv()

        # 训练完成后的总结
        total_time = time.time() - start_time
        self._log_final_summary(total_time)

        # 清理分布式训练
        if self.distributed:
            dist.destroy_process_group()

    def train(self, epoch):
        self.model.train()
        if hasattr(self.criterion, 'train'):
            self.criterion.train()
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        total_pixels = 0

        # 只在主进程显示进度条
        if self.is_main_process:
            pbar = tqdm(self.dataloader_train, desc=f'Epoch {epoch} [Training]')
        else:
            pbar = self.dataloader_train

        for images_a, images_b, labels in pbar:
            images_a = images_a.to(self.device)
            images_b = images_b.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images_a, images_b)

            # 根据 classes 和 new_classes 的关系调整输出维度
            if self.new_classes and len(self.new_classes) < len(self.classes):
                indices = [i for i, cls in enumerate(self.classes) if cls in self.new_classes]
                outputs['prediction'] = outputs['prediction'][:, indices]  # [B, N_new, H, W]
                if 'gates' in outputs:
                    outputs['gates'] = outputs['gates'][:, indices]  # [B, N_new, H, W]

                # 同步裁剪 labels
                if len(self.new_classes) < len(self.classes):
                    labels = labels[:, indices]  # [B, N_new, H, W]

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

            if self.is_main_process and hasattr(pbar, 'set_postfix'):
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

        # 只在主进程显示进度条
        if self.is_main_process:
            pbar = tqdm(self.dataloader_val, desc=f'Epoch {epoch} [Validation]')
        else:
            pbar = self.dataloader_val

        with torch.no_grad():
            for images_a, images_b, labels in pbar:
                images_a = images_a.to(self.device)
                images_b = images_b.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images_a, images_b)

                # 根据 classes 和 new_classes 的关系调整输出维度
                if self.new_classes and len(self.new_classes) < len(self.classes):
                    indices = [i for i, cls in enumerate(self.classes) if cls in self.new_classes]
                    outputs['prediction'] = outputs['prediction'][:, indices]  # [B, N_new, H, W]
                    if 'gates' in outputs:
                        outputs['gates'] = outputs['gates'][:, indices]  # [B, N_new, H, W]

                    # 同步裁剪 labels
                    if len(self.new_classes) == len(self.classes):
                        labels = labels[:, indices]  # [B, N_new, H, W]

                losses = self.criterion(outputs, labels)
                loss = losses['total_loss']

                total_loss += loss.item() * images_a.size(0)
                total_samples += images_a.size(0)

                # 获取预测输出
                if isinstance(outputs, dict):
                    predictions = outputs['prediction']  # shape: [B, N, H, W]
                else:
                    predictions = outputs

                for idx, cls in enumerate(self.classes):
                    pred_channel = predictions[:, idx, :, :].squeeze(1)  # 正确提取第idx个类别的预测
                    label_channel = labels[:, idx, :, :].squeeze(1)  # 确保这是对应类别的真实标签

                    pred = (torch.sigmoid(pred_channel) > self.cfg.training.threshold).cpu().numpy()
                    true = label_channel.cpu().numpy()

                    class_preds[cls].extend(pred.flatten())
                    class_labels[cls].extend(true.flatten())

                if self.is_main_process and hasattr(pbar, 'set_postfix'):
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
            accuracy = accuracy_score(y_true, y_pred)

            metrics['precision'][cls] = precision
            metrics['recall'][cls] = recall
            metrics['f1'][cls] = f1
            metrics['iou'][cls] = iou
            metrics['oa'][cls] = accuracy

        return metrics


def get_avg_metrics(metrics: dict):
    """对输入metric的每一个指标单独计算所有类别的平均值"""
    metric = {
        'loss': metrics['loss'],
        'precision': np.mean(list(metrics['precision'].values())),
        'recall': np.mean(list(metrics['recall'].values())),
        'f1': np.mean(list(metrics['f1'].values())),
        'iou': np.mean(list(metrics['iou'].values())),
        'oa': np.mean(list(metrics['oa'].values()))
    }

    return metric
