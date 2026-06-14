from collections import deque
import scipy.io
import os
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from matplotlib import pyplot as plt, ticker, gridspec
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange
import re
from matplotlib.gridspec import GridSpec
import random
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path
from datetime import datetime
from datetime import datetime
import numpy as np
import torch

from datetime import datetime
import numpy as np
import torch
import time
from collections import deque
from typing import Optional, Dict, List


class QuickMonitor:
    """
    快速监控器 - 最小侵入式

    使用方法:
    ```python
    monitor = QuickMonitor(enable_timing=True, enable_health=True)

    for epoch in range(epochs):
        for batch_idx, data in enumerate(loader):
            monitor.tick('data_load')

            # ... 前向传播 ...
            monitor.tick('forward')

            # ... 损失计算 ...
            monitor.tick('loss')
            monitor.check_loss(loss.item(), epoch, batch_idx)

            # ... 反向传播 ...
            monitor.tick('backward')
            monitor.check_grad(grad_norm, epoch, batch_idx)

            monitor.tick('batch_end')
            monitor.batch_done()

        monitor.epoch_done(epoch)
    ```
    """

    def __init__(
            self,
            enable_timing: bool = True,
            enable_health: bool = True,
            print_freq: int = 0,  # 0表示只在epoch结束时打印
            loss_spike_threshold: float = 3.0,
            grad_threshold: float = 100.0,
    ):
        self.enable_timing = enable_timing
        self.enable_health = enable_health
        self.print_freq = print_freq
        self.loss_spike_threshold = loss_spike_threshold
        self.grad_threshold = grad_threshold

        self.reset()

    def reset(self):
        # 计时
        self._last_tick = None
        self._tick_name = None
        self._intervals: Dict[str, List[float]] = {}

        # 健康监控
        self._loss_history = deque(maxlen=100)
        self._grad_history = deque(maxlen=100)
        self._delta_history = deque(maxlen=100)
        self._warnings: List[str] = []
        self._batch_count = 0

    def tick(self, name: str):
        """标记时间点，自动计算与上一个tick的间隔"""
        if not self.enable_timing:
            return

        now = time.time()
        if self._last_tick is not None and self._tick_name is not None:
            interval = now - self._last_tick
            key = f"{self._tick_name}→{name}"
            if key not in self._intervals:
                self._intervals[key] = []
            self._intervals[key].append(interval)

        self._last_tick = now
        self._tick_name = name

    def check_loss(self, loss: float, epoch: int, batch: int) -> bool:
        """检查loss，返回是否健康"""
        if not self.enable_health:
            return True

        if not np.isfinite(loss):
            self._warnings.append(f"❌ [E{epoch}B{batch}] NaN/Inf loss!")
            return False

        self._loss_history.append(loss)

        if len(self._loss_history) >= 10:
            recent_mean = np.mean(list(self._loss_history)[-10:-1])
            if loss > recent_mean * self.loss_spike_threshold:
                self._warnings.append(
                    f"⚠️ [E{epoch}B{batch}] Loss spike: {loss:.4e} > {self.loss_spike_threshold}x mean({recent_mean:.4e})"
                )

        # 检测连续上升
        if len(self._loss_history) >= 5:
            recent = list(self._loss_history)[-5:]
            if all(recent[i] < recent[i + 1] for i in range(4)):
                self._warnings.append(f"⚠️ [E{epoch}B{batch}] Loss rising for 5 consecutive batches")

        return True

    def check_grad(self, grad_norm: float, epoch: int, batch: int) -> bool:
        """检查梯度范数"""
        if not self.enable_health:
            return True

        if not np.isfinite(grad_norm):
            self._warnings.append(f"❌ [E{epoch}B{batch}] NaN/Inf gradient!")
            return False

        self._grad_history.append(grad_norm)

        if grad_norm > self.grad_threshold:
            self._warnings.append(f"⚠️ [E{epoch}B{batch}] Large gradient: {grad_norm:.2f}")

        # 梯度突增检测
        if len(self._grad_history) >= 5:
            recent_mean = np.mean(list(self._grad_history)[-5:-1])
            if recent_mean > 0 and grad_norm > recent_mean * 10:
                self._warnings.append(
                    f"⚠️ [E{epoch}B{batch}] Gradient spike: {grad_norm:.2f} > 10x mean({recent_mean:.2f})"
                )

        return True

    def check_delta(self, delta: torch.Tensor, epoch: int, batch: int, threshold: float = 5.0) -> Dict:
        """检查delta统计量"""
        if not self.enable_health:
            return {}

        with torch.no_grad():
            stats = {
                'mean': delta.mean().item(),
                'std': delta.std().item(),
                'max': delta.abs().max().item(),
            }

        self._delta_history.append(stats['max'])

        if len(self._delta_history) >= 10:
            recent_mean = np.mean(list(self._delta_history)[-10:-1])
            if recent_mean > 0 and stats['max'] > recent_mean * threshold:
                self._warnings.append(
                    f"⚠️ [E{epoch}B{batch}] Delta spike: max={stats['max']:.4e} > {threshold}x mean({recent_mean:.4e})"
                )

        return stats

    def batch_done(self):
        """batch结束时调用"""
        self._batch_count += 1

        if self.print_freq > 0 and self._batch_count % self.print_freq == 0:
            self._print_status()

    def epoch_done(self, epoch: int):
        """epoch结束时调用"""
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch} 监控报告")
        print('=' * 60)

        # 打印耗时
        if self.enable_timing and self._intervals:
            print("\n⏱️ 耗时分析:")
            total = sum(sum(v) for v in self._intervals.values())
            sorted_items = sorted(
                self._intervals.items(),
                key=lambda x: sum(x[1]),
                reverse=True
            )
            for name, times in sorted_items:
                avg = np.mean(times) * 1000
                pct = sum(times) / total * 100 if total > 0 else 0
                print(f"  {name:25s}: {avg:8.2f}ms avg ({pct:5.1f}%)")
            print(f"  {'Total':25s}: {total:.2f}s")

        # 打印健康状态
        if self.enable_health:
            print("\n🏥 健康状态:")
            if self._loss_history:
                recent_loss = list(self._loss_history)[-10:]
                print(f"  Loss: {recent_loss[-1]:.4e} (recent mean: {np.mean(recent_loss):.4e})")
            if self._grad_history:
                recent_grad = list(self._grad_history)[-10:]
                print(f"  Grad norm: {recent_grad[-1]:.2f} (recent mean: {np.mean(recent_grad):.2f})")

            # 打印警告
            if self._warnings:
                print(f"\n⚠️ 警告 ({len(self._warnings)} 条):")
                for w in self._warnings[-10:]:  # 只显示最近10条
                    print(f"  {w}")
                if len(self._warnings) > 10:
                    print(f"  ... 还有 {len(self._warnings) - 10} 条警告")
            else:
                print("\n✅ 无警告")

        print('=' * 60 + "\n")

        # 重置batch计数和间隔记录
        self._intervals.clear()
        self._warnings.clear()
        self._batch_count = 0

    def _print_status(self):
        """打印当前状态"""
        status = []
        if self._loss_history:
            status.append(f"loss={self._loss_history[-1]:.4e}")
        if self._grad_history:
            status.append(f"grad={self._grad_history[-1]:.2f}")
        print(f"  [Batch {self._batch_count}] {', '.join(status)}")

    def get_early_warning_signs(self) -> List[str]:
        """
        获取早期预警信号 - 在爆炸前识别问题
        """
        signs = []

        # 检查loss趋势
        if len(self._loss_history) >= 20:
            first_half = list(self._loss_history)[:10]
            second_half = list(self._loss_history)[-10:]
            if np.mean(second_half) > np.mean(first_half) * 1.5:
                signs.append("Loss整体趋势上升")

        # 检查梯度趋势
        if len(self._grad_history) >= 20:
            first_half = list(self._grad_history)[:10]
            second_half = list(self._grad_history)[-10:]
            if np.mean(second_half) > np.mean(first_half) * 2:
                signs.append("梯度范数整体趋势上升")

        # 检查delta趋势
        if len(self._delta_history) >= 20:
            first_half = list(self._delta_history)[:10]
            second_half = list(self._delta_history)[-10:]
            if np.mean(second_half) > np.mean(first_half) * 2:
                signs.append("Delta max整体趋势上升")

        return signs


class TrainingCallback:
    """
    训练回调：检测 loss 突变并回滚（内存优化版）
    """

    def __init__(self, model, optimizer, save_dir,
                 loss_spike_threshold=2.0,
                 spike_count_to_reduce_lr=3,
                 lr_reduce_factor=0.5,
                 history_size=100,
                 min_history_for_detection=10,
                 checkpoint_interval=5):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.loss_spike_threshold = loss_spike_threshold
        self.spike_count_to_reduce_lr = spike_count_to_reduce_lr
        self.lr_reduce_factor = lr_reduce_factor
        self.history_size = history_size
        self.min_history_for_detection = min_history_for_detection
        self.checkpoint_interval = checkpoint_interval

        # 状态
        self.best_loss = float('inf')
        self.best_model_state = None
        self.loss_history = []
        self.spike_count = 0
        self.total_spikes = 0
        self.total_rollbacks = 0
        self.lr_reductions = 0

        # 只保存一个最近的检查点（内存优化）
        self.recent_checkpoint = None
        self.batch_counter = 0

    def on_batch_start(self, epoch, batch):
        """每个 batch 开始时调用，定期保存检查点"""
        self.batch_counter += 1

        # 每 N 个 batch 保存一次检查点（覆盖之前的）
        if self.batch_counter % self.checkpoint_interval == 0:
            # 先清理旧的
            if self.recent_checkpoint is not None:
                del self.recent_checkpoint
                torch.cuda.empty_cache()

            self.recent_checkpoint = {
                'epoch': epoch,
                'batch': batch,
                'model': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                # 只保存 lr，不保存完整优化器状态（省内存）
                'lr': self.optimizer.param_groups[0]['lr'],
            }

    def on_batch_end(self, epoch, batch, loss):
        """每个 batch 结束时调用，返回 True 表示需要跳过当前 batch"""
        current_loss = loss.item() if torch.is_tensor(loss) else loss

        # 检查 NaN
        if not np.isfinite(current_loss):
            self._log(f"[NaN Loss] epoch {epoch}, batch {batch}")
            return self._handle_spike(epoch, batch, current_loss, float('nan'), reason="nan")

        # 检测 loss 突变
        if len(self.loss_history) >= self.min_history_for_detection:
            avg_recent = np.mean(self.loss_history[-self.min_history_for_detection:])
            if current_loss > avg_recent * self.loss_spike_threshold:
                return self._handle_spike(epoch, batch, current_loss, avg_recent, reason="spike")

        # 正常：记录 loss
        self.loss_history.append(current_loss)
        if len(self.loss_history) > self.history_size:
            self.loss_history.pop(0)
        self.spike_count = 0

        return False

    def on_epoch_end(self, epoch, avg_loss):
        """每个 epoch 结束时调用"""
        if not np.isfinite(avg_loss):
            return False

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss

            # 清理旧的最佳模型
            if self.best_model_state is not None:
                del self.best_model_state
                torch.cuda.empty_cache()

            self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            self._log(f"[Best Checkpoint] epoch {epoch}, loss: {avg_loss:.4e}")
            return True
        return False

    def _handle_spike(self, epoch, batch, current_loss, avg_loss, reason):
        """处理 loss 突变"""
        self.spike_count += 1
        self.total_spikes += 1

        self._log(f"[{reason.upper()}] epoch {epoch}, batch {batch}: "
                  f"current={current_loss:.4e}, avg={avg_loss:.4e}, "
                  f"spike_count={self.spike_count}/{self.spike_count_to_reduce_lr}")

        # 回滚
        self._rollback()

        # 连续 N 次突变后降低 lr
        if self.spike_count >= self.spike_count_to_reduce_lr:
            self._reduce_lr(epoch, batch)
            self.spike_count = 0

        return True

    def _rollback(self):
        """回滚到最近的检查点"""
        # ===== 关键：先清理 GPU 缓存 =====
        torch.cuda.empty_cache()

        if self.recent_checkpoint is not None:
            self.model.load_state_dict(self.recent_checkpoint['model'])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.recent_checkpoint['lr']
            self.total_rollbacks += 1
            self._log(f"[Rollback] Restored to epoch {self.recent_checkpoint['epoch']}, "
                      f"batch {self.recent_checkpoint['batch']} (total: {self.total_rollbacks})")

            if len(self.loss_history) > self.checkpoint_interval:
                self.loss_history = self.loss_history[:-self.checkpoint_interval]

            # ===== 再清理一次 =====
            torch.cuda.empty_cache()
            return True

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.total_rollbacks += 1
            self._log(f"[Rollback] Restored to best model (total: {self.total_rollbacks})")
            torch.cuda.empty_cache()
            return True

        self._log("[Rollback] No checkpoint available!")
        return False

    def _reduce_lr(self, epoch, batch):
        """降低学习率"""
        old_lr = self.optimizer.param_groups[0]['lr']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.lr_reduce_factor
        new_lr = self.optimizer.param_groups[0]['lr']
        self.lr_reductions += 1
        self._log(f"[LR Reduce] {old_lr:.2e} -> {new_lr:.2e} (total: {self.lr_reductions})")

    def _log(self, message):
        """写入日志"""
        print(message)
        with open(f'{self.save_dir}/Experiment_record.txt', 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"│  {message} | {timestamp}\n")

    def summary(self):
        """打印训练总结"""
        summary = f"""
╔══════════════════════════════════════╗
║       Training Callback Summary       ║
╠══════════════════════════════════════╣
║  Best Loss:       {self.best_loss:.4e}         ║
║  Total Spikes:    {self.total_spikes:<20} ║
║  Total Rollbacks: {self.total_rollbacks:<20} ║
║  LR Reductions:   {self.lr_reductions:<20} ║
║  Final LR:        {self.optimizer.param_groups[0]['lr']:.2e}         ║
╚══════════════════════════════════════╝
"""
        print(summary)
        with open(f'{self.save_dir}/Experiment_record.txt', 'a', encoding='utf-8') as f:
            f.write(summary)

    def cleanup(self):
        """清理内存"""
        if self.recent_checkpoint is not None:
            del self.recent_checkpoint
            self.recent_checkpoint = None
        if self.best_model_state is not None:
            del self.best_model_state
            self.best_model_state = None
        torch.cuda.empty_cache()


class ResidualAdaptiveWeightScheduler:
    """
    残差自适应权重调度器
    结合位置权重和残差自适应，防止模型放弃高残差点
    """

    def __init__(
            self,
            position_decay: float = 0.9,
            res_clamp_min: float = 0.5,
            res_clamp_max: float = 3.0,
            use_position_weights: bool = True,
            use_res_adaptive: bool = True,
            warmup_no_weight_epochs: int = 0,  # 阶段1：完全不加权
            warmup_position_only_epochs: int = 0,  # 阶段2：只用位置权重
            decay_position_weight: bool = False,
            final_position_decay: float = 1.0,
            total_epochs: int = 100,
            log_dir: Optional[str] = './ResidualAdaptiveWeightScheduler',
    ):
        """
        Args:
            position_decay: 位置权重衰减系数，越小前面权重越高
            res_clamp_min: 残差自适应权重下限
            res_clamp_max: 残差自适应权重上限
            use_position_weights: 是否使用位置权重
            use_res_adaptive: 是否使用残差自适应权重
            warmup_epochs: 预热期，期间只用位置权重
            decay_position_weight: 是否在训练过程中衰减位置权重的影响
            final_position_decay: 训练结束时的position_decay值
            total_epochs: 总训练轮数
            log_dir: 日志保存目录
        """
        self.position_decay = position_decay
        self.initial_position_decay = position_decay
        self.final_position_decay = final_position_decay
        self.res_clamp_min = res_clamp_min
        self.res_clamp_max = res_clamp_max
        self.use_position_weights = use_position_weights
        self.use_res_adaptive = use_res_adaptive
        self.warmup_no_weight_epochs = warmup_no_weight_epochs
        self.warmup_position_only_epochs = warmup_position_only_epochs
        self.decay_position_weight = decay_position_weight
        self.total_epochs = total_epochs

        self.current_epoch = 0

        # 缓存
        self._position_weight_cache = {}

        # 历史记录
        self.history = {
            'epoch': [],
            'position_decay': [],
            'residual_per_t': [],
            'weights': [],
            'loss': [],
        }

        # 日志目录
        if log_dir is not None:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.log_dir = None

    def step(self, epoch: int = None):
        """更新epoch，调整调度参数"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        # 动态调整position_decay
        if self.decay_position_weight and self.current_epoch > self.warmup_epochs:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(1.0, max(0.0, progress))
            self.position_decay = (
                    self.initial_position_decay +
                    progress * (self.final_position_decay - self.initial_position_decay)
            )
            # 清空缓存
            self._position_weight_cache = {}

    def should_record(self, epoch, checkpoint_interval):
        """判断当前epoch是否需要记录"""
        return epoch % checkpoint_interval == 0

    def get_position_weights(self, n_t: int, device='cpu'):
        """获取位置权重（带缓存）"""
        cache_key = (n_t, self.position_decay)
        if cache_key not in self._position_weight_cache:
            indices = torch.arange(n_t, dtype=torch.float32)
            weights = torch.pow(torch.tensor(self.position_decay), indices)
            self._position_weight_cache[cache_key] = weights
        return self._position_weight_cache[cache_key].to(device)

    def compute_res_adaptive_weights(self, residual_per_t: torch.Tensor):
        """
        计算残差自适应权重
        残差大的点权重更高，防止被放弃
        """
        res_mean = residual_per_t.mean()
        if res_mean > 1e-8:
            weights = residual_per_t / res_mean
        else:
            weights = torch.ones_like(residual_per_t)

        weights = torch.clamp(weights, min=self.res_clamp_min, max=self.res_clamp_max)
        return weights.detach()

    def compute_weights(self, residual_per_t: torch.Tensor):
        n_t = residual_per_t.shape[0]
        device = residual_per_t.device

        # 阶段1：完全不加权
        if self.current_epoch < self.warmup_no_weight_epochs:
            return torch.ones(n_t, device=device)

        # 阶段2的起点
        stage2_start = self.warmup_no_weight_epochs
        stage3_start = self.warmup_no_weight_epochs + self.warmup_position_only_epochs

        weights = torch.ones(n_t, device=device)

        # 阶段2及之后：位置权重
        if self.use_position_weights and self.current_epoch >= stage2_start:
            position_weights = self.get_position_weights(n_t, device)
            weights = weights * position_weights

        # 阶段3：残差自适应
        if self.use_res_adaptive and self.current_epoch >= stage3_start:
            res_weights = self.compute_res_adaptive_weights(residual_per_t)
            weights = weights * res_weights

        # 归一化
        weights = weights / weights.mean()
        return weights.detach()

    def compute_weighted_loss(self, residuals: torch.Tensor, reduce='mean', record=False):
        """
        计算带权重的损失
        :param residuals: [bs, nx, nt, 1] 残差
        :param reduce: 'mean' or 'sum'
        :param record: 是否记录到历史
        :return: (加权损失, 每个时间点的MSE, 权重)
        """
        # [bs, nx, nt, 1] -> [bs, nx, nt]
        residuals = residuals.squeeze(-1)

        # 计算每个时间点的MSE: [bs, nx, nt] -> [nt]
        residual_per_t = (residuals ** 2).mean(dim=(0, 1))  # 在batch和空间维度上平均

        # 计算权重: [nt]
        weights = self.compute_weights(residual_per_t)

        # 加权: [nt] -> 广播到 [bs, nx, nt]
        weighted_residual = (residuals ** 2) * weights.view(1, 1, -1)

        if reduce == 'mean':
            loss = weighted_residual.mean()
        else:
            loss = weighted_residual.sum()

        # 记录历史
        if record:
            self.history['epoch'].append(self.current_epoch)
            self.history['position_decay'].append(self.position_decay)
            self.history['residual_per_t'].append(residual_per_t.detach().cpu().numpy())
            self.history['weights'].append(weights.detach().cpu().numpy())
            self.history['loss'].append(loss.item())

        return loss, residual_per_t.detach(), weights

    def plot_weights_vs_residual(
            self,
            residual_per_t: Optional[torch.Tensor] = None,
            weights: Optional[torch.Tensor] = None,
            title: Optional[str] = None,
            save_path: Optional[str] = None,
            show: bool = True,
    ):
        """
        绘制权重和残差对比图
        :param residual_per_t: [n_t] 每个时间点的残差，如果为None则用最近一次记录
        :param weights: [n_t] 权重，如果为None则用最近一次记录
        :param title: 图标题
        :param save_path: 保存路径
        :param show: 是否显示
        """
        # 获取数据
        if residual_per_t is None:
            if len(self.history['residual_per_t']) == 0:
                raise ValueError("No history recorded. Call compute_weighted_loss first.")
            residual_per_t = self.history['residual_per_t'][-1]
        elif isinstance(residual_per_t, torch.Tensor):
            residual_per_t = residual_per_t.detach().cpu().numpy()

        if weights is None:
            if len(self.history['weights']) == 0:
                raise ValueError("No history recorded. Call compute_weighted_loss first.")
            weights = self.history['weights'][-1]
        elif isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()

        n_t = len(residual_per_t)
        t_indices = np.arange(n_t)

        # 创建图
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 左轴：残差 (log scale)
        color1 = 'tab:red'
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Residual (MSE)', color=color1, fontsize=12)
        bars1 = ax1.bar(t_indices - 0.2, residual_per_t, 0.4, label='Residual', color=color1, alpha=0.7)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_yscale('log')

        # 右轴：权重
        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('Weight', color=color2, fontsize=12)
        bars2 = ax2.bar(t_indices + 0.2, weights, 0.4, label='Weight', color=color2, alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Weight=1.0')

        # 标题
        if title is None:
            title = f'Epoch {self.current_epoch}: Residual vs Weight'
        plt.title(title, fontsize=14)

        # 图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # 添加数值标注
        for i, (r, w) in enumerate(zip(residual_per_t, weights)):
            ax1.annotate(f'{r:.2e}', (i - 0.2, r), ha='center', va='bottom', fontsize=8, color=color1)
            ax2.annotate(f'{w:.2f}', (i + 0.2, w), ha='center', va='bottom', fontsize=8, color=color2)

        plt.tight_layout()

        # 保存
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        elif self.log_dir is not None:
            plt.savefig(self.log_dir / f'weights_vs_residual_epoch{self.current_epoch}.png',
                        dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_history(
            self,
            save_path: Optional[str] = None,
            show: bool = True,
    ):
        """
        绘制训练历史
        """
        if len(self.history['epoch']) == 0:
            raise ValueError("No history recorded.")

        # 每个epoch取最后一次记录
        epochs = []
        losses = []
        residuals_first = []  # 第一个时间步的残差
        residuals_mean = []  # 平均残差
        weights_first = []  # 第一个时间步的权重

        last_epoch = -1
        for i, ep in enumerate(self.history['epoch']):
            if ep != last_epoch:
                epochs.append(ep)
                losses.append(self.history['loss'][i])
                residuals_first.append(self.history['residual_per_t'][i][0])
                residuals_mean.append(np.mean(self.history['residual_per_t'][i]))
                weights_first.append(self.history['weights'][i][0])
                last_epoch = ep

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss曲线
        ax = axes[0, 0]
        ax.semilogy(epochs, losses, 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weighted Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)

        # 第一个时间步残差 vs 平均残差
        ax = axes[0, 1]
        ax.semilogy(epochs, residuals_first, 'r-', linewidth=2, label='First step residual')
        ax.semilogy(epochs, residuals_mean, 'g--', linewidth=2, label='Mean residual')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Residual (MSE)')
        ax.set_title('Residual Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 第一个时间步的权重
        ax = axes[1, 0]
        ax.plot(epochs, weights_first, 'm-', linewidth=2)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight')
        ax.set_title('First Step Weight')
        ax.grid(True, alpha=0.3)

        # Position decay
        ax = axes[1, 1]
        ax.plot(epochs, [self.history['position_decay'][self.history['epoch'].index(e)] for e in epochs],
                'c-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Position Decay')
        ax.set_title('Position Decay Schedule')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Residual Adaptive Weight Scheduler History', fontsize=14)
        plt.tight_layout()

        # 保存
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        elif self.log_dir is not None:
            plt.savefig(self.log_dir / 'training_history.png', dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_residual_heatmap(
            self,
            n_epochs: int = 20,
            save_path: Optional[str] = None,
            show: bool = True,
    ):
        """
        绘制残差随时间和epoch的热力图
        """
        if len(self.history['epoch']) == 0:
            raise ValueError("No history recorded.")

        # 收集数据，每个epoch取最后一次
        epoch_residuals = {}
        for i, ep in enumerate(self.history['epoch']):
            epoch_residuals[ep] = self.history['residual_per_t'][i]

        epochs = sorted(epoch_residuals.keys())[-n_epochs:]
        residual_matrix = np.array([epoch_residuals[e] for e in epochs])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 残差热力图
        ax = axes[0]
        im = ax.imshow(np.log10(residual_matrix + 1e-10), aspect='auto', cmap='hot_r')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Epoch')
        ax.set_yticks(range(len(epochs)))
        ax.set_yticklabels(epochs)
        ax.set_title('Log10(Residual) Heatmap')
        plt.colorbar(im, ax=ax, label='log10(MSE)')

        # 权重热力图
        epoch_weights = {}
        for i, ep in enumerate(self.history['epoch']):
            epoch_weights[ep] = self.history['weights'][i]
        weight_matrix = np.array([epoch_weights[e] for e in epochs])

        ax = axes[1]
        im = ax.imshow(weight_matrix, aspect='auto', cmap='Blues')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Epoch')
        ax.set_yticks(range(len(epochs)))
        ax.set_yticklabels(epochs)
        ax.set_title('Weight Heatmap')
        plt.colorbar(im, ax=ax, label='Weight')

        plt.suptitle('Residual and Weight Evolution', fontsize=14)
        plt.tight_layout()

        # 保存
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        elif self.log_dir is not None:
            plt.savefig(self.log_dir / 'residual_heatmap.png', dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def state_dict(self):
        """保存状态"""
        return {
            'current_epoch': self.current_epoch,
            'position_decay': self.position_decay,
            'history': self.history,
        }

    def load_state_dict(self, state_dict):
        """加载状态"""
        self.current_epoch = state_dict['current_epoch']
        self.position_decay = state_dict['position_decay']
        if 'history' in state_dict:
            self.history = state_dict['history']
        self._position_weight_cache = {}

    def get_status(self):
        """获取当前状态信息"""
        return {
            'epoch': self.current_epoch,
            'position_decay': self.position_decay,
            'use_position_weights': self.use_position_weights,
            'use_res_adaptive': self.use_res_adaptive,
        }

    def clear_history(self):
        """清空历史记录（节省内存）"""
        self.history = {
            'epoch': [],
            'position_decay': [],
            'residual_per_t': [],
            'weights': [],
            'loss': [],
        }


def plot_training_curves(loss_list, lr_list, test_loss_list, grad_array, save_path):
    """
    绘制训练曲线

    参数:
        loss_list: list of [loss_data, loss_f, total_loss, epoch]
        lr_list: list of [current_lr, current_t_train, epoch]
        test_loss_list: list of [test_l2, epoch]
        grad_array: list of [epoch, avg_epoch_loss, avg_grad_norm]
        save_path: 保存路径（不含扩展名）
    """
    # 转换为numpy数组
    loss_array = np.array(loss_list)
    test_array = np.array(test_loss_list)
    lr_array = np.array(lr_list)
    grad_array = np.array(grad_array)

    # 提取各个变量
    loss_data = loss_array[:, 0]
    loss_f = loss_array[:, 1]
    loss_epochs = loss_array[:, 3]

    test_l2 = test_array[:, 0]
    test_epochs = test_array[:, 1]

    lr = lr_array[:, 0]
    lr_epochs = lr_array[:, 2]

    grad_epochs = grad_array[:, 0]
    avg_grad_norm = grad_array[:, 2]

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 子图1: loss_f (主) + lr (副)
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    ln1 = ax1.plot(loss_epochs, loss_f, 'b-', alpha=0.7, label='loss_f')
    ln2 = ax1_twin.plot(lr_epochs, lr, 'r-', alpha=0.7, label='lr')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('loss_f', color='b')
    ax1_twin.set_ylabel('lr', color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_yscale('log')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')
    ax1.set_title('PDE Loss & Learning Rate')
    ax1.grid(True, alpha=0.3)

    # 子图2: loss_data (主) + avg_grad_norm (副)
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    ln3 = ax2.plot(loss_epochs, loss_data, 'b-', alpha=0.7, label='loss_data')
    ln4 = ax2_twin.plot(grad_epochs, avg_grad_norm, 'r-', alpha=0.7, label='grad_norm')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('loss_data', color='b')
    ax2_twin.set_ylabel('avg_grad_norm', color='r')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.set_yscale('log')
    ax2_twin.set_yscale('log')

    lns = ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='upper right')
    ax2.set_title('Data Loss & Gradient Norm')
    ax2.grid(True, alpha=0.3)

    # 子图3: test_l2 (主) + lr (副)
    ax3 = axes[2]
    ax3_twin = ax3.twinx()

    ln5 = ax3.plot(test_epochs, test_l2, 'b-', marker='o', markersize=3, alpha=0.7, label='test_l2')
    ln6 = ax3_twin.plot(lr_epochs, lr, 'r-', alpha=0.7, label='lr')

    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('test_l2', color='b')
    ax3_twin.set_ylabel('lr', color='r')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    ax3.set_yscale('log')

    lns = ln5 + ln6
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc='upper right')
    ax3.set_title('Test L2 Error & Learning Rate')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图片已保存为 {save_path}.png")


def plot_visualization_results(visualize_results, save_dir=None):
    """
    绘制可视化结果

    Args:
        visualize_results: 包含 9 个样本的列表
        save_dir: 保存路径（可选）
    """
    # 按类别分组
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    best_samples = [r for r in visualize_results if r['category'] == 'best']
    mid_samples = [r for r in visualize_results if r['category'] == 'mid']
    worst_samples = [r for r in visualize_results if r['category'] == 'worst']

    # ========== 图1-3：pred vs yy 对比（每类别一张图）==========
    for category, samples in [('best', best_samples), ('mid', mid_samples), ('worst', worst_samples)]:
        plot_pred_vs_yy(samples, category, save_dir)

    # ========== 图4：残差对比（9 个样本）==========
    plot_residual_comparison(visualize_results, save_dir)


def plot_pred_vs_yy(samples, category, save_dir=None):
    """
    绘制 pred vs yy 对比图
    5 行 3 列：
    - 行1: 初始条件
    - 行2: pred 时空图
    - 行3: yy 时空图
    - 行4: 绝对误差
    - 行5: 不同时间步切片对比

    网格：x轴为CGL点，t轴均匀分布
    """
    fig = plt.figure(figsize=(15, 20))
    gs = GridSpec(5, 3, figure=fig, hspace=0.3, wspace=0.25)

    for col, sample in enumerate(samples):
        pred = sample['pred']  # [nx, nt]
        yy = sample['yy']  # [nx, nt]
        idx = sample['index']
        l2_error = sample['l2_error']

        nx, nt = pred.shape

        # CGL点 (Chebyshev-Gauss-Lobatto)
        i = np.arange(nx)
        x_cgl = np.cos(np.pi * i / (nx - 1))  # [1, -1]，需要翻转
        x_cgl = x_cgl[::-1]  # [-1, 1]

        # 均匀时间网格
        t_grid = np.linspace(0, 1, nt)

        # 时间步索引
        t_indices = [nt // 3, nt // 2, nt - 1]

        # 统一色标范围
        vmin = min(pred.min(), yy.min())
        vmax = max(pred.max(), yy.max())
        error = np.abs(pred - yy)

        # 行1：初始条件
        ax1 = fig.add_subplot(gs[0, col])
        ax1.plot(x_cgl, yy[:, 0], 'b-', label='IC', linewidth=2)
        ax1.set_title(f'Sample {idx}\nL2 Error: {l2_error:.4e}', fontsize=10)
        ax1.set_xlabel('x')
        ax1.set_ylabel('u')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 创建网格用于pcolormesh（CGL点在x方向非均匀）
        T, X = np.meshgrid(t_grid, x_cgl)

        # 行2：pred 时空图
        ax2 = fig.add_subplot(gs[1, col])
        im2 = ax2.pcolormesh(T, X, pred, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax2.set_title('Prediction', fontsize=10)
        ax2.set_xlabel('t')
        ax2.set_ylabel('x')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # 行3：yy 时空图
        ax3 = fig.add_subplot(gs[2, col])
        im3 = ax3.pcolormesh(T, X, yy, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax3.set_title('Ground Truth', fontsize=10)
        ax3.set_xlabel('t')
        ax3.set_ylabel('x')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # 行4：绝对误差
        ax4 = fig.add_subplot(gs[3, col])
        im4 = ax4.pcolormesh(T, X, error, shading='auto', cmap='hot')
        ax4.set_title(f'Absolute Error\nMax: {error.max():.4e}', fontsize=10)
        ax4.set_xlabel('t')
        ax4.set_ylabel('x')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        # 行5：时间切片对比
        ax5 = fig.add_subplot(gs[4, col])
        colors = ['blue', 'green', 'red']
        labels = [f't={t_indices[0]}/{nt}', f't={t_indices[1]}/{nt}', f't={t_indices[2]}/{nt}']

        for i, t_idx in enumerate(t_indices):
            ax5.plot(x_cgl, yy[:, t_idx], linestyle='--', color=colors[i],
                     label=f'GT {labels[i]}', linewidth=1.5, alpha=0.7)
            ax5.plot(x_cgl, pred[:, t_idx], linestyle='-', color=colors[i],
                     label=f'Pred {labels[i]}', linewidth=1.5)

        ax5.set_xlabel('x')
        ax5.set_ylabel('u')
        ax5.set_title('Time Slices Comparison', fontsize=10)
        ax5.legend(loc='upper right', fontsize=6, ncol=2)
        ax5.grid(True, alpha=0.3)

    fig.suptitle(f'{category.upper()} Samples - Prediction vs Ground Truth',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/{category}_pred_vs_yy.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{save_dir}/{category}_pred_vs_yy.pdf', bbox_inches='tight')

    plt.show()
    plt.close()


def plot_residual_comparison(visualize_results, save_dir=None):
    """绘制残差对比图"""
    sorted_results = (
            [r for r in visualize_results if r.get('category') == 'best'] +
            [r for r in visualize_results if r.get('category') == 'mid'] +
            [r for r in visualize_results if r.get('category') == 'worst']
    )

    n_samples = len(sorted_results)
    if n_samples == 0:
        print("警告：没有可绘制的样本")
        return

    fig = plt.figure(figsize=(15, 3 * n_samples))
    gs = GridSpec(n_samples, 3, figure=fig, hspace=0.35, wspace=0.25)

    for row, sample in enumerate(sorted_results):
        pred = sample['pred']  # [nx, nt]
        yy = sample['yy']  # [nx, nt]
        pred_du = sample['pred_du']  # [nx, nt]
        yy_du = sample['yy_du']  # [nx, nt]
        idx = sample['index']
        category = sample['category']
        l2_error = sample['l2_error']

        error = np.abs(pred - yy)

        # 各自独立的色标范围
        pred_du_abs_max = np.abs(pred_du).max()
        yy_du_abs_max = np.abs(yy_du).max()

        # 列1：pred_du（使用自己的色标）
        ax1 = fig.add_subplot(gs[row, 0])
        im1 = ax1.imshow(pred_du, aspect='auto', cmap='RdBu_r',
                         vmin=-pred_du_abs_max, vmax=pred_du_abs_max,
                         extent=[0, 1, -1, 1], origin='lower')
        ax1.set_ylabel(f'{category.upper()}\nSample {idx}\nx', fontsize=9)
        if row == 0:
            ax1.set_title('Pred Residual (Du)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('t')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax1.text(0.02, 0.98, f'Mean: {np.mean(np.abs(pred_du)):.2e}\nMax: {pred_du_abs_max:.2e}',
                 transform=ax1.transAxes, fontsize=7, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 列2：yy_du（使用自己的色标）
        ax2 = fig.add_subplot(gs[row, 1])
        im2 = ax2.imshow(yy_du, aspect='auto', cmap='RdBu_r',
                         vmin=-yy_du_abs_max, vmax=yy_du_abs_max,
                         extent=[0, 1, -1, 1], origin='lower')
        if row == 0:
            ax2.set_title('GT Residual (Du)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('t')
        ax2.set_ylabel('x')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax2.text(0.02, 0.98, f'Mean: {np.mean(np.abs(yy_du)):.2e}\nMax: {yy_du_abs_max:.2e}',
                 transform=ax2.transAxes, fontsize=7, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 列3：绝对误差（使用自己的色标）
        ax3 = fig.add_subplot(gs[row, 2])
        im3 = ax3.imshow(error, aspect='auto', cmap='hot',
                         extent=[0, 1, -1, 1], origin='lower')
        if row == 0:
            ax3.set_title('|Pred - GT|', fontsize=11, fontweight='bold')
        ax3.set_xlabel('t')
        ax3.set_ylabel('x')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        ax3.text(0.02, 0.98, f'L2: {l2_error:.2e}\nMax: {error.max():.2e}',
                 transform=ax3.transAxes, fontsize=7, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('Residual Comparison: Best (top 3) → Mid (middle 3) → Worst (bottom 3)',
                 fontsize=14, fontweight='bold', y=1.001)

    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/residual_comparison.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{save_dir}/residual_comparison.pdf', bbox_inches='tight')

    plt.show()
    plt.close()


#

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = True
        self.h5 = False
        self._load_file()

    def _load_file(self):

        if self.file_path[-3:] == '.h5':
            self.data = h5py.File(self.file_path, 'r')
            self.h5 = True

        else:
            try:
                self.data = scipy.io.loadmat(self.file_path)
            except:
                self.data = h5py.File(self.file_path, 'r')
                self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if self.h5:
            x = x[()]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


def FC1d(u, order=5):
    if not 1 <= order <= 5:
        raise ValueError(f"`order` must be between 1 and 5; got {order}")

    AQ1 = MatReader(f"FC_data/AlQl_d{order}_C_25.mat", to_cuda=True).read_field('AlQl').double()
    AQ2 = MatReader(f"FC_data/ArQr_d{order}_C_25.mat", to_cuda=True).read_field('ArQr').double()

    u1 = torch.einsum("xy,bcy->bcx", AQ1, u[..., :order])
    u2 = torch.einsum("xy,bcy->bcx", AQ2, u[..., -order:])
    return torch.cat([u, u1 + u2], dim=-1)


def save_and_record(num_epochs, optimizer, scheduler, model, filename='EndModel', ):
    state = {
        'epoch': num_epochs,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state, filename + '.pth.tar')
    torch.save(model, filename + '.pkl')


def save_checkpoint(model, LogIter, optimizer, scheduler, loss_list, test_loss_list, lr_list, model_save_record,
                    grad, filename='checkpoint'):
    state = {
        'epoch': LogIter,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss_list': loss_list,
        'test_loss_list': test_loss_list,
        'lr_list': lr_list,
        'model_save_record': model_save_record,
        'grad': grad
    }
    torch.save(state, filename + '.pth.tar')
    if filename[-4:] == 'best':
        torch.save(model, filename + '.pkl')


def analyze_epoch_loss_simple(epoch_markers, loss_values):
    """
    简化版epoch loss分析，支持一维/二维loss_values输入
    输入:
        epoch_markers: [n_batches,] 标记每个batch所属的epoch（整数）
        loss_values:   [n_batches,] 或 [n_batches, n_loss_types] 的loss值

    输出:
        epoch_stats: {
            'mean': [n_epochs, (n_loss_types)],
            'median': [...],
            'max': [...],
            'min': [...],
            'std': [...]
        }
        intra_anomalies: [异常字典列表]
        inter_anomalies: [突变字典列表]
    """
    unique_epochs = np.unique(epoch_markers)

    # 确保loss_values是二维（n_batches, n_loss_types）
    if loss_values.ndim == 1:
        loss_values = loss_values[:, np.newaxis]
    n_loss_types = loss_values.shape[1]

    # 按epoch重组数据
    epoch_loss = []  # 最终形状: [n_epochs, n_batches_in_epoch, n_loss_types]
    batches_per_epoch = []
    for epoch in unique_epochs:
        mask = (epoch_markers == epoch)
        epoch_loss.append(loss_values[mask])
        batches_per_epoch.append(np.sum(mask))
    max_batches = max(batches_per_epoch)

    # 计算统计量
    epoch_stats = {}
    for stat_name, stat_func in [('mean', np.mean), ('median', np.median),
                                 ('max', np.max), ('min', np.min), ('std', np.std)]:
        # 对每个epoch的batch维度计算统计量
        epoch_stats[stat_name] = np.array([stat_func(epoch, axis=0) for epoch in epoch_loss])

    # 1. 检测epoch内异常
    intra_anomalies = []
    for epoch_idx, epoch in enumerate(unique_epochs):
        for loss_type in range(n_loss_types):
            losses = epoch_loss[epoch_idx][:, loss_type]
            median = np.median(losses)
            mad = stats.median_abs_deviation(losses)
            if mad == 0:
                continue

            z_scores = np.abs(losses - median) / mad
            anomalies = np.where(z_scores > 3.0)[0]
            if len(anomalies) > 0:
                intra_anomalies.append({
                    'epoch': epoch,
                    'loss_type': loss_type,
                    'batch_indices': anomalies,
                    'max_z_score': np.max(z_scores)
                })

    # 2. 检测epoch间异常
    inter_anomalies = []
    for loss_type in range(n_loss_types):
        means = epoch_stats['mean'][:, loss_type]
        changes = np.abs(np.diff(means)) / (means[:-1] + 1e-8)
        median_change = np.median(changes)
        mad_change = stats.median_abs_deviation(changes)

        z_scores = changes / (median_change + mad_change + 1e-8)
        anomalies = np.where(z_scores > 3.0)[0]

        for trans in anomalies:
            inter_anomalies.append({
                'transition': (unique_epochs[trans], unique_epochs[trans + 1]),
                'loss_type': loss_type,
                'change_ratio': changes[trans],
                'epoch_means': (means[trans], means[trans + 1])
            })

    return epoch_stats, intra_anomalies, inter_anomalies


def plot_loss_with_analysis_II(loss_list, lr_list, test_loss_list, grad_array, title="training"):
    """改进的绘图函数，包含：
    1. 学习率曲线（副坐标轴）
    2. 坐标轴颜色与loss类型一致
    3. 只标记每个epoch中最高loss异常点

    [loss.item(), loss_f.item(), total_loss.item(), e]
    绘制data_loss:[:,0]
    绘制training loss：[:,2]
    绘制[test_l2, e]
    绘制lr变化 lr_list.append([current_lr, e])
    grad：[e, loss.item(), grad_norm.cpu()]
    """
    loss_array = np.array(loss_list)
    lr_array = np.array(lr_list)
    test_loss_list = np.array(test_loss_list)
    grad_array = np.array(grad_array)

    unique_epochs = np.unique(loss_array[:, -1].astype(int))

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    loss_types = ['Training Loss', 'Data Loss', 'Testing Loss']
    colors = ['#1f77b4', '#8A0011', '#107A38']  # 标准matplotlib颜色
    lr_color = '#9467bd'  # 学习率曲线颜色
    plot_data = [[loss_array[:, -1].astype(int), loss_array[:, 2], lr_array[:, -1].astype(int), lr_array[:, 0]],
                 # 绘制训练总损失
                 [grad_array[:, 0].astype(int), grad_array[:, 1], grad_array[:, 0].astype(int), grad_array[:, 2]],
                 # 绘制自回归过程中的data损失和梯度变化
                 [test_loss_list[:, -1].astype(int), test_loss_list[:, 0], lr_array[:, -1].astype(int), lr_array[:, 0]],
                 # 绘制测试l2损失
                 ]
    plot_label = [['Training MSE', 'Learning Rate'], ['Training l2 data error', 'grad_norm'],
                  ['Testing l2 data error', 'Learning Rate']]

    for i in trange(len(plot_data)):
        # 主坐标轴设置（保持与loss类型相同颜色）
        # region 绘制基础loss散点图
        data = plot_data[i]
        ax = axes[i]
        if data[0][0] == data[0][1]:
            ax.scatter(data[0],
                       data[1],
                       alpha=0.2, color=colors[i], s=10)
            epoch_stats, intra_anomalies, inter_anomalies = analyze_epoch_loss_simple(
                data[0], data[1])
            ax.plot(unique_epochs, epoch_stats['mean'],
                    color='black', label=f'Epoch Mean of {plot_label[i][0]}', linewidth=1)
        else:
            ax.plot(data[0], data[1],
                    color=colors[i], label=plot_label[i][0], linewidth=1)
        # endregion
        # region 主坐标轴标签设置
        # 主坐标轴标签设置（与loss类型同色）

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Error', fontsize=12, color=colors[i])
        ax.set_yscale('log')
        # ax.tick_params(axis='x', colors=colors[i])
        ax.tick_params(axis='y', colors=colors[i])
        ax.set_title(f"{loss_types[i]}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        # endregion
        # region 绘制副坐标轴数据
        # 添加学习率曲线（副坐标轴）
        ax2 = ax.twinx()
        ax2.plot(data[2], data[3],
                 color=lr_color,
                 linestyle=':',
                 linewidth=1.5,
                 alpha=1,
                 label=plot_label[i][1])
        ax2.set_ylabel(plot_label[i][1], fontsize=12, color=lr_color)
        # ax2.set_yscale('log')
        ax2.tick_params(axis='y', colors=lr_color)

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=10,
                  loc='upper right',
                  framealpha=0.9)
        # endregion

    plt.tight_layout()
    plt.savefig(f'{title}_loss_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('plot succeed')


class CurriculumScheduler:
    def __init__(self,
                 max_t_train,
                 min_steps=20,
                 warmup_epochs=0,
                 rollback_prob=0.1,
                 rollback_decay=0.95):
        """
        课程学习调度器
        :param max_t_train: 最大时间步数（原始t_train）
        :param min_steps: 初始最小预测步数
        :param warmup_epochs: 达到最大步数需要的epoch数
        :param rollback_prob: 回滚概率
        :param rollback_decay: 回滚概率衰减系数
        """
        self.max_t_train = max_t_train
        self.min_steps = min_steps
        self.warmup_epochs = warmup_epochs
        self.rollback_prob = rollback_prob
        self.rollback_decay = rollback_decay
        self.current_t_train = min_steps
        self.base_schedule = [
            min(
                max_t_train,
                min_steps + int(epoch * (max_t_train - min_steps) / warmup_epochs)
            )
            for epoch in range(warmup_epochs + 1)  # +1防止索引越界
        ]
        if (self.max_t_train - self.min_steps) < self.warmup_epochs:
            epoch_per_t = self.warmup_epochs / (self.max_t_train - self.min_steps)
            print(
                f"Curriculum training starts at {self.min_steps} steps, increasing by {epoch_per_t} epoch every t for {self.warmup_epochs} epochs to reach {self.max_t_train} steps")
        else:
            avg_step = (self.max_t_train - self.min_steps) / self.warmup_epochs
            print(
                f"Curriculum training starts at {self.min_steps} steps, increasing by ~{avg_step:.1f} steps on average over {self.warmup_epochs} epochs to reach {self.max_t_train} steps")

    def update(self, epoch):
        """每个epoch开始时更新当前t_train"""
        # 基础线性增长
        new_t_train = self.base_schedule[min(epoch, self.warmup_epochs)]

        # 随机回滚机制
        p = random.random()
        if p < self.rollback_prob and new_t_train > self.min_steps:
            rollback_risol = 0.5 + 0.25 * random.random()
            self.current_t_train = max(
                self.min_steps,
                int(rollback_risol * new_t_train)  # 回滚到50%-100%之间
            )

            self.rollback_prob *= self.rollback_decay  # 衰减回滚概率
        else:
            self.current_t_train = min(new_t_train, self.max_t_train)

        return int(self.current_t_train)


class RobustGradientClipper:
    """
    稳健梯度裁剪器：异常检测 + 跳过 + 固定阈值裁剪
    """

    def __init__(
            self,
            max_norm=1.0,  # 固定裁剪阈值
            anomaly_multiplier=20,  # 超过中位数多少倍视为异常
            warmup_batches=30,  # 预热期（收集统计量）
            history_size=100,  # 历史窗口大小
    ):
        self.max_norm = max_norm
        self.anomaly_multiplier = anomaly_multiplier
        self.warmup_batches = warmup_batches
        self.grad_norm_history = deque(maxlen=history_size)

        # 统计信息
        self.total_batches = 0
        self.skipped_batches = 0

    def compute_grad_norm(self, model):
        """计算原始梯度范数（不修改梯度）"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def step(self, model, epoch=None, batch=None):
        """
        返回: (grad_norm, skipped)
            - grad_norm: 原始梯度范数
            - skipped: 是否跳过本次更新
        """
        self.total_batches += 1
        current_norm = self.compute_grad_norm(model)

        # 检查 NaN/Inf
        if not np.isfinite(current_norm):
            print(f"[E{epoch}B{batch}] ❌ NaN/Inf 梯度，跳过")
            self._zero_grad(model)
            self.skipped_batches += 1
            return current_norm, True

        # 异常检测（预热期后生效）
        if len(self.grad_norm_history) >= self.warmup_batches:
            median = np.median(list(self.grad_norm_history))
            threshold = median * self.anomaly_multiplier

            if current_norm > threshold:
                print(
                    f"[E{epoch}B{batch}] ⚠️ 异常梯度: {current_norm:.4f} > {threshold:.4f} (median={median:.4f})，跳过")
                self._zero_grad(model)
                self.skipped_batches += 1
                # 不记录异常值到历史，避免污染统计
                return current_norm, True

        # 记录正常梯度
        self.grad_norm_history.append(current_norm)

        # 执行裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)

        return current_norm, False

    def _zero_grad(self, model):
        """清零所有梯度"""
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def get_stats(self):
        """获取统计信息"""
        return {
            'total_batches': self.total_batches,
            'skipped_batches': self.skipped_batches,
            'skip_rate': self.skipped_batches / max(self.total_batches, 1) * 100,
            'median_grad': np.median(list(self.grad_norm_history)) if self.grad_norm_history else 0,
            'current_threshold': np.median(list(self.grad_norm_history)) * self.anomaly_multiplier if len(
                self.grad_norm_history) >= self.warmup_batches else None,
        }

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print(f"梯度裁剪统计: 跳过 {stats['skipped_batches']}/{stats['total_batches']} "
              f"({stats['skip_rate']:.1f}%), median={stats['median_grad']:.4f}")


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, lr_min=1e-9
                                    ):
    def lr_lambda(current_step):
        # print('num_warmup_steps:', num_warmup_steps)
        if current_step < num_warmup_steps:  # 线性预热
            return float(current_step) / float(max(1, num_warmup_steps))
        # 余弦衰减
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            lr_min,
            0.5 * (1.0 + math.cos(math.pi * progress)),
        )

    return LambdaLR(optimizer, lr_lambda)


class CustomFormatter(ticker.ScalarFormatter):
    '''
    设置cbar的科学计数法表示
    '''

    def __init__(self, useMathText=True, powerlimits=(-1, 1)):
        super().__init__(useMathText=useMathText)
        self.set_powerlimits(powerlimits)
        self.set_scientific(True)

    def __call__(self, x, pos=None):
        # 缩放数值以适应科学计数法的基数部分
        scale = np.power(10, -self.orderOfMagnitude)
        return f'{x * scale:.1f}'


class DescStr:
    def __init__(self):
        self._desc = ''

    def write(self, instr):
        # 清理控制字符
        cleaned_instr = re.sub('\n|\x1b.*|\r', '', instr)
        # 将清理后的信息存储到 _desc
        self._desc += cleaned_instr

    def read(self, b):
        ret = self._desc
        self._desc = f'batch {b}:'
        return ret

    def flush(self):
        pass


def calculate_fig_size(Aspect_Ratio, word_width=210, word_margins=25.4, ):
    '''
    计算出图时图的大小
    :param Aspect_Ratio: 目标图的横纵比
    :param word_width: 文档宽度
    :param word_margins: 文档横向页边距
    :return:
    '''
    fig_width = (word_width - 2 * word_margins) / 25.4
    fig_lenth = fig_width / Aspect_Ratio
    return [fig_width, fig_lenth]


def round_down_auto(number):
    if number == 0:
        return 0
    decimals = -int(math.floor(math.log10(abs(number))))
    factor = 10 ** decimals
    return math.floor(number * factor) / factor


def plot_visualization_results_2d(visualize_results, save_dir=None, plot_residual=False):
    """
    绘制2D反应扩散方程的可视化结果

    Args:
        visualize_results: 包含样本的列表，每个样本包含:
            - pred: [nx, ny, nt, 2] 预测值
            - yy: [nx, ny, nt, 2] 真实值
            - pred_du: [nx, ny, nt] 或 [nx, ny, nt, 2] 预测残差
            - yy_du: [nx, ny, nt] 或 [nx, ny, nt, 2] 真实残差
            - index: 样本索引
            - category: 'best', 'mid', 'worst'
            - l2_error: L2误差
        save_dir: 保存路径（可选）
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    best_samples = [r for r in visualize_results if r['category'] == 'best']
    mid_samples = [r for r in visualize_results if r['category'] == 'mid']
    worst_samples = [r for r in visualize_results if r['category'] == 'worst']

    # 图1-3：pred vs yy 对比（每类别一张图）
    for category, samples in [('best', best_samples), ('mid', mid_samples), ('worst', worst_samples)]:
        if samples:
            plot_pred_vs_yy_2d_compact(samples, category, save_dir)

    # 图4：残差对比
    if plot_residual:
        plot_residual_comparison_2d(visualize_results, save_dir)


def plot_pred_vs_yy_2d(samples, category, save_dir=None):
    """
    绘制 2D pred vs yy 对比图

    每个样本显示：
    - 行1: u 分量在不同时刻的对比 (pred vs yy)
    - 行2: v 分量在不同时刻的对比 (pred vs yy)
    - 行3: 误差分布
    """
    n_samples = len(samples)
    n_times = 4  # 显示4个时刻: t=0, t=T/3, t=2T/3, t=T

    fig = plt.figure(figsize=(4 * n_times + 2, 4 * n_samples * 3))

    for sample_idx, sample in enumerate(samples):
        pred = sample['pred']  # [nx, ny, nt, 2]
        yy = sample['yy']  # [nx, ny, nt, 2]
        idx = sample['index']
        l2_error = sample['l2_error']

        nx, ny, nt, _ = pred.shape
        t_indices = [0, nt // 3, 2 * nt // 3, nt - 1]

        # u 分量
        for var_idx, var_name in enumerate(['u', 'v']):
            pred_var = pred[..., var_idx]  # [nx, ny, nt]
            yy_var = yy[..., var_idx]

            vmin = min(pred_var.min(), yy_var.min())
            vmax = max(pred_var.max(), yy_var.max())

            for t_col, t_idx in enumerate(t_indices):
                # Prediction
                row = sample_idx * 6 + var_idx * 2
                ax_pred = fig.add_subplot(n_samples * 6, n_times, row * n_times + t_col + 1)
                im = ax_pred.imshow(pred_var[:, :, t_idx].T, origin='lower', cmap='viridis',
                                    vmin=vmin, vmax=vmax, aspect='equal')
                if t_col == 0:
                    ax_pred.set_ylabel(f'Sample {idx}\nPred {var_name}', fontsize=9)
                if sample_idx == 0 and var_idx == 0:
                    ax_pred.set_title(f't = {t_idx}/{nt - 1}', fontsize=10)
                ax_pred.set_xticks([])
                ax_pred.set_yticks([])
                plt.colorbar(im, ax=ax_pred, fraction=0.046, pad=0.04)

                # Ground Truth
                row = sample_idx * 6 + var_idx * 2 + 1
                ax_gt = fig.add_subplot(n_samples * 6, n_times, row * n_times + t_col + 1)
                im = ax_gt.imshow(yy_var[:, :, t_idx].T, origin='lower', cmap='viridis',
                                  vmin=vmin, vmax=vmax, aspect='equal')
                if t_col == 0:
                    ax_gt.set_ylabel(f'GT {var_name}', fontsize=9)
                ax_gt.set_xticks([])
                ax_gt.set_yticks([])
                plt.colorbar(im, ax=ax_gt, fraction=0.046, pad=0.04)

        # 误差行 (u 和 v 分开)
        for var_idx, var_name in enumerate(['u', 'v']):
            error = np.abs(pred[..., var_idx] - yy[..., var_idx])

            for t_col, t_idx in enumerate(t_indices):
                row = sample_idx * 6 + 4 + var_idx
                ax_err = fig.add_subplot(n_samples * 6, n_times, row * n_times + t_col + 1)
                im = ax_err.imshow(error[:, :, t_idx].T, origin='lower', cmap='hot', aspect='equal')
                if t_col == 0:
                    ax_err.set_ylabel(f'|Δ{var_name}|', fontsize=9)
                ax_err.set_xticks([])
                ax_err.set_yticks([])
                plt.colorbar(im, ax=ax_err, fraction=0.046, pad=0.04)

    fig.suptitle(f'{category.upper()} Samples - 2D Prediction vs Ground Truth\n(L2 errors shown per sample)',
                 fontsize=14, fontweight='bold', y=1.001)

    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/{category}_pred_vs_yy_2d.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{save_dir}/{category}_pred_vs_yy_2d.pdf', bbox_inches='tight')

    plt.show()
    plt.close()


def plot_pred_vs_yy_2d_compact(samples, category, save_dir=None):
    """
    更紧凑的2D对比图：每个样本一行，显示关键时刻

    布局: n_samples 行 x 10 列
    每行: [u_IC, u_pred_tT, u_gt_tT, u_error, v_IC, v_pred_tT, v_gt_tT, v_error]

    颜色规则:
    - IC: 单独的 colorbar
    - pred 和 gt: 共用 gt 的 colorbar
    - error: 科学计数法，蓝色系
    """
    n_samples = len(samples)
    n_cols = 8

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(22, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    col_titles = ['u IC (t=0)', 'u Pred (t=T)', 'u GT (t=T)', '|u Error|',
                  'v IC (t=0)', 'v Pred (t=T)', 'v GT (t=T)', '|v Error|']

    for row, sample in enumerate(samples):
        pred = sample['pred']  # [nx, ny, nt, 2]
        yy = sample['yy']
        idx = sample['index']
        l2_error = sample['l2_error']

        nx, ny, nt, _ = pred.shape

        # 分离 u, v 分量
        u_pred, u_gt = pred[..., 0], yy[..., 0]
        v_pred, v_gt = pred[..., 1], yy[..., 1]

        # 计算误差
        u_error = np.abs(u_pred[:, :, -1] - u_gt[:, :, -1])
        v_error = np.abs(v_pred[:, :, -1] - v_gt[:, :, -1])

        # ===== 色标范围 =====
        # IC: 单独范围
        u_ic_vmin, u_ic_vmax = u_gt[:, :, 0].min(), u_gt[:, :, 0].max()
        v_ic_vmin, v_ic_vmax = v_gt[:, :, 0].min(), v_gt[:, :, 0].max()

        # pred 和 gt 共用 gt 的范围 (t=T 时刻)
        u_gt_vmin, u_gt_vmax = u_gt[:, :, -1].min(), u_gt[:, :, -1].max()
        v_gt_vmin, v_gt_vmax = v_gt[:, :, -1].min(), v_gt[:, :, -1].max()

        # 数据列表: (data, vmin, vmax, cmap, is_error)
        data_list = [
            (u_gt[:, :, 0], u_ic_vmin, u_ic_vmax, 'viridis', False),  # u IC
            (u_pred[:, :, -1], u_gt_vmin, u_gt_vmax, 'viridis', False),  # u Pred
            (u_gt[:, :, -1], u_gt_vmin, u_gt_vmax, 'viridis', False),  # u GT
            (u_error, None, None, 'Blues', True),  # u Error
            (v_gt[:, :, 0], v_ic_vmin, v_ic_vmax, 'viridis', False),  # v IC
            (v_pred[:, :, -1], v_gt_vmin, v_gt_vmax, 'viridis', False),  # v Pred
            (v_gt[:, :, -1], v_gt_vmin, v_gt_vmax, 'viridis', False),  # v GT
            (v_error, None, None, 'Blues', True),  # v Error
        ]

        for col, (data, vmin, vmax, cmap, is_error) in enumerate(data_list):
            ax = axes[row, col]

            if is_error:
                # 误差图：使用科学计数法
                im = ax.imshow(data.T, origin='lower', cmap=cmap, aspect='equal')
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format='%.1e')
                cbar.ax.tick_params(labelsize=7)

                # 添加统计信息
                ax.text(0.02, 0.98, f'Max:{data.max():.1e}\nMean:{data.mean():.1e}',
                        transform=ax.transAxes, fontsize=6, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                im = ax.imshow(data.T, origin='lower', cmap=cmap,
                               vmin=vmin, vmax=vmax, aspect='equal')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if row == 0:
                ax.set_title(col_titles[col], fontsize=9, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'#{idx}\nL2={l2_error:.2e}', fontsize=8)

            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f'{category.upper()} Samples - Prediction vs Ground Truth',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/{category}_pred_vs_yy_2d_compact.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{save_dir}/{category}_pred_vs_yy_2d_compact.pdf', bbox_inches='tight')

    plt.show()
    plt.close()


def plot_residual_comparison_2d(visualize_results, save_dir=None):
    """
    绘制2D残差对比图

    每个样本显示: pred_du, yy_du, |pred - yy| 在某个时刻的切片
    """
    sorted_results = (
            [r for r in visualize_results if r.get('category') == 'best'] +
            [r for r in visualize_results if r.get('category') == 'mid'] +
            [r for r in visualize_results if r.get('category') == 'worst']
    )

    n_samples = len(sorted_results)
    if n_samples == 0:
        print("警告：没有可绘制的样本")
        return

    fig = plt.figure(figsize=(18, 4 * n_samples))
    gs = GridSpec(n_samples, 6, figure=fig, hspace=0.3, wspace=0.3)

    col_titles = ['Pred Residual (u)', 'Ref Residual (u)', '|Δu|',
                  'Pred Residual (v)', 'Ref Residual (v)', '|Δv|']

    for row, sample in enumerate(sorted_results):
        pred = sample['pred']  # [nx, ny, nt, 2]
        yy = sample['yy']
        pred_du = sample.get('pred_du')  # [f_u, f_v] 列表
        yy_du = sample.get('yy_du')  # [f_u, f_v] 列表
        idx = sample['index']
        category = sample['category']
        l2_error = sample['l2_error']

        nx, ny, nt = pred.shape[:3]
        t_mid = nt // 2

        # 处理残差：pred_du 和 yy_du 是 [f_u, f_v] 列表
        if pred_du is not None and len(pred_du) == 2:
            pred_du_u = pred_du[0]  # [nx, ny, nt] 或 [nx-2, ny-2, nt-2]
            pred_du_v = pred_du[1]
            # 取中间时刻
            t_mid_du = pred_du_u.shape[-1] // 2
            pred_du_u = pred_du_u[:, :, t_mid_du]
            pred_du_v = pred_du_v[:, :, t_mid_du]
        else:
            pred_du_u = pred_du_v = np.zeros((nx, ny))

        if yy_du is not None and len(yy_du) == 2:
            yy_du_u = yy_du[0]
            yy_du_v = yy_du[1]
            t_mid_du = yy_du_u.shape[-1] // 2
            yy_du_u = yy_du_u[:, :, t_mid_du]
            yy_du_v = yy_du_v[:, :, t_mid_du]
        else:
            yy_du_u = yy_du_v = np.zeros((nx, ny))

        # 计算误差
        error_u = np.abs(pred[:, :, t_mid, 0] - yy[:, :, t_mid, 0])
        error_v = np.abs(pred[:, :, t_mid, 1] - yy[:, :, t_mid, 1])

        data_list = [
            (pred_du_u, 'RdBu_r', True),  # Pred residual u
            (yy_du_u, 'RdBu_r', True),  # Ref residual u
            (error_u, 'Blues', False),  # Error u
            (pred_du_v, 'RdBu_r', True),  # Pred residual v
            (yy_du_v, 'RdBu_r', True),  # Ref residual v
            (error_v, 'Blues', False),  # Error v
        ]

        for col, (data, cmap, symmetric) in enumerate(data_list):
            ax = fig.add_subplot(gs[row, col])

            if symmetric:
                vmax = np.abs(data).max()
                vmin = -vmax
                im = ax.imshow(data.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format='%.1e')
            else:
                im = ax.imshow(data.T, origin='lower', cmap=cmap, aspect='equal')
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format='%.1e')

            cbar.ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{category.upper()}\n#{idx}\nL2={l2_error:.2e}', fontsize=8)

            ax.set_xticks([])
            ax.set_yticks([])

            # 添加统计信息
            ax.text(0.02, 0.98, f'Mean:{np.mean(np.abs(data)):.1e}\nMax:{np.abs(data).max():.1e}',
                    transform=ax.transAxes, fontsize=6, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(f'Residual Comparison at t = T/2 (Best → Mid → Worst)',
                 fontsize=14, fontweight='bold', y=1.001)

    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/residual_comparison_2d.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{save_dir}/residual_comparison_2d.pdf', bbox_inches='tight')

    plt.show()
    plt.close()


def plot_time_evolution_2d(sample, save_dir=None, n_times=6):
    """
    绘制单个样本的时间演化图

    Args:
        sample: 单个样本字典
        save_dir: 保存路径
        n_times: 显示的时刻数
    """
    pred = sample['pred']  # [nx, ny, nt, 2]
    yy = sample['yy']
    idx = sample['index']
    l2_error = sample['l2_error']

    nx, ny, nt, _ = pred.shape
    t_indices = np.linspace(0, nt - 1, n_times, dtype=int)

    fig, axes = plt.subplots(4, n_times, figsize=(3 * n_times, 12))

    row_titles = ['u Prediction', 'u Ground Truth', 'v Prediction', 'v Ground Truth']

    for var_idx, var_name in enumerate(['u', 'v']):
        pred_var = pred[..., var_idx]
        yy_var = yy[..., var_idx]

        vmin = min(pred_var.min(), yy_var.min())
        vmax = max(pred_var.max(), yy_var.max())

        for t_col, t_idx in enumerate(t_indices):
            # Prediction
            ax_pred = axes[var_idx * 2, t_col]
            im = ax_pred.imshow(pred_var[:, :, t_idx].T, origin='lower', cmap='viridis',
                                vmin=vmin, vmax=vmax, aspect='equal')
            if t_col == 0:
                ax_pred.set_ylabel(row_titles[var_idx * 2], fontsize=10)
            ax_pred.set_title(f't = {t_idx}', fontsize=9)
            ax_pred.set_xticks([])
            ax_pred.set_yticks([])
            plt.colorbar(im, ax=ax_pred, fraction=0.046, pad=0.04)

            # Ground Truth
            ax_gt = axes[var_idx * 2 + 1, t_col]
            im = ax_gt.imshow(yy_var[:, :, t_idx].T, origin='lower', cmap='viridis',
                              vmin=vmin, vmax=vmax, aspect='equal')
            if t_col == 0:
                ax_gt.set_ylabel(row_titles[var_idx * 2 + 1], fontsize=10)
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])
            plt.colorbar(im, ax=ax_gt, fraction=0.046, pad=0.04)

    fig.suptitle(f'Sample {idx} Time Evolution (L2 Error: {l2_error:.4e})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/sample_{idx}_time_evolution.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{save_dir}/sample_{idx}_time_evolution.pdf', bbox_inches='tight')

    plt.show()
    plt.close()


class ResolutionPhaseManager:
    def __init__(self, phases):
        self.phases = sorted(phases, key=lambda x: x["start_epoch"])
        self.current_phase_idx = 0
        self.last_resolutions = None
        self.max_epoch = max(phase["start_epoch"] for phase in phases)

    def get_resolutions(self, epoch):
        """根据当前epoch返回应激活的分辨率"""
        # 找到最后一个满足 start_epoch <= epoch 的phase
        for phase in reversed(self.phases):
            if epoch >= phase["start_epoch"]:
                return phase["resolutions"]
        return self.phases[0]["resolutions"]  # 默认返回第一个phase


class CausalCurriculumScheduler:
    """结合课程学习、因果权重和自适应学习率的调度器"""

    def __init__(self,
                 max_t_train,
                 min_steps=20,
                 warmup_epochs=100,
                 # 课程学习参数
                 rollback_prob=0.1,
                 rollback_decay=0.95,
                 # 自适应门控参数（基于 plateau 检测）
                 adaptive_gate=False,
                 loss_plateau_patience=5,
                 loss_plateau_threshold=0.01,
                 force_expand_patience=15,
                 force_patience_early_ratio=1.5,  # 新增
                 force_patience_late_ratio=0.5,
                 # 因果权重参数
                 use_causal_weights=False,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 # 学习率调度参数
                 use_lr_schedule=False,
                 lr_boost=2.0,
                 lr_warmup_epochs=5,
                 lr_scheduler_patience=3,
                 lr_scheduler_factor=0.5,
                 lr_scheduler_threshold=1e-2,
                 lr_min_ratio=0.001,
                 log_file=None):
        """
        参数说明
        ----------
        基础课程学习：
            max_t_train: 最大时间步数
            min_steps: 初始最小预测步数
            warmup_epochs: 达到最大步数的参考epoch数（用于lr调度的进度计算）
            rollback_prob: 回滚概率
            rollback_decay: 回滚概率衰减系数

        自适应门控（adaptive_gate=True 启用）：
            loss_plateau_patience: 连续几个epoch loss不降才算plateau
            loss_plateau_threshold: 下降多少才算"有改善"（如0.01表示1%）
            force_expand_patience: 单个窗口最多待多少epoch

        因果权重（use_causal_weights=True 启用）：
            epsilon_start: 因果强度初始值（强约束）
            epsilon_end: 因果强度最终值（弱约束）

        学习率调度（use_lr_schedule=True 启用）：
            lr_boost: 窗口扩展时lr可提升到的最大倍数
            lr_warmup_epochs: 最大warmup持续epoch数
            lr_scheduler_patience: ReduceLROnPlateau的patience
            lr_scheduler_factor: ReduceLROnPlateau的衰减系数
            lr_scheduler_threshold: ReduceLROnPlateau的threshold
            lr_min_ratio: 最低lr比例

        退化模式
        ----------
        adaptive_gate=False, use_causal_weights=False, use_lr_schedule=False
        → 等价于原始 CurriculumScheduler
        """
        self.log_file = log_file
        print(f"DEBUG: log_file = {self.log_file}")
        # ============ 课程学习参数 ============
        self.max_t_train = max_t_train
        self.min_steps = min_steps
        self.warmup_epochs = warmup_epochs
        self.rollback_prob = rollback_prob
        self.rollback_prob_init = rollback_prob
        self.rollback_decay = rollback_decay

        # ============ 自适应门控参数 ============
        self.adaptive_gate = adaptive_gate
        self.loss_plateau_patience = loss_plateau_patience
        self.loss_plateau_threshold = loss_plateau_threshold
        self.force_expand_patience = force_expand_patience
        self.best_loss_in_window = float('inf')
        self.epochs_no_improve = 0
        self.epochs_in_window = 0

        # ============ 因果权重参数 ============
        self.use_causal_weights = use_causal_weights
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.current_epsilon = epsilon_start

        # ============ 学习率调度参数 ============
        self.use_lr_schedule = use_lr_schedule
        self.lr_boost = lr_boost
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_threshold = lr_scheduler_threshold
        self.lr_min_ratio = lr_min_ratio

        # lr调度状态
        self.optimizer = None
        self.lr_scheduler = None
        self.base_lr = None
        self.epochs_since_expansion = 0
        self.lr_at_window_start = None
        self.lr_target = None
        self.current_warmup_epochs = lr_warmup_epochs
        self.in_warmup = False

        # ============ 当前状态 ============
        self.current_t_train = min_steps
        self.prev_t_train = None
        self.current_epoch = 0

        # ============ 因果矩阵缓存 ============
        self._causal_matrices = {}
        self._print_config()

        # ============ 动态窗口最大epoch ============
        self.force_expand_patience = force_expand_patience
        self.force_patience_early_ratio = force_patience_early_ratio  # 新增
        self.force_patience_late_ratio = force_patience_late_ratio

    def _print_config(self):
        """打印并记录配置信息"""
        config_msg = [
            "=" * 70,
            "Causal Curriculum Scheduler Configuration",
            "=" * 70,
            f"  Time steps: {self.min_steps} → {self.max_t_train}",
            f"  Warmup epochs: {self.warmup_epochs}",
            f"  Rollback prob: {self.rollback_prob}",
        ]

        if self.adaptive_gate:
            config_msg.append(f"  Adaptive gate: ON")
            config_msg.append(f"    - plateau_patience: {self.loss_plateau_patience}")
            config_msg.append(f"    - plateau_threshold: {self.loss_plateau_threshold}")
            config_msg.append(f"    - force_expand_patience: {self.force_expand_patience}")
        else:
            config_msg.append(f"  Adaptive gate: OFF (linear expansion)")

        if self.use_causal_weights:
            config_msg.append(f"  Causal weights: ON")
            config_msg.append(f"    - epsilon: {self.epsilon_start} → {self.epsilon_end}")
        else:
            config_msg.append(f"  Causal weights: OFF")

        if self.use_lr_schedule:
            config_msg.append(f"  LR schedule: ON")
            config_msg.append(f"    - lr_boost: {self.lr_boost}")
            config_msg.append(f"    - lr_warmup_epochs: {self.lr_warmup_epochs}")
            config_msg.append(f"    - lr_scheduler_patience: {self.lr_scheduler_patience}")
            config_msg.append(f"    - lr_scheduler_factor: {self.lr_scheduler_factor}")
            config_msg.append(f"    - lr_min_ratio: {self.lr_min_ratio}")
        else:
            config_msg.append(f"  LR schedule: OFF")

        config_msg.append("=" * 70)

        # 打印到控制台
        for line in config_msg:
            print(line)
        # 写入日志文件
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write("\n")
                for line in config_msg:
                    f.write(line + "\n")
                f.write("\n")

    def _log_event(self, message):
        """打印并写入日志"""
        print(message)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')

    # ================================================================
    # 学习率调度相关
    # ================================================================

    def init_lr_scheduler(self, optimizer):
        """
        初始化学习率调度器（必须在训练开始前调用）
        :param optimizer: PyTorch optimizer
        """
        if not self.use_lr_schedule:
            return

        self.optimizer = optimizer
        self.base_lr = optimizer.param_groups[0]['lr']
        self.lr_at_window_start = self.base_lr
        self.lr_target = self.base_lr
        self.current_warmup_epochs = 0
        self.in_warmup = False
        self._create_new_scheduler()

        print(f"  [LR Scheduler] Initialized with base_lr={self.base_lr:.2e}")

    def _create_new_scheduler(self):
        """创建新的ReduceLROnPlateau调度器"""
        if self.optimizer is None:
            return

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            threshold=self.lr_scheduler_threshold,
            min_lr=self.base_lr * self.lr_min_ratio,
            verbose=True
        )

    def _get_target_lr_for_epoch(self):
        """根据全局epoch进度计算目标lr"""
        global_progress = min(self.current_epoch / max(self.warmup_epochs, 1), 1.0)
        lr_max = self.base_lr * self.lr_boost
        lr_min = self.base_lr * self.lr_min_ratio
        return lr_max - global_progress * (lr_max - lr_min)

    def _reset_lr_for_new_window(self):
        """新窗口开始时判断是否需要warmup"""
        if self.optimizer is None:
            return

        # 1. 记录当前lr
        self.lr_at_window_start = self.optimizer.param_groups[0]['lr']

        # 2. 根据全局进度计算目标lr
        epoch_target_lr = self._get_target_lr_for_epoch()

        # 3. 只有当前lr低于目标时才提升（方案2）
        if self.lr_at_window_start < epoch_target_lr:
            # 需要提升，启动warmup
            self.lr_target = epoch_target_lr
            self.in_warmup = True

            # 根据提升幅度计算warmup长度
            boost_ratio = self.lr_target / max(self.lr_at_window_start, 1e-10)
            warmup_scale = math.log(boost_ratio + 1) / math.log(self.lr_boost + 1)
            self.current_warmup_epochs = max(1, int(self.lr_warmup_epochs * warmup_scale))

            # 重置计数器，创建新scheduler
            self.epochs_since_expansion = 0
            self._create_new_scheduler()

            print(f"  [LR Scheduler] New window (progress={self.current_epoch / self.warmup_epochs:.1%}): "
                  f"{self.lr_at_window_start:.2e} ↑ {self.lr_target:.2e} "
                  f"(×{boost_ratio:.2f}), warmup={self.current_warmup_epochs} epochs")
        else:
            # 当前lr已经足够高，不需要提升
            self.in_warmup = False

    def _warmup_lr(self):
        """Warmup阶段：从上一个窗口的lr逐步增加到目标lr"""
        if self.optimizer is None:
            return

        if self.current_warmup_epochs <= 0:
            current_lr = self.lr_target
        else:
            progress = min(self.epochs_since_expansion / self.current_warmup_epochs, 1.0)
            current_lr = self.lr_at_window_start + progress * (self.lr_target - self.lr_at_window_start)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

    def step_lr_scheduler(self, loss):
        """
        更新学习率调度器（每个epoch结束时调用）
        :param loss: 当前epoch的loss
        """
        if not self.use_lr_schedule or self.optimizer is None:
            # print(f"DEBUG step_lr_scheduler: SKIP - use_lr_schedule={self.use_lr_schedule}, optimizer={self.optimizer}")
            return

        lr_before = self.optimizer.param_groups[0]['lr']
        # print(f"DEBUG step_lr_scheduler: loss={loss:.4e}, lr_before={lr_before:.2e}, in_warmup={self.in_warmup}")

        if self.in_warmup:
            self.epochs_since_expansion += 1
            # print(
            # f"DEBUG: In warmup, epochs_since_expansion={self.epochs_since_expansion}/{self.current_warmup_epochs}")

            if self.epochs_since_expansion <= self.current_warmup_epochs:
                # Warmup阶段：lr逐步增加
                self._warmup_lr()
            else:
                # Warmup结束，切换到ReduceLROnPlateau
                self.in_warmup = False
                self.lr_scheduler.step(loss)
                # print(f"DEBUG: Warmup ended, lr_scheduler.step() called")
        else:
            # 非warmup阶段：直接用ReduceLROnPlateau
            # print(f"DEBUG: Calling lr_scheduler.step(loss={loss:.4e})")
            self.lr_scheduler.step(loss)

        lr_after = self.optimizer.param_groups[0]['lr']
        # print(f"DEBUG step_lr_scheduler: lr_after={lr_after:.2e}")

        if lr_after < lr_before:
            # lr 下降了，重置 plateau 计数，给新 lr 机会
            self.epochs_no_improve = 0
            print(f"  [Scheduler] LR dropped {lr_before:.2e} → {lr_after:.2e}, reset plateau counter")

    # ================================================================
    # 因果权重相关
    # ================================================================

    def get_causal_matrix(self, n_t, device='cpu'):
        """获取因果累积矩阵（带缓存）"""
        if n_t not in self._causal_matrices:
            M = torch.tril(torch.ones(n_t, n_t), diagonal=-1)
            self._causal_matrices[n_t] = M
        return self._causal_matrices[n_t].to(device)

    def compute_causal_weights(self, residual_per_t):
        """
        计算因果权重
        :param residual_per_t: [n_t] 每个时间点的平均残差
        :return: [n_t] 因果权重（detached）
        """
        n_t = residual_per_t.shape[0]
        device = residual_per_t.device

        M = self.get_causal_matrix(n_t, device)
        cumsum_residual = M @ residual_per_t
        weights = torch.exp(-self.current_epsilon * cumsum_residual)

        return weights.detach()

    def compute_weighted_loss(self, residuals, reduce='mean'):
        """
        计算带因果权重的损失
        :param residuals: [n_t, ...] 每个时间点的残差（第一维是时间）
        :param reduce: 'mean' or 'sum'
        :return: (加权损失, 每个时间点的MSE)
        """
        # 计算每个时间点的MSE
        if residuals.dim() > 1:
            residual_per_t = (residuals ** 2).mean(dim=tuple(range(1, residuals.dim())))
        else:
            residual_per_t = residuals ** 2

        if self.use_causal_weights:
            weights = self.compute_causal_weights(residual_per_t)
            weighted_residual = weights * residual_per_t
        else:
            weighted_residual = residual_per_t

        if reduce == 'mean':
            return weighted_residual.mean(), residual_per_t.detach()
        else:
            return weighted_residual.sum(), residual_per_t.detach()

    # ================================================================
    # 核心调度逻辑
    # ================================================================

    def _check_plateau(self, current_loss):
        """
        检查是否达到plateau（loss不再下降）
        :return: True表示可以扩展，False表示继续当前窗口
        """
        # print(f"  DEBUG _check_plateau: epoch={self.current_epoch}, "
        #       f"epochs_in_window={self.epochs_in_window}, "
        #       f"epochs_no_improve={self.epochs_no_improve}, "
        #       f"loss={current_loss}")
        if current_loss is None:
            return False

        self.epochs_in_window += 1

        # 第一次有 loss 时，初始化 best_loss，不做判定
        if self.best_loss_in_window == float('inf'):
            self.best_loss_in_window = current_loss
            self.epochs_no_improve = 0  # 确保重置
            return False

        # 计算相对改善
        improvement = (self.best_loss_in_window - current_loss) / self.best_loss_in_window

        if improvement > self.loss_plateau_threshold:
            # 有改善
            self.best_loss_in_window = current_loss
            self.epochs_no_improve = 0
        else:
            # 没改善
            self.epochs_no_improve += 1

        # 判断是否扩展
        plateau_reached = self.epochs_no_improve >= self.loss_plateau_patience
        progress = self.current_t_train / self.max_t_train
        ratio = self.force_patience_early_ratio - progress * (
                self.force_patience_early_ratio - self.force_patience_late_ratio)
        dynamic_force_patience = max(5, int(self.force_expand_patience * ratio))

        force_expand = self.epochs_in_window >= dynamic_force_patience

        if plateau_reached:
            print(f"  [Scheduler] Plateau detected (no improve for {self.epochs_no_improve} epochs), expanding...")
        elif force_expand:
            print(
                f"  [Scheduler] Force expand (stayed {self.epochs_in_window}/{dynamic_force_patience} epochs in window)")

        return plateau_reached or force_expand

    def _reset_window_tracking(self, current_loss):
        """新窗口开始时重置追踪状态"""
        self.best_loss_in_window = current_loss if current_loss else float('inf')
        self.epochs_no_improve = 0
        self.epochs_in_window = 0

    def update(self, epoch, current_loss=None):
        """
        每个epoch开始时更新调度状态
        :param epoch: 当前epoch
        :param current_loss: 上一个epoch的loss（用于plateau检测）
        :return: 当前应该训练的时间步数
        """
        self.current_epoch = epoch

        # ============ 更新epsilon（如果启用因果权重）============
        if self.use_causal_weights:
            progress = min(epoch / max(self.warmup_epochs, 1), 1.0)
            self.current_epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

        # ============ 自适应门控：基于plateau检测 ============
        if self.adaptive_gate and current_loss is not None:
            can_expand = self._check_plateau(current_loss)

            if not can_expand and self.current_t_train < self.max_t_train:
                return int(self.current_t_train)

            # 可以扩展
            if can_expand and self.current_t_train < self.max_t_train:
                remaining = self.max_t_train - self.current_t_train
                progress = self.current_t_train / self.max_t_train

                # 前期小步（1-2步），后期大步（5-6步）
                base_step = 1 + int(progress * 5)
                expand_step = min(base_step, remaining)

                new_t_train = min(self.current_t_train + expand_step, self.max_t_train)

                if new_t_train > self.current_t_train:
                    msg = (f"\n{'=' * 60}\n"
                           f"[Epoch {self.current_epoch}] Window Expanded: "
                           f"{self.current_t_train} → {new_t_train} (+{expand_step})\n"
                           f"  Progress: {new_t_train}/{self.max_t_train} ({new_t_train / self.max_t_train:.1%})\n"
                           f"  Loss: {current_loss:.2e}")

                    if self.use_causal_weights:
                        msg += f", Epsilon: {self.current_epsilon:.3f}"

                    if self.use_lr_schedule and self.optimizer:
                        msg += f"\n  LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                        msg += f" → target: {self._get_target_lr_for_epoch():.2e}"

                    msg += f"\n{'=' * 60}"

                    self._log_event(msg)
                    self.current_t_train = new_t_train
                    self._reset_window_tracking(current_loss)

                    if self.use_lr_schedule:
                        self._reset_lr_for_new_window()

        else:
            # 非自适应模式：线性扩展（原始逻辑）
            step_size = (self.max_t_train - self.min_steps) / max(self.warmup_epochs, 1)
            target_t_train = min(
                self.max_t_train,
                self.min_steps + int(epoch * step_size)
            )

            # 随机回滚机制
            if random.random() < self.rollback_prob and target_t_train > self.min_steps:
                rollback_ratio = 0.5 + 0.25 * random.random()
                new_t_train = max(self.min_steps, int(rollback_ratio * target_t_train))
                self.rollback_prob *= self.rollback_decay
                print(f"  [Scheduler] Rollback at epoch {epoch}: {self.current_t_train} → {new_t_train}")
                self.current_t_train = new_t_train
            else:
                if target_t_train > self.current_t_train:
                    self.current_t_train = target_t_train
                    if self.use_lr_schedule:
                        self._reset_lr_for_new_window()

        self.prev_t_train = self.current_t_train
        return int(self.current_t_train)

    # ================================================================
    # 获取状态
    # ================================================================

    def get_current_lr(self):
        """获取当前学习率"""
        if self.optimizer is not None:
            return self.optimizer.param_groups[0]['lr']
        return None

    def get_state(self):
        """获取当前完整状态（用于logging）"""
        state = {
            'epoch': self.current_epoch,
            't_train': self.current_t_train,
            'progress': self.current_t_train / self.max_t_train
        }

        if self.adaptive_gate:
            state['epochs_in_window'] = self.epochs_in_window
            state['epochs_no_improve'] = self.epochs_no_improve
            state['best_loss_in_window'] = self.best_loss_in_window

        if self.use_causal_weights:
            state['epsilon'] = self.current_epsilon

        if self.use_lr_schedule and self.optimizer is not None:
            state['lr'] = self.optimizer.param_groups[0]['lr']
            state['in_warmup'] = self.in_warmup
            state['target_lr'] = self._get_target_lr_for_epoch()

        return state

    def reset(self):
        """重置调度器状态"""
        self.current_t_train = self.min_steps
        self.prev_t_train = None
        self.current_epoch = 0
        self.rollback_prob = self.rollback_prob_init

        # 重置自适应门控状态
        self.best_loss_in_window = float('inf')
        self.epochs_no_improve = 0
        self.epochs_in_window = 0

        # 重置因果权重
        self.current_epsilon = self.epsilon_start

        # 重置lr调度
        self.current_warmup_epochs = 0
        self.in_warmup = False
        self.epochs_since_expansion = 0

        if self.base_lr is not None:
            self.lr_at_window_start = self.base_lr
            self.lr_target = self.base_lr

        if self.use_lr_schedule and self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr
            self._create_new_scheduler()


def log_abnormal(file_path, log_type, **kwargs):
    """实时记录异常"""
    with open(file_path, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"│  [{log_type}] {timestamp}\n")
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple)):
                f.write(f"│    {key}: [{value[0]:.4f}, {value[1]:.4f}]\n")
            else:
                f.write(f"│    {key}: {value}\n")
        f.write(f"│    Action: Skipped\n")


# ================================================================
# 测试代码
# ================================================================

if __name__ == "__main__":
    import torch.nn as nn

    print("\n" + "=" * 70)
    print("测试1：原始模式（全部关闭）")
    print("=" * 70)

    scheduler1 = CausalCurriculumScheduler(
        max_t_train=100,
        min_steps=10,
        warmup_epochs=50,
        rollback_prob=0.0,
        adaptive_gate=False,
        use_causal_weights=False,
        use_lr_schedule=False
    )

    for epoch in range(0, 60, 10):
        t = scheduler1.update(epoch)
        print(f"  Epoch {epoch}: t_train = {t}")

    print("\n" + "=" * 70)
    print("测试2：完整模式（基于plateau检测）")
    print("=" * 70)

    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler2 = CausalCurriculumScheduler(
        max_t_train=100,
        min_steps=10,
        warmup_epochs=300,
        adaptive_gate=True,
        loss_plateau_patience=5,
        loss_plateau_threshold=0.01,
        force_expand_patience=15,
        use_causal_weights=True,
        epsilon_start=1.0,
        epsilon_end=0.1,
        use_lr_schedule=True,
        lr_boost=2.0,
        lr_warmup_epochs=5,
        lr_scheduler_patience=3,
        lr_scheduler_factor=0.5
    )

    scheduler2.init_lr_scheduler(optimizer)

    prev_loss = None
    print("\n模拟训练过程：")
    print("-" * 70)

    for epoch in range(500):
        # 模拟loss：快速下降，然后plateau，然后再下降
        if epoch < 10:
            simulated_loss = 0.1 * (0.8 ** epoch)  # 快速下降
        elif epoch < 20:
            simulated_loss = 0.01 + 0.001 * np.random.randn()  # plateau
        elif epoch < 35:
            simulated_loss = 0.01 * (0.9 ** (epoch - 20))  # 再次下降
        elif epoch < 50:
            simulated_loss = 0.002 + 0.0005 * np.random.randn()  # plateau
        else:
            simulated_loss = 0.002 * (0.95 ** (epoch - 50))  # 缓慢下降

        simulated_loss = max(simulated_loss, 1e-6)

        t = scheduler2.update(epoch, current_loss=prev_loss)
        scheduler2.step_lr_scheduler(simulated_loss)
        prev_loss = simulated_loss

        if epoch % 10 == 0 or scheduler2.current_t_train != scheduler2.prev_t_train:
            state = scheduler2.get_state()
            print(f"  Epoch {epoch:2d}: t={state['t_train']:3d}, "
                  f"ε={state.get('epsilon', 0):.2f}, "
                  f"lr={state.get('lr', 0):.2e}, "
                  f"no_improve={state.get('epochs_no_improve', 0)}, "
                  f"loss={simulated_loss:.2e}")

    print("\n" + "=" * 70)
    print("测试3：因果权重计算")
    print("=" * 70)

    residuals = torch.randn(20, 32, 64)
    loss, residual_per_t = scheduler2.compute_weighted_loss(residuals)
    weights = scheduler2.compute_causal_weights(residual_per_t)

    print(f"  Residuals shape: {residuals.shape}")
    print(f"  Weighted loss: {loss.item():.4f}")
    print(f"  Causal weights (first 5): {weights[:5].numpy().round(3)}")
    print(f"  Causal weights (last 5): {weights[-5:].numpy().round(3)}")
