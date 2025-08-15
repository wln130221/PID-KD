import torch
import torch.nn as nn


class FADLoss(nn.Module):
    def __init__(self, alpha: float = 0.9, use_zn: bool = True, zn_interval=10):
        super(FADLoss, self).__init__()
        self.Kp = nn.Parameter(torch.tensor(0.8))  # 手动初值
        self.Ki = nn.Parameter(torch.tensor(0.4))
        self.Kd = nn.Parameter(torch.tensor(0.2))

        self.alpha = alpha
        self.integral = torch.tensor(0.0)
        self.prev_error = torch.tensor(0.0)

        self.crit = nn.MSELoss()
        self.loss_history = []

        self.use_zn = use_zn
        self.zn_interval = zn_interval
        self.last_zn_epoch = 0  # 初始化为 0，表示从 epoch=0 起计算

    def forward(self, f_s, f_t, current_epoch=True):

        # print(f"[FADLoss] called: Kp={self.Kp.item():.4f}, Ki={self.Ki.item():.4f}, Kd={self.Kd.item():.4f}")
        e_t = self.crit(f_s, f_t)
        self.integral = self.alpha * self.integral + (1.0 - self.alpha) * e_t.detach()
        d_t = e_t.detach() - self.prev_error

        Kp = F.softplus(self.Kp)
        Ki = F.softplus(self.Ki)
        Kd = F.softplus(self.Kd)

        FAD_loss = Kp * e_t + Ki * self.integral + Kd * d_t
        self.prev_error = e_t.detach()

        # 记录 loss 用于调参
        if current_epoch is not None:
            self.loss_history.append(e_t.item())

            if self.use_zn and (current_epoch - self.last_zn_epoch) >= self.zn_interval:
                self._auto_zn_tuning(current_epoch)
                self.last_zn_epoch = current_epoch
        # print( f"[Epoch {current_epoch}] e_t: {e_t.item():.4f}, integral: {self.integral.item():.4f}, d_t: {d_t.item():.4f}")
        # print(f"Kp: {Kp.item():.4f}, Ki: {Ki.item():.4f}, Kd: {Kd.item():.4f}, FAD_loss: {FAD_loss.item():.4f}")

        return FAD_loss

    def _auto_zn_tuning(self, epoch):
        print(f"[Z-N] FADLoss - 自动调参 @Epoch {epoch}...")
        loss_array = np.array(self.loss_history)
        peaks, _ = find_peaks(loss_array, distance=2)
        print(f"[Z-N] 找到的 loss 峰值索引: {peaks}")

        if len(peaks) >= 2:
            tu = np.mean(np.diff(peaks))
            ku = F.softplus(self.Kp).item()
            kp_new = 0.6 * ku
            ki_new = 2 * kp_new / tu
            kd_new = kp_new * tu / 8

            with torch.no_grad():
                self.Kp.copy_(F.softplus_inverse(torch.tensor(kp_new)))
                self.Ki.copy_(F.softplus_inverse(torch.tensor(ki_new)))
                self.Kd.copy_(F.softplus_inverse(torch.tensor(kd_new)))

            print(f"[Z-N] 设置 Kp={kp_new:.4f}, Ki={ki_new:.4f}, Kd={kd_new:.4f}, Tu={tu:.2f}")
        else:
            print("[Z-N] 峰值不足，跳过本轮 FAD 调参")



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import find_peaks

class TDA_KLLoss(nn.Module):
    def __init__(self, init_temp=7.0, Kp=0.5, Ki=0.05, Kd=0.2,
                 temp_range=(1.0, 8.0), window_size=10):
        super(TDA_KLLoss, self).__init__()

        self.T = nn.Parameter(torch.tensor(init_temp), requires_grad=False)

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.temp_min, self.temp_max = temp_range
        self.window_size = window_size
        self.error_window = []
        self.prev_kl = None
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')


    def forward(self, student_logits, teacher_logits, current_epoch=True):
        print(f"[TDA] Epoch {current_epoch} | Current T = {self.T.item():.4f}")

        p_s = F.log_softmax(student_logits / self.T.item(), dim=1)
        p_t = F.softmax(teacher_logits / self.T.item(), dim=1)
        kl_loss = self.kl_loss_fn(p_s, p_t) * (self.T.item() ** 2)

        self._update_temperature(kl_loss.item(), current_epoch)

        return kl_loss

    def _update_temperature(self, current_kl, current_epoch=True):
        if current_epoch < 10:
            self.T.data = torch.tensor(self.temp_max)
            print(f"[Stage 1] Epoch {current_epoch}: T fixed at {self.temp_max}")
            self.CAR_adjust(current_kl)
        elif 10 <= current_epoch < 200:
            print(f"[Stage 2] Epoch {current_epoch}:TDA adjust T")
            self.CAR_adjust(current_kl)

        else:
            decay_epoch = current_epoch - 200
            total_decay = 100
            end_T = self.temp_min

            if decay_epoch >= total_decay:
                new_T = end_T
            else:
                self._car_adjust(current_kl)

            print(f"[Stage 3] Epoch {current_epoch}: Annealing T to {self.T.item():.4f}")

        self.prev_kl = current_kl

    def CAR_adjust(self, current_kl):
        if self.prev_kl is None:
            self.prev_kl = current_kl

        error = current_kl - self.prev_kl
        self.error_window.append(error)
        if len(self.error_window) > self.window_size:
            self.error_window.pop(0)

        integral = sum(self.error_window) / len(self.error_window)
        derivative = error - self.error_window[-2] if len(self.error_window) > 1 else 0.0

        delta_T = self.Kp * error + self.Ki * integral + self.Kd * derivative

        with torch.no_grad():
            new_T = self.T + delta_T
            new_T = torch.clamp(new_T, self.temp_min, self.temp_max)
            self.T.copy_(new_T)

        print(f"[TDA] ΔT: {delta_T:.4f}, New T: {self.T.item():.4f}")
        self.prev_kl = current_kl
