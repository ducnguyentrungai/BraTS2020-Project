import time
import os
import numpy as np
from pytorch_lightning.callbacks import Callback

class TrainingTimerCallback(Callback):
    def __init__(self, save_path="training_time.txt"):
        super().__init__()
        self.save_path = save_path
        self.epoch_times = []

    def on_fit_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        print(f"[Epoch {trainer.current_epoch}] ⏱ Thời gian: {epoch_time:.2f} giây")

    def on_fit_end(self, trainer, pl_module):
        total_time = time.time() - self.start_time
        avg_epoch_time = np.mean(self.epoch_times)

        def format_seconds(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h:02d}h:{m:02d}m:{s:02d}s"

        log_lines = [
            f"Tổng thời gian huấn luyện: {format_seconds(total_time)} ({total_time:.2f} giây)\n",
            f"Thời gian trung bình mỗi epoch: {format_seconds(avg_epoch_time)} ({avg_epoch_time:.2f} giây)\n",
        ]
        log_lines += [
            f"Epoch {i}: {format_seconds(t)} ({t:.2f} giây)\n"
            for i, t in enumerate(self.epoch_times)
        ]

        with open(self.save_path, "w") as f:
            f.writelines(log_lines)

        print("📝 Đã lưu thời gian huấn luyện vào:", self.save_path)
