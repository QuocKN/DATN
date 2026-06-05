from __future__ import annotations

from dataclasses import dataclass
import os

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class WaveformPlotter:
    enabled: bool = True
    prefix: str = "waveform"
    max_points: int = 20_000
    subdir_name: str = "waveforms"

    def prepare_output_dir(self, output_dir: str) -> str:
        waveform_dir = self.waveform_dir(output_dir)
        if self.enabled:
            os.makedirs(waveform_dir, exist_ok=True)
        return waveform_dir

    def waveform_dir(self, output_dir: str) -> str:
        return os.path.join(output_dir, self.subdir_name)

    def output_path(self, output_dir: str, index: int) -> str:
        return os.path.join(self.waveform_dir(output_dir), f"{self.prefix}_{index:06d}.png")

    def title(self, source_path: str, index: int, sample_count: int) -> str:
        return f"{os.path.basename(source_path)} | chunk={index} | samples={sample_count}"

    def save(
        self,
        iq: np.ndarray,
        sample_rate: float,
        output_dir: str,
        source_path: str,
        index: int,
        title: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        self.save_image(
            iq=iq,
            sample_rate=sample_rate,
            output_path=self.output_path(output_dir, index),
            title=title or self.title(source_path, index, iq.size),
        )

    def save_image(
        self,
        iq: np.ndarray,
        sample_rate: float,
        output_path: str,
        title: str,
    ) -> None:
        i = self.downsample(iq.real)
        q = self.downsample(iq.imag)
        amp = self.downsample(np.abs(iq))

        n = min(i.size, q.size, amp.size)
        i = i[:n]
        q = q[:n]
        amp = amp[:n]
        t = np.arange(n, dtype=np.float32) / sample_rate

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(t, i, linewidth=0.7)
        axes[0].set_ylabel("I")
        axes[0].grid(alpha=0.25)

        axes[1].plot(t, q, linewidth=0.7, color="tab:orange")
        axes[1].set_ylabel("Q")
        axes[1].grid(alpha=0.25)

        axes[2].plot(t, amp, linewidth=0.7, color="tab:green")
        axes[2].set_ylabel("|IQ|")
        axes[2].set_xlabel("Time (s)")
        axes[2].grid(alpha=0.25)

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def downsample(self, x: np.ndarray) -> np.ndarray:
        if self.max_points <= 0 or x.size <= self.max_points:
            return x
        step = int(np.ceil(x.size / self.max_points))
        return x[::step]
