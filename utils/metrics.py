"""指标聚合工具"""
import numpy as np


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = {}
        self.counts = {}

    def update(self, metrics_dict):
        for k, v in metrics_dict.items():
            self.values[k] = self.values.get(k, 0) + v
            self.counts[k] = self.counts.get(k, 0) + 1

    def average(self):
        return {k: self.values[k] / self.counts[k] for k in self.values}

    def summary(self):
        avg = self.average()
        return ', '.join(f'{k}={v:.4f}' for k, v in avg.items())


def moving_average(data, window=10):
    """滑动平均"""
    if len(data) < window:
        return data
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='valid')
