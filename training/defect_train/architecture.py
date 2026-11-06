"""EfficientAD (small) network, packaged as one module that outputs an anomaly map.

get_autoencoder and get_pdn_small are copied unchanged from nelson1425/EfficientAD,
commit fcab5146f84ae17597044ad5ddf1656ccf805401, Apache-2.0. See NOTICE.
"""
import torch
from torch import nn


def get_autoencoder(out_channels=384):
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
        # decoder
        nn.Upsample(size=3, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=8, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=56, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3,
                  stride=1, padding=1)
    )

def get_pdn_small(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )


class AnomalyNet(nn.Module):
    """Teacher, student and optional autoencoder; forward returns the anomaly map.

    Adapted from predict in efficientad.py. Without an autoencoder, the student has
    the same number of channels as the teacher and the map is the student-teacher one.
    """

    def __init__(self, teacher, student, autoencoder, teacher_mean, teacher_std,
                 q_st_start, q_st_end, q_ae_start, q_ae_end):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.autoencoder = autoencoder
        for name, value in (("teacher_mean", teacher_mean), ("teacher_std", teacher_std),
                            ("q_st_start", q_st_start), ("q_st_end", q_st_end),
                            ("q_ae_start", q_ae_start), ("q_ae_end", q_ae_end)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))

    def forward(self, image):
        teacher_output = (self.teacher(image) - self.teacher_mean) / self.teacher_std
        student_output = self.student(image)
        out_channels = teacher_output.shape[1]
        map_st = torch.mean((teacher_output - student_output[:, :out_channels]) ** 2,
                            dim=1, keepdim=True)
        map_st = 0.1 * (map_st - self.q_st_start) / (self.q_st_end - self.q_st_start)
        if self.autoencoder is None:
            return map_st
        map_ae = torch.mean((self.autoencoder(image) - student_output[:, out_channels:]) ** 2,
                            dim=1, keepdim=True)
        map_ae = 0.1 * (map_ae - self.q_ae_start) / (self.q_ae_end - self.q_ae_start)
        return 0.5 * map_st + 0.5 * map_ae


def build_random_net(image_size=256, use_autoencoder=True):
    """Build an untrained network with the same shape as a trained one, for timing."""
    if use_autoencoder and image_size != 256:
        raise ValueError("the autoencoder has fixed sizes and needs image_size 256")
    out_channels = 384
    student_channels = out_channels * 2 if use_autoencoder else out_channels
    autoencoder = get_autoencoder(out_channels) if use_autoencoder else None
    net = AnomalyNet(
        teacher=get_pdn_small(out_channels),
        student=get_pdn_small(student_channels),
        autoencoder=autoencoder,
        teacher_mean=0.0, teacher_std=1.0,
        q_st_start=0.0, q_st_end=1.0, q_ae_start=0.0, q_ae_end=1.0,
    )
    return net.eval()
