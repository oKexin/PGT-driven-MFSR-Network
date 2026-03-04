import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import os
import matplotlib.pyplot as plt
import time
import random
from torchmetrics.image import StructuralSimilarityIndexMeasure

def normalized(image):
    return (image - image.min()) / (image.max() - image.min())

def dowmsampling(image, h_ratio, w_ratio, flag = True):
    h, w = image.shape[:2]
    if flag:
        lr_patch = image[::h_ratio, ::w_ratio]
    else:
        lr_patch = np.zeros((h, w), dtype=image.dtype)
        lr_patch[::h_ratio, ::w_ratio] = image[::h_ratio, ::w_ratio]
    return lr_patch
def is_high_frequency_patch(img_patch, brightness_threshold=3, variance_threshold1=10, variance_threshold2=200):
    # 计算亮度均值和方差
    brightness_mean = np.mean(img_patch)
    brightness_var = np.var(img_patch)

    # 快速过滤纯黑/近黑块
    if (brightness_mean < brightness_threshold or brightness_var < variance_threshold1 or
            (brightness_mean< 10 and brightness_var > variance_threshold2)):
        return False, {
            'brightness_mean': brightness_mean,
            'brightness_var': brightness_var,
        }
    return True, {
        'brightness_mean': brightness_mean,
        'brightness_var': brightness_var,
    }

def random_rotate(patch, angle):
    """随机旋转90/180/270度或不旋转"""
    if angle == 90:
        return cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(patch, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:  # 0度
        return patch

def random_flip(patch, flip_type):
    """随机水平或垂直翻转"""
    if flip_type is None:
        return patch
    return cv2.flip(patch, flip_type)

def add_gaussian_noise(patch, sigma):
    """添加高斯噪声（仅对低分辨率帧，模拟真实场景的噪声）"""
    gauss = np.random.normal(0, sigma, patch.shape)
    noisy_patch = patch + gauss
    noisy_patch = np.clip(noisy_patch, 0, 1)  # 归一化后的值需限制在[0,1]
    return noisy_patch

class VideoSRDataset(Dataset):
    def __init__(self, hr_dirs, h_scale, w_scale, input_frames, D_Flag, h_patch_size=96, w_patch_size=96):
        self.hr_dirs = hr_dirs  # 所有训练数据路径
        self.h_scale = h_scale  # 超分辨率尺度因子
        self.w_scale = w_scale  # 超分辨率尺度因子
        self.h_patch_size = h_patch_size
        self.w_patch_size = w_patch_size
        self.input_frames = input_frames
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # 统计每个文件夹的帧数
        self.dir_frame_counts = []
        self.total_sequences = 0
        self.downsample_flag = D_Flag
        for hr_dir in self.hr_dirs:
            # 假设帧文件按数字顺序命名：1_warped.png, 2_warped.png, ...
            frame_count = len([f for f in os.listdir(hr_dir) if f.endswith('_warped.png')])
            # 计算可提取的序列数（每个序列包含input_frames帧）
            seq_count = max(0, frame_count - input_frames + 1)
            self.dir_frame_counts.append((frame_count, seq_count))
            self.total_sequences += seq_count

        # 构建累积序列索引，用于快速定位
        self.cumulative_seq_counts = [0]
        for count in self.dir_frame_counts:
            self.cumulative_seq_counts.append(self.cumulative_seq_counts[-1] + count[1])

    def __getitem__(self, idx):
        # 确定当前索引对应的文件夹和帧起始位置
        dir_idx = 0
        while idx >= self.cumulative_seq_counts[dir_idx + 1]:
            dir_idx += 1
        # 计算在当前文件夹中的序列起始帧索引
        seq_start_idx = idx - self.cumulative_seq_counts[dir_idx]
        hr_dir = self.hr_dirs[dir_idx]
        hr_patches = []
        lr_patches = []
        # print(hr_dir)
        # 循环采样直到找到高频子块
        hr_frame = cv2.imread(f"{hr_dir}/{seq_start_idx + 1}_warped.png", 0)  # 灰度图
        h, w = hr_frame.shape
        # 循环查找高质量特征补丁
        # h_start = np.random.randint(0, h - self.h_patch_size + 1)
        # w_start = np.random.randint(0, w - self.w_patch_size + 1)
        while True:
            h_start = np.random.randint(0, h - self.h_patch_size + 1)
            w_start = np.random.randint(0, w - self.w_patch_size + 1)
            hr_patch = hr_frame[h_start:h_start + self.h_patch_size, w_start:w_start + self.w_patch_size]
            is_high_freq, _ = is_high_frequency_patch(hr_patch)
            if is_high_freq:
                break

        # 获取随机旋转角度
        angle = random.choice([0, 90, 180, 270])
        # 获取随机翻转模式
        flip_type = random.choice([-1, 0, 1, None])  # -1: 水平+垂直, 0: 垂直, 1: 水平, None: 不翻转
        # 获取随机降采样噪声方差
        sigma = np.random.uniform(0.001, 0.025)
        for t in range(seq_start_idx , seq_start_idx + self.input_frames):
            # 读取高分辨率帧
            hr_frame = cv2.imread(f"{hr_dir}/{t+1}_warped.png", 0)  # 灰度图
            hr_patch = hr_frame[h_start:h_start + self.h_patch_size, w_start:w_start + self.w_patch_size]
            # 随机旋转
            hr_patch = random_rotate(hr_patch, angle)
            # 随机翻转
            hr_patch = random_flip(hr_patch, flip_type)
            hr_patch = normalized(hr_patch).astype(np.float32)
            # 生成低分辨率补丁
            lr_patch = dowmsampling(hr_patch, self.h_scale, self.w_scale, self.downsample_flag) #True插值， False零填充
            # 对低分辨率补丁添加噪声（模拟真实场景）
            lr_patch = add_gaussian_noise(lr_patch,sigma)  # 仅对低分辨率帧加噪声
            lr_patch = normalized(lr_patch).astype(np.float32)
            hr_patches.append(self.transform(hr_patch))
            lr_patches.append(self.transform(lr_patch))
        # 组装输入：[input_frames, 1, patch_size, patch_size]
        __lr_sequence = torch.stack(lr_patches)
        __hr_sequence = torch.stack(hr_patches)
        __hr_center = hr_patches[self.input_frames // 2]  # 中心高分辨率帧
        return __lr_sequence, __hr_center, __hr_sequence

    def __len__(self):
        return  self.total_sequences

def get_all_folders(path):
    folder_paths = []
    for root, dirs, files in os.walk(path):
        # root 是当前遍历到的目录路径
        # dirs 是当前目录下的子文件夹列表
        for directory in dirs:
            folder_path = os.path.join(root, directory)
            folder_paths.append(folder_path)
    return folder_paths

if __name__ == "__main__":
    video_dirs  = get_all_folders('dataset8/')
    train_dataset = VideoSRDataset(hr_dirs=video_dirs, input_frames=1,h_scale=2,w_scale=2,D_Flag=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=1)
    batch_idx, (lr_sequence, hr_center, _) = list(enumerate(train_loader))[0]
    fig, axes = plt.subplots(2, 3)
    start_epoch = time.time()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction=None)
    for epoch in range(2):
        for batch_idx,(lr_seq, hr_target, _) in enumerate(train_loader):
            if batch_idx == 0:
                # batch, num_frames, c, h, w = lr_seq.size()
                # lr_seq_reshaped = lr_seq.view(-1, c, h, w)
                # print(lr_seq_reshaped.shape)
                # lr_middle_frame = lr_seq[:, 1, ...]
                # lr_middle_expanded = lr_middle_frame.unsqueeze(1).repeat(1, 3, 1, 1, 1)
                # middle_expanded_reshaped = lr_middle_expanded.view(-1, c, h, w)
                # print(middle_expanded_reshaped.shape)
                # ssim_value = ssim(lr_seq_reshaped, middle_expanded_reshaped)
                # print(ssim_value)
                print(lr_seq.shape)
                axes[epoch][0].imshow(lr_seq.squeeze()[0])
                axes[epoch][1].imshow(lr_seq.squeeze()[0])
                axes[epoch][2].imshow(lr_seq.squeeze()[0])
            else:
                pass
    plt.show()
