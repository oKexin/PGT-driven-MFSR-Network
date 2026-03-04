import matplotlib.pyplot as plt
from model2 import VSRRDN
import torch
import cv2
import numpy as np
from torchvision import transforms
from utilis import PSNRCalculator
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips
import warnings
import torch.nn.functional as F
import os
warnings.filterwarnings("ignore", category=UserWarning)

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

def upsampling(image, h_ratio, w_ratio):
    h, w = image.shape[:2]
    sr_patch = cv2.resize(image, (w * w_ratio, h * h_ratio),
                          interpolation=cv2.INTER_CUBIC)
    return sr_patch


if __name__ == '__main__':
    num_frames = 5
    scale_factor = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = VSRRDN(upscale_factor=scale_factor).to(device)
    generator.eval()
    generator.load_state_dict(torch.load('Result/Propose_{}x.pth'.format(scale_factor), map_location=device))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # datapath = 'dataset11/data3'
    hr_frames = []
    lr_patches = []
    lr_patche1s = []
    central_index = 5
    start_index = central_index - num_frames // 2
    for t in range(num_frames):
        num = t+start_index
        hr_frame = cv2.imread(f"{'Result/Test/data1'}/{num}_warped.png", 0)
        hr_frame = normalized(hr_frame).astype(np.float32)
        h_scale = scale_factor
        w_scale = scale_factor
        lr_patch = dowmsampling(hr_frame, h_scale, w_scale, True)  # False-zero fill
        lr_patch = normalized(lr_patch)
        hr_frames.append(transform(hr_frame))
        lr_patches.append(transform(lr_patch))
    # input：[num_frames, 1, patch_size, patch_size]
    lr_sequence = torch.stack(lr_patches)
    hr_center = hr_frames[num_frames // 2] 
    lr_center = lr_patches[num_frames // 2]
    lr_sequence = lr_sequence.unsqueeze(0).to(device)
    with torch.no_grad():
        hr_pred = generator(lr_sequence)
    # print(hr_pred.shape)
    result = hr_pred.squeeze().detach().cpu().numpy()
    # print(np.max(result), np.min(result))
    result = normalized(result)

    # fig, axes = plt.subplots(3, 1)
    # axes[0].imshow(hr_center.squeeze(), cmap='hot')
    # axes[1].imshow(lr_center.squeeze(), cmap='hot')
    # axes[2].imshow(result, cmap='hot')
    # # axes[2].imshow(hr_center2.squeeze(), cmap='hot')
    # # axes[0].axis('off')
    # # axes[1].axis('off')
    # # axes[2].axis('off')
    # plt.figure()
    # plt.imshow(hr_center.squeeze(), cmap='hot')
    # plt.axis('off')
    # plt.savefig('HR.png', dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.figure()
    # plt.imshow(lr_center.squeeze(), cmap='hot')
    # plt.axis('off')
    # plt.savefig('LR.png', dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.figure()
    # plt.imshow(result, cmap='hot')
    # plt.axis('off')
    # plt.savefig('Propose.png', dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show()
    # #Count
    # 
    # lpips_loss_fn = lpips.LPIPS(net='vgg', verbose=False).to(device)
    # lpips_value = lpips_loss_fn(hr_pred, hr_center.unsqueeze(0).to(device)).item()
    # psnr_count = PSNRCalculator().to(device)
    # ssim_count = StructuralSimilarityIndexMeasure(data_range=1.0, reduction=None).to(device)
    # psnr = psnr_count(hr_pred, hr_center.unsqueeze(0).to(device)).item()
    # ssim = ssim_count(hr_pred, hr_center.unsqueeze(0).to(device)).item()
    # print(f"{ssim:.4f}, {psnr:.2f}, LPIPS: {lpips_value:.4f}")

        # image_uint8 = (result * 255).astype(np.uint8)
        # save_path = 'Result/TC/propose/{}_warped.png'.format(central_index-14)
        # save_path2 = 'Result/TC/raw/{}_warped.png'.format(central_index-14)
        # if os.path.exists(save_path):
        #     try:
        #         os.remove(save_path)
        #     except PermissionError:
        #         pass
        # if os.path.exists(save_path2):
        #     try:
        #         os.remove(save_path2)
        #     except PermissionError:
        #         pass
        # cv2.imwrite(save_path, image_uint8)
        # raw = cv2.imread(f"{datapath}/{central_index}_warped.png", 0)
        # cv2.imwrite(save_path2, raw)

    # ratio analysis
    # hr_uint8 = (hr_center.squeeze(0).cpu().numpy()* 255).astype(np.uint8)
    # lr_uint8 = (lr_center.squeeze(0).cpu().numpy()* 255).astype(np.uint8)
    result_uint8 = (result * 255).astype(np.uint8)
    # cv2.imwrite('Result/clear/HR2_{}x.png'.format(scale_factor), hr_uint8)
    # cv2.imwrite('Result/clear/LR2_{}x.png'.format(scale_factor), lr_uint8)
    cv2.imwrite('Result/clear/Propose_{}x.png'.format(scale_factor), result_uint8)



