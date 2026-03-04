from model import VSRResNet,FSTRN,FD_UNet
from model2 import VSRRDN,VSRRDN_woSTCSA
from model3 import RDN
import torch
import cv2
import numpy as np
from torchvision import transforms
import statistics
from utilis import PSNRCalculator
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips
import warnings
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
                          interpolation=cv2.INTER_LINEAR)
    return sr_patch

if __name__ == '__main__':
    num_frames = 5
    scale_factor = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_index = 2 #1-Proposed,2-VSRResFeatGAN,3-FSTRN,4-SRRDN,5-FD_Unet
    # Dataset_Path = 'Result/Pseudo_test/data'
    Dataset_Path = 'Result/test/data'
    Sample_Flag = True
    if model_index == 1:
        generator = VSRRDN(upscale_factor=scale_factor).to(device)
        generator.eval()
        generator.load_state_dict(torch.load('Result/Propose_{}x.pth'.format(scale_factor), map_location=device))
        # generator.load_state_dict(torch.load('Result/Propose_wSTCSA.pth', map_location=device))
        # generator.load_state_dict(torch.load('Result/Propose_{}f.pth'.format(num_frames), map_location=device))
    elif model_index == 2:
        generator = VSRResNet(scale_factor=scale_factor,num_frames=num_frames).to(device)
        generator.eval()
        generator.load_state_dict(torch.load('Result/VSRResFeatGAN_{}x.pth'.format(scale_factor), map_location=device))
    elif model_index == 3:
        generator = FSTRN(scale_factor=scale_factor).to(device)
        generator.eval()
        generator.load_state_dict(torch.load('Result/FSTRN_{}x.pth'.format(scale_factor), map_location=device))
    elif model_index == 4:
        generator = RDN(upscale_factor=scale_factor).to(device)
        generator.eval()
        generator.load_state_dict(torch.load('Result/SRRDN_{}x.pth'.format(scale_factor), map_location=device))
    elif model_index == 5:
        generator = FD_UNet().to(device)
        generator.eval()
        generator.load_state_dict(torch.load('Result/FD_UNet_{}x.pth'.format(scale_factor), map_location=device))
        Sample_Flag = False
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    central_index = 5
    total_ssim = []
    total_psnr = []
    total_lpips = []
    for i in range(4):
        hr_frames = []
        lr_patches = []
        lr_patche1s = []
        start_index = central_index - num_frames // 2
        datapath = f"{Dataset_Path}{i+1}"
        print(datapath)
        lr_sequence = []
        if model_index == 4 or model_index == 5 or model_index == 6:
            num = central_index
            hr_frame = cv2.imread(f"{datapath}/{num}_warped.png", 0)
            # print(hr_frame.shape)
            h, w = hr_frame.shape
            hr_frame = normalized(hr_frame).astype(np.float32)
            h_scale = scale_factor
            w_scale = scale_factor
            lr_patch = dowmsampling(hr_frame, h_scale, w_scale, Sample_Flag) 
            lr_patch = normalized(lr_patch)
            lr_patch = transform(lr_patch)
            lr_patch = lr_patch.unsqueeze(0)
            lr_sequence = lr_patch.to(device)
            hr_center = transform(hr_frame).to(device)
        else:
            for t in range(num_frames):
                num = t+start_index
                hr_frame = cv2.imread(f"{datapath}/{num}_warped.png", 0)  
                # hr_frame = hr_frame[:,129:640]
                hr_frame = normalized(hr_frame).astype(np.float32)
                h_scale = scale_factor
                w_scale = scale_factor
                lr_patch = dowmsampling(hr_frame, h_scale, w_scale, True)
                lr_patch = normalized(lr_patch)
                hr_frames.append(transform(hr_frame))
                lr_patches.append(transform(lr_patch))
            lr_sequence = torch.stack(lr_patches)
            hr_center = hr_frames[num_frames // 2] 
            lr_center = lr_patches[num_frames // 2]
            lr_sequence = lr_sequence.unsqueeze(0).to(device)
        if model_index == 6:
            hr_pred = upsampling(lr_sequence.cpu().squeeze().numpy(), scale_factor, scale_factor)
            hr_pred = transform(hr_pred).unsqueeze(0).to(device)
        else:
            with torch.no_grad():
                hr_pred = generator(lr_sequence)
        result = hr_pred.squeeze().detach().cpu().numpy()
        result = normalized(result)
        lpips_loss_fn = lpips.LPIPS(net='vgg',verbose=False).to(device)
        lpips_value = lpips_loss_fn(hr_pred, hr_center.unsqueeze(0).to(device)).item()
        psnr_count = PSNRCalculator().to(device)
        ssim_count = StructuralSimilarityIndexMeasure(data_range=1.0, reduction=None).to(device)
        psnr = psnr_count(hr_pred, hr_center.unsqueeze(0).to(device)).item()
        ssim = ssim_count(hr_pred, hr_center.unsqueeze(0).to(device)).item()
        print(f"{ssim:.4f}, {psnr:.2f}, LPIPS: {lpips_value:.4f}")
        total_ssim.append(ssim)
        total_psnr.append(psnr)
        total_lpips.append(lpips_value)
    maen_ssim = statistics.mean(total_ssim)
    maen_psnr = statistics.mean(total_psnr)
    maen_lpips = statistics.mean(total_lpips)
    var_ssim = statistics.pstdev(total_ssim)
    var_psnr = statistics.pstdev(total_psnr)
    var_lpips = statistics.pstdev(total_lpips)
    print(f"{maen_ssim:.4f}-{var_ssim:.4f}, {maen_psnr:.2f}-{var_psnr:.2f}, {maen_lpips:.4f}-{var_lpips:.4f}")




