import argparse
import os
import sys
import torch.optim as optim
from load_dataset_V2 import VideoSRDataset
from model2 import VSRRDN, VSRRDN_woSTCSA
import torch
from torch.utils.data import DataLoader
from torch.autograd import grad
from utilis import CharbonnierLoss, SSIMLoss
import warnings
import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure
import time
warnings.filterwarnings("ignore", category=UserWarning)

def get_all_folders(path):
    folder_paths = []
    for root, dirs, files in os.walk(path):
        for directory in dirs:
            folder_path = os.path.join(root, directory)
            folder_paths.append(folder_path)
    return folder_paths

def gradient_penalty(gp_discriminator, real_samples, fake_samples, gp_device='cuda'):
    gp_alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(gp_device)
    gp_alpha = gp_alpha.expand_as(real_samples)

    interpolated = gp_alpha * real_samples + ((1 - gp_alpha) * fake_samples)
    interpolated = interpolated.to(gp_device)
    interpolated.requires_grad_(True)

    d_interpolated = gp_discriminator(interpolated)

    gradients = grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(d_interpolated.size()).to(gp_device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale_factor', type=int, default=6, help='Downsampling factor')
    parser.add_argument('--num_frames', type=int, default=5, help='input number of frames')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--StartEpochs', type=int, default=0, help='number of epochs to train for')
    parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
    parser.add_argument('--generatorWeights', type=str, default='', help='path to generator weights (to continue training)')
    parser.add_argument('--datapath', type=str, default='dataset5/', help='folder of dataset')
    parser.add_argument('--out', type=str, default='checkpoint', help='folder to output model checkpoints')
    opt = parser.parse_args()
    print(opt)
    try:
        os.makedirs(opt.out)
    except OSError:
        pass
    # init TensorBoard
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"exp_{timestamp}"
    log_dir = os.path.join("runs", experiment_name)
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device('cuda')
    video_dirs = get_all_folders(opt.datapath)
    train_dataset = VideoSRDataset(hr_dirs=video_dirs, h_scale=opt.scale_factor, h_patch_size = 96, w_patch_size = 96,
                                   w_scale=opt.scale_factor, input_frames=opt.num_frames,D_Flag=True)
    train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle=True, num_workers=8)
    generator = VSRRDN_woSTCSA(upscale_factor=opt.scale_factor).to(device)
    if opt.generatorWeights != '':
        generator.load_state_dict(torch.load(opt.generatorWeights))

    # pretrain loss
    criterion = CharbonnierLoss().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction=None).to(device)
    optim_generator = optim.AdamW(generator.parameters(), lr=0.0001)
    train_losses = []
    c = 1
    h = 96
    w = 96
    middle_frame_idx = opt.num_frames // 2
    len_train = len(train_loader)
    best_psnr = 0.0
    best_ssim = 0.0
    ssim_weights = 0.1
    print('Training Start')
    for epoch in range(opt.StartEpochs, opt.nEpochs):
        epoch_start_time = time.time()
        # Training
        generator.train()
        epoch_loss = 0.0
        mean_pixel_loss = 0.0
        mean_ssim_loss = 0.0
        for batch_idx, (lr_seq, hr_target, hr_seq) in enumerate(train_loader):
            batch_size = lr_seq.size(0) 
            lr_seq = lr_seq.to(device)
            hr_seq = hr_seq.to(device)

            hr_seq_reshaped = hr_seq.view(-1, c, h, w)
            hr_middle_frame = hr_seq[:, middle_frame_idx, ...]
            middle_expanded_reshaped = hr_middle_frame.unsqueeze(1).repeat(1, opt.num_frames, 1, 1, 1).view(-1, c, h, w)
            init_ssim_value = ssim(hr_seq_reshaped, middle_expanded_reshaped).view(batch_size, opt.num_frames) #[batch,num_frames]

            optim_generator.zero_grad()
            high_res_real = hr_target.to(device)
            high_res_fake = generator(lr_seq)

            fake_expand_reshaped = high_res_fake.unsqueeze(1).repeat(1, opt.num_frames, 1, 1, 1).view(-1, c, h, w)
            result_ssim_value = ssim(hr_seq_reshaped, fake_expand_reshaped).view(batch_size, opt.num_frames)  # [batch,num_frames]

            pixel_loss = criterion(high_res_fake, high_res_real)
            ssim_loss = criterion(init_ssim_value,result_ssim_value)
            generator_total_loss = pixel_loss + ssim_weights * ssim_loss

            generator_total_loss.backward()
            optim_generator.step()
            epoch_loss += generator_total_loss.item()
            mean_pixel_loss += pixel_loss.item()
            mean_ssim_loss += ssim_loss.item()
        epoch_loss /= len_train
        mean_pixel_loss /= len_train
        mean_ssim_loss /= len_train
        train_losses.append(epoch_loss)
        epoch_time = time.time() - epoch_start_time
        sys.stdout.write('\r[%d/%d] Generator_Loss: %.4f, Time: %.2f seconds\n' % (epoch+1, opt.nEpochs, epoch_loss, epoch_time))

        writer.add_scalar('Train/Generator_Loss', epoch_loss, epoch + 1)
        writer.add_scalar('Train/Mean_pixel_loss', mean_pixel_loss, epoch + 1)
        writer.add_scalar('Train/Mean_ssim_loss', mean_ssim_loss, epoch + 1)
        if ((epoch+1) % 20 == 0):
            torch.save(generator.state_dict(), f'{opt.out}/generator_epoch_{epoch+1}.pth')
    writer.close()

    plt.figure("Generator_Loss", (18, 6))
    plt.title("Generator_Loss")
    x = [i + 1 for i in range(len(train_losses))]
    y = [train_losses[i] for i in range(len(train_losses))]
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")

    plt.show()


