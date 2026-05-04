import math
import sys
import os
import shutil

import torch
import numpy as np
import cv2
import torch as th
import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
from pytorch_fid import fid_score
import copy
import lpips
from guided_diffusion import dist_util
import matplotlib.pyplot as plt

def train_one_epoch_control(diffusion,model,control, data_loader,optimizer, device, epoch, batch_size, log_writer=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        x = x.to(device, non_blocking=True).to(torch.float32)
        mask = mask.to(device, non_blocking=True).to(torch.float32)
        

        t = torch.randint(0, 1000, (batch_size,), device=device).long()
        t_img = t.view(x.size(0), 1, 1, 1)
        e = torch.randn_like(x)
        m=0
        n=8
        scale=0.5
        
        loss_mask,loss_nomask,loss_vb = diffusion.forward(model, x, t, e, mask,control)
        loss = (loss_mask+scale*loss_nomask)/n

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss.backward()


        if (data_iter_step + 1) % 8 == 0:
            optimizer.step()      # 真正的权重更新
            optimizer.zero_grad()  # 清空梯度，准备下一轮累加
            
            control.update_ema()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_mask=loss_mask.item()/n)
        metric_logger.update(loss_nomask=scale*loss_nomask.item()/n)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % 2 == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

def test(diffusion,model,control,metrics_log,best_lpips,conf):
    device = dist_util.dev(conf.get('device'))
    checkpoint_path="/root/autodl-tmp/RePaint/data/pretrained/controlnet/last.pth"
    data_package = th.load(checkpoint_path, map_location="cpu")

    print(len(list(control.parameters())),len(data_package["ema_params1"]))
    for p, ema_p in zip(control.parameters(), data_package["ema_params1"]):
        p.data.copy_(ema_p.to(device))
        
    print("Successfully loaded EMA weights to model!")
    show_progress = conf.show_progress
    
    print("sampling...")
    total_psnr = 0.0
    total_lpips = 0.0
    total_ssim = 0.0
    processed_count = 0

    import lpips
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    loss_fn_vgg = lpips.LPIPS(net='alex').to(device)
    loss_fn_vgg.eval()

    all_images = []

    dset = 'eval'

    eval_name = conf.get_default_eval_name()

    dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    for batch in iter(dl):

        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}

        model_kwargs["gt"] = batch['GT']

        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )


        result = sample_fn(
            model,
            control,
            (batch_size, 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=None,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf
        )

        with th.no_grad():
            recon = result['sample'].detach().float()  # [-1,1]
            gt = result['gt'].detach().float()
            mask = model_kwargs.get('gt_keep_mask').detach().float()

            # repaint 中 1 = 保留区域，0 = 需要生成
            # 我们评估生成区域 → (1 - mask)
            eval_mask = 1 - mask

            mse_map = ((recon - gt) * eval_mask) ** 2
            mse_masked = mse_map.sum(dim=(1, 2, 3)) / (eval_mask.sum(dim=(1, 2, 3)) + 1e-8)

            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction=None).to(device)
            batch_ssim = ssim_metric((recon+1)/2, (gt+1)/2)
            total_ssim += batch_ssim.sum().item()

            # 你的公式 MAX_I = 2 （因为 [-1,1]）
            batch_psnr = 10 * th.log10(4.0 / (mse_masked + 1e-8))

            total_psnr += batch_psnr.sum().item()

            # ---- LPIPS 分块计算 ----
            micro_batch_size = 4
            num_samples = recon.size(0)

            for start in range(0, num_samples, micro_batch_size):
                end = min(start + micro_batch_size, num_samples)
                dist = loss_fn_vgg(recon[start:end], gt[start:end])
                total_lpips += dist.sum().item()

            processed_count += recon.size(0)

            print("Running LPIPS: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}".format(
                total_lpips / processed_count,
                total_psnr / processed_count,
                total_ssim / processed_count
            ))
    avg_lpips = total_lpips / processed_count
    avg_ssim = total_ssim / processed_count
    metrics_log["lpips"].append(avg_lpips)
    control.load_state_dict(data_package["model"])
    print("Restored original model weights!")

    if avg_lpips < best_lpips:
        best_lpips = avg_lpips
        print(f"New best LPIPS: {best_lpips:.4f}, saving model...")

        th.save({
            "model": control.state_dict(),
            "ema_params1": control.ema_params1,
        }, "data/pretrained/controlnet/best_model.pth")

    plot_metrics(metrics_log)
    return best_lpips,metrics_log


def plot_metrics(log):
    plt.figure()

    plt.plot(log["lpips"], label="LPIPS")

    plt.legend()
    plt.xlabel("Evaluation Step")
    plt.ylabel("Value")
    plt.title("Metrics Curve")

    plt.savefig("metrics.png")
    plt.close()

#python train_control.py --conf_path confs/test_c256_thin.yml