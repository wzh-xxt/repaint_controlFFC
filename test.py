
import os
import argparse
import torch as th
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def main(conf: conf_mgt.Default_Conf):

    print("Start", conf['name'])

    device = dist_util.dev(conf.get('device'))


    model, diffusion,_ = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )

    checkpoint_path="/root/autodl-tmp/RePaint/data/pretrained/best_model2.pth"
    data_package = th.load(checkpoint_path, map_location="cpu")
    ema_list = data_package["ema_params1"]
    new_state_dict = {}
    for i, (name, _value) in enumerate(model.named_parameters()):
        new_state_dict[name] = ema_list[i]
        
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(data_package)
    print("Successfully loaded EMA weights to model!")
      
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

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

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )


        result = sample_fn(
            model,
            (batch_size, 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
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

        srs = toU8(result['sample'])
        gts = toU8(result['gt'])
        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                   th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))

        gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

        conf.eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

    print("sampling complete")
    print("Final LPIPS: {:.4f}, Final PSNR: {:.4f}, Final SSIM: {:.4f}".format(
        total_lpips / processed_count,
        total_psnr / processed_count,
        total_ssim / processed_count
    ))

    import torch_fidelity
    from pytorch_fid import fid_score

    save_folder = "./log/test_c256_ev2li/inpainted"
    gt_folder = "./log/test_c256_ev2li/gt"

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=save_folder,  # 生成图目录
        input2=gt_folder,  # GT图目录
        cuda=True,
        fid=True,
        isc=True,
        kid=False,
        prc=False,
        verbose=False,
    )

    inception_score = metrics_dict['inception_score_mean']
    fid = metrics_dict['frechet_inception_distance']

    print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))


if __name__ == "__main__":

    import torch
    import numpy as np
    import random
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 在 main 开始时调用
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    main(conf_arg)
