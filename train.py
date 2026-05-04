
import os
import copy
import argparse
import torch as th
import torch
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread
from torch.utils.tensorboard import SummaryWriter
from guided_diffusion import dist_util
from guided_diffusion.image_datasets import *
import util.misc as misc
from engine import *

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


    model, diffusion_test , diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )

    checkpoint_path = "/root/autodl-tmp/RePaint/data/pretrained/celeba256_250000.pt"
    ckpt = th.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(ckpt, strict=True)
    print("model参数加载完成")

    if "ema_params1" in ckpt:
        print("EMA参数加载完成")
        model.ema_params1 = [p.clone().to(device) for p in ckpt["ema_params1"]]
    else:
        print("没有 EMA，使用当前模型初始化")
        model.ema_params1 = [p.clone().to(device) for p in model.parameters()]

    model.to(device)
    model.train()

    # def model_fn(x, t, y=None, gt=None, **kwargs):
    #     return model(x, t, y if conf.class_cond else None, gt=gt)

    epochs=100
    batch_size=3
    save_last_freq=1
    img_size=256
    num_workers=16
    weight_decay=0
    lr=2e-5
    output_dir='./output_dir'
    os.makedirs(output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=output_dir)

    gt_path='/root/autodl-tmp/RePaint/data/data256x256'
    mask_path='/root/autodl-tmp/RePaint/data/masks/thin'
    dataloader_train = load_data_train(gt_path=gt_path,mask_path=mask_path,batch_size=batch_size,
                                        image_size=img_size,num_workers=num_workers)

    
    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))
    model.to(device)

    param_groups = misc.add_weight_decay(model, weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr/5, betas=(0.9, 0.95))
    # print(optimizer)
    print(optimizer)

    # Training loop
    print(f"Start training for {epochs} epochs")
    start_time = time.time()

    best_lpips = float("inf")
    metrics_log = {
    "lpips": []
    }
    for epoch in range(0, epochs):

        train_one_epoch(diffusion,model, dataloader_train, optimizer, device, epoch, batch_size=batch_size,log_writer=log_writer)
        save_path = "data/pretrained/last.pth"
        torch.save({
            "model": model.state_dict(),
            "ema_params1": model.ema_params1, # 手动把这个 list 存进去
        }, save_path)
        
        best_lpips,metrics_log=test(diffusion_test,model,metrics_log,best_lpips,conf)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)

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
