
import enum

import numpy as np
import torch as th
import torch

from collections import defaultdict

from guided_diffusion.scheduler import get_schedule_jump
from .losses import normal_kl, discretized_gaussian_log_likelihood
from .nn import *

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, use_scale):
    if schedule_name == "linear":
        # diffusion steps.

        if use_scale:
            scale = 1000 / num_diffusion_timesteps
        else:
            scale = 1

        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """
    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        conf=None
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        self.conf = conf

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_prev_prev = np.append(
            1.0, self.alphas_cumprod_prev[:-1])

        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = np.sqrt(self.alphas_cumprod_prev)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(
            1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def undo(self, image_before_step, img_after_model, est_x_0, t, debug=False):
        return self._undo(img_after_model, t)

    def _undo(self, img_out, t):
        beta = _extract_into_tensor(self.betas, t, img_out.shape)

        img_in_est = th.sqrt(1 - beta) * img_out + \
            th.sqrt(beta) * th.randn_like(img_out)

        return img_in_est

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1,
                                 t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2,
                                   t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(
            self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model,control,mask,x_start, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        # print(t)
        x_masked = (x_start+1)/2 * mask
        input = torch.cat([x_masked, (1-mask[:, :1, :, :])], dim=1)
        # out=control(x,self._scale_timesteps(t))
        # model_output = model(x, self._scale_timesteps(t),out, **model_kwargs)
        out=control(input,t)
        model_output = model(x, t,out)

        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, model_var_values = th.split(model_output, C, dim=1)

        if self.model_var_type == ModelVarType.LEARNED:
            model_log_variance = model_var_values
            model_variance = th.exp(model_log_variance)
        else:
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)  #自己特定任务的需求函数
            if clip_denoised:
                return x.clamp(-1, 1)   #把t-1时刻预测的超出阈值的像素剪切，再作为输入，防止误差积累
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )   #通过预测出的噪声和xt后验出来x0
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )   #通过x0,xt计算出来t-1的均值
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(
                self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """

        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)


        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] *
            gradient.float()
        )
        return new_mean
        
    def _vb_terms_bpd(
        self, model,mask,control, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model,control,mask,x_start, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def forward(self, model_fn, x, t, noise, mask, control):

        B, C = x.shape[:2]
        assert t.shape == (B,)

        alpha_cumprod = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        
        gt_weight = th.sqrt(alpha_cumprod)
        gt_part = gt_weight * x

        noise_weight = th.sqrt((1 - alpha_cumprod))
        noise_part = noise_weight * noise

        x_t = gt_part + noise_part

        x_masked = (x+1)/2 * (1 - mask) 
        input = torch.cat([x_masked, mask[:, :1, :, :]], dim=1)
        out = control(input, t)

        model_output = model_fn(x_t, t, out)

        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, model_var_values = th.split(model_output, C, dim=1)
        frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)

        # loss_mse = th.nn.functional.mse_loss(model_output, noise)

        loss_mask = ((model_output - noise) ** 2 * mask).sum() / (mask.sum() + 1e-6)
        loss_nomask = ((model_output - noise) ** 2 * (1-mask)).sum() / ((1-mask).sum() + 1e-6)

        # loss_mse = loss_mask + 0.2*loss_nomask

        loss_vb = self._vb_terms_bpd(
            # 修改 lambda，确保无论输入什么，都返回预先计算好的 frozen_out
            model=lambda *args, **kwargs: frozen_out, 
            mask=mask,
            control=control,
            x_start=x,
            x_t=x_t,
            t=t,
            clip_denoised=False,
        )["output"].mean()

        return loss_mask,loss_nomask,loss_vb
    
    def p_sample(
        self,
        model,
        control,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        conf=None,
        meas_fn=None,
        pred_xstart=None,
        idx_wall=-1
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        noise = th.randn_like(x)
        gt_keep_mask = model_kwargs.get('gt_keep_mask')
        gt = model_kwargs['gt']

        if conf.inpa_inj_sched_prev:

            if pred_xstart is not None:
                gt_keep_mask = model_kwargs.get('gt_keep_mask')
                if gt_keep_mask is None:
                    gt_keep_mask = conf.get_inpa_mask(x)

                gt = model_kwargs['gt']

                alpha_cumprod = _extract_into_tensor(
                    self.alphas_cumprod, t, x.shape)

                if conf.inpa_inj_sched_prev_cumnoise:
                    weighed_gt = self.get_gt_noised(gt, int(t[0].item()))
                else:
                    gt_weight = th.sqrt(alpha_cumprod)
                    gt_part = gt_weight * gt

                    noise_weight = th.sqrt((1 - alpha_cumprod))
                    noise_part = noise_weight * th.randn_like(x)

                    weighed_gt = gt_part + noise_part

                x = (
                    gt_keep_mask * (
                        weighed_gt
                    )
                    +
                    (1 - gt_keep_mask) * (
                        x
                    )
                )


        out = self.p_mean_variance(
            model,
            control,
            gt_keep_mask,
            gt,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )   #当t=0时，不用加方差，直接出均值

        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        sample = out["mean"] + nonzero_mask * \
            th.exp(0.5 * out["log_variance"]) * noise

        result = {"sample": sample,
                  "pred_xstart": out["pred_xstart"], 'gt': model_kwargs.get('gt')}

        return result

    def p_sample_loop(
        self,
        model,
        control,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        return_all=False,
        conf=None
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            control,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            conf=conf
        ):
            final = sample

        if return_all:
            return final
        else:
            return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        control,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        conf=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image_after_step = noise
        else:
            image_after_step = th.randn(*shape, device=device)

        debug_steps = conf.pget('debug.num_timesteps')

        self.gt_noises = None  # reset for next image


        pred_xstart = None

        idx_wall = -1
        sample_idxs = defaultdict(lambda: 0)

        if conf.schedule_jump_params:
            times = get_schedule_jump(**conf.schedule_jump_params)

            time_pairs = list(zip(times[:-1], times[1:]))
            if progress:
                from tqdm.auto import tqdm
                time_pairs = tqdm(time_pairs)

            for t_last, t_cur in time_pairs:
                idx_wall += 1
                t_last_t = th.tensor([t_last] * shape[0],  # pylint: disable=not-callable
                                     device=device)

                if t_cur < t_last:  # reverse
                    with th.no_grad():
                        image_before_step = image_after_step.clone()
                        out = self.p_sample(
                            model,
                            control,
                            image_after_step,
                            t_last_t,
                            clip_denoised=clip_denoised,
                            denoised_fn=denoised_fn,
                            cond_fn=cond_fn,
                            model_kwargs=model_kwargs,
                            conf=conf,
                            pred_xstart=pred_xstart
                        )
                        image_after_step = out["sample"]
                        pred_xstart = out["pred_xstart"]

                        sample_idxs[t_cur] += 1

                        yield out

                else:
                    t_shift = conf.get('inpa_inj_time_shift', 1)

                    image_before_step = image_after_step.clone()
                    image_after_step = self.undo(
                        image_before_step, image_after_step,
                        est_x_0=out['pred_xstart'], t=t_last_t+t_shift, debug=False)
                    pred_xstart = out["pred_xstart"]
                    

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
