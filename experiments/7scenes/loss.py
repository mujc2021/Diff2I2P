import pdb

import torch
import torch.nn as nn

from vision3d.loss import CircleLoss
from vision3d.ops import apply_transform, pairwise_distance, random_choice
from vision3d.ops.metrics import compute_isotropic_transform_error

from diffusers import (
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
)
import torch.nn.functional as F
import numpy as np
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from PIL import Image
from torchvision import transforms
from copy import deepcopy




class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = CircleLoss(
            cfg.loss.coarse_loss.positive_margin,
            cfg.loss.coarse_loss.negative_margin,
            cfg.loss.coarse_loss.positive_optimal,
            cfg.loss.coarse_loss.negative_optimal,
            cfg.loss.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.loss.coarse_loss.positive_overlap
        self.negative_overlap = cfg.loss.coarse_loss.negative_overlap
        self.eps = 1e-8

        if cfg.loss.coarse_loss.enable_ms_loss:
            self.ms_loss = True
            self.weighted_circle_loss_latent_s4 = CircleLoss(
                cfg.loss.coarse_loss.positive_margin,
                cfg.loss.coarse_loss.negative_margin,
                cfg.loss.coarse_loss.positive_optimal,
                cfg.loss.coarse_loss.negative_optimal,
                cfg.loss.coarse_loss.log_scale,
            )

    def forward(self, output_dict):
        img_feats = output_dict["img_feats_c"]
        pcd_feats = output_dict["pcd_feats_c"]
        gt_img_node_corr_indices = output_dict["gt_img_node_corr_indices"]
        gt_pcd_node_corr_indices = output_dict["gt_pcd_node_corr_indices"]
        gt_node_corr_min_overlaps = output_dict["gt_node_corr_min_overlaps"]
        gt_node_corr_max_overlaps = output_dict["gt_node_corr_min_overlaps"]

        feat_dists = torch.sqrt(pairwise_distance(img_feats, pcd_feats, normalized=True) + self.eps)

        min_overlaps = torch.zeros_like(feat_dists)
        min_overlaps[gt_img_node_corr_indices, gt_pcd_node_corr_indices] = gt_node_corr_min_overlaps
        pos_masks = torch.gt(min_overlaps, self.positive_overlap)
        pos_scales = torch.sqrt(min_overlaps * pos_masks.float())

        max_overlaps = torch.zeros_like(feat_dists)
        max_overlaps[gt_img_node_corr_indices, gt_pcd_node_corr_indices] = gt_node_corr_max_overlaps
        neg_masks = torch.lt(max_overlaps, self.negative_overlap)

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        if self.ms_loss:
            img_feats_latent_s4 = output_dict["img_feats_latent_s4"]
            pcd_feats_latent_s4 = output_dict["pcd_feats_latent_s4"]
            gt_img_node_corr_indices_latent_s4 = output_dict["gt_img_node_corr_indices"]
            gt_pcd_node_corr_indices_latent_s4 = output_dict["gt_pcd_node_corr_indices"]
            gt_node_corr_min_overlaps_latent_s4 = output_dict["gt_node_corr_min_overlaps"]
            gt_node_corr_max_overlaps_latent_s4 = output_dict["gt_node_corr_min_overlaps"]

            feat_dists_latent_s4 = torch.sqrt(pairwise_distance(img_feats_latent_s4, pcd_feats_latent_s4, normalized=True) + self.eps)

            min_overlaps_latent_s4 = torch.zeros_like(feat_dists_latent_s4)
            min_overlaps_latent_s4[gt_img_node_corr_indices_latent_s4, gt_pcd_node_corr_indices_latent_s4] = gt_node_corr_min_overlaps_latent_s4
            pos_masks_latent_s4 = torch.gt(min_overlaps_latent_s4, self.positive_overlap)
            pos_scales_latent_s4 = torch.sqrt(min_overlaps_latent_s4 * pos_masks_latent_s4.float())

            max_overlaps_latent_s4 = torch.zeros_like(feat_dists_latent_s4)
            max_overlaps_latent_s4[gt_img_node_corr_indices_latent_s4, gt_pcd_node_corr_indices_latent_s4] = gt_node_corr_max_overlaps_latent_s4
            neg_masks_latent_s4 = torch.lt(max_overlaps_latent_s4, self.negative_overlap)

            loss_latent_s4 = self.weighted_circle_loss_latent_s4(pos_masks_latent_s4, neg_masks_latent_s4, feat_dists_latent_s4, pos_scales_latent_s4)

            loss += loss_latent_s4 / 2.

        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()

        self.max_correspondences = cfg.loss.fine_loss.max_correspondences
        self.pos_radius_3d = cfg.loss.fine_loss.positive_radius_3d
        self.neg_radius_3d = cfg.loss.fine_loss.negative_radius_3d
        self.pos_radius_2d = cfg.loss.fine_loss.positive_radius_2d
        self.neg_radius_2d = cfg.loss.fine_loss.negative_radius_2d

        self.circle_loss = CircleLoss(
            cfg.loss.fine_loss.positive_margin,
            cfg.loss.fine_loss.negative_margin,
            cfg.loss.fine_loss.positive_optimal,
            cfg.loss.fine_loss.negative_optimal,
            cfg.loss.fine_loss.log_scale,
        )

    @torch.no_grad()
    def get_recall(self, gt_corr_mat, fdist_mat):
        # Get feature match recall, divided by number of points which has inlier matches
        num_gt_corr = torch.gt(gt_corr_mat.sum(-1), 0).float().sum() + 1e-12
        src_indices = torch.arange(fdist_mat.shape[0]).cuda()
        src_nn_indices = fdist_mat.min(-1)[1]
        pred_corr_mat = torch.zeros_like(fdist_mat)
        pred_corr_mat[src_indices, src_nn_indices] = 1.0
        recall = (pred_corr_mat * gt_corr_mat).sum() / num_gt_corr
        return recall

    def forward(self, data_dict, output_dict):
        assert data_dict["batch_size"] == 1, "Only support the batch_size of 1."

        # 1. unpack data
        img_points = output_dict["img_points_f"]  # (HxW, 3)
        img_feats = output_dict["img_feats_f"]  # (HxW, C)
        # img_corr_points = output_dict["img_corr_points"]  # (N, 3)

        pcd_points = output_dict["pcd_points_f"]  # (N, 3)
        pcd_pixels = output_dict["pcd_pixels_f"]  # (N, 3)
        # pcd_corr_pixels = output_dict["pcd_corr_pixels"]  # (N, 2)
        pcd_feats = output_dict["pcd_feats_f"]  # (N, C)

        transform = data_dict["transform"]  # (4, 4)
        img_corr_pixels = data_dict["img_corr_pixels"]
        pcd_corr_indices = data_dict["pcd_corr_indices"]

        image_w = data_dict["image_w"]

        pcd_points = apply_transform(pcd_points, transform)  # (N, 3)

        # 2. sample correspondences
        if pcd_corr_indices.shape[0] > self.max_correspondences:
            sel_indices = random_choice(pcd_corr_indices.shape[0], size=self.max_correspondences, replace=False)
            img_sel_pixels = img_corr_pixels[sel_indices]
            pcd_sel_indices = pcd_corr_indices[sel_indices]
        else:
            img_sel_pixels = img_corr_pixels
            pcd_sel_indices = pcd_corr_indices
            sel_indices = torch.arange(pcd_corr_indices.shape[0]).to(pcd_corr_indices.device)

        img_sel_v_coords = img_sel_pixels[:, 0]
        img_sel_u_coords = img_sel_pixels[:, 1]
        img_sel_indices = img_sel_v_coords * image_w + img_sel_u_coords
        img_sel_points = img_points[img_sel_indices]  # (M, 3)
        # img_sel_points = img_corr_points[sel_indices]  # (M, 3)
        img_sel_pixels = img_sel_pixels.float()
        img_sel_feats = img_feats[img_sel_indices]  # (M, 3)

        pcd_sel_points = pcd_points[pcd_sel_indices]  # (M, C)
        pcd_sel_pixels = pcd_pixels[pcd_sel_indices]  # (M, C)
        # pcd_sel_pixels = pcd_corr_pixels[sel_indices]  # (M, C)
        pcd_sel_feats = pcd_feats[pcd_sel_indices]  # (M, C)

        dist3d_mat = pairwise_distance(img_sel_points, pcd_sel_points, squared=False, strict=True)
        dist2d_mat = pairwise_distance(img_sel_pixels, pcd_sel_pixels, squared=False, strict=True)
        pos_masks = torch.logical_and(
            torch.lt(dist3d_mat, self.pos_radius_3d),
            torch.lt(dist2d_mat, self.pos_radius_2d),
        )
        neg_masks = torch.logical_or(
            torch.gt(dist3d_mat, self.neg_radius_3d),
            torch.gt(dist2d_mat, self.neg_radius_2d),
        )
        fdist_mat = pairwise_distance(img_sel_feats, pcd_sel_feats, normalized=False)

        # 3. circle loss
        loss = self.circle_loss(pos_masks, neg_masks, fdist_mat)

        # 5. matching recall
        recall = self.get_recall(pos_masks.float(), fdist_mat)

        return loss, recall



class SdsLoss(nn.Module):
    def __init__(self, cfg, t_range=[0.02, 0.98],):
        super(SdsLoss, self).__init__()
        self.dtype = torch.float32
        self.device = 'cuda'

        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth",
                                                     torch_dtype=self.dtype)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=self.dtype
        )

        self.control_image_processor = pipe.control_image_processor

        pipe.to('cuda')

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.controlnet = pipe.controlnet

        self.scheduler = DDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="scheduler", torch_dtype=self.dtype
        )
        del pipe
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
        self.embeddings = {}

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds

    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):

        # image = torch.nn.functional.interpolate(image, size=(height, width), mode='bilinear', align_corners=False)
        # image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        # image_batch_size = image.shape[0]

        # if image_batch_size == 1:
        #     repeat_by = batch_size
        # else:
        #     # image batch size is the same as prompt batch size
        #     repeat_by = num_images_per_prompt
        # image = image.repeat_interleave(repeat_by, dim=0)

        # image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def forward(
        self,
        pred_rgb,
        depth,
        guidance_scale=4,  # 100
        guess_mode=True,
        do_classifier_free_guidance=True,
        mask=None,
    ):

        with torch.no_grad():
            self.get_text_embeds("best quality, a photo of a room,furniture, household items",
                                 "low-resolusion, bad anatomy, bad hands, cropped, worst quality")
            batch_size = pred_rgb.shape[0]
            pred_rgb = pred_rgb.to(self.dtype)

            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

            # predict the noise residual with unet, NO grad!
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # tt = torch.cat([t] * 2)

            prompt_embeds = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1),
                                    self.embeddings['neg'].expand(batch_size, -1, -1)])

            # image = self.prepare_image(
            #     image=depth,
            #     width=512,
            #     height=512,
            #     batch_size=batch_size,
            #     device=self.device,
            #     dtype=self.dtype,
            #     num_images_per_prompt=1,
            #     do_classifier_free_guidance=do_classifier_free_guidance,
            #     guess_mode=guess_mode,
            # )
            image = depth[:, :3, :, :]


            if guess_mode and do_classifier_free_guidance:
                # Infer ControlNet only for the conditional batch.
                control_model_input = latents
                control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
            else:
                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                control_model_input,
                t,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=image,
                # conditioning_scale=cond_scale,
                guess_mode=guess_mode,
                return_dict=False,
            )

            if guess_mode and do_classifier_free_guidance:
                # Infered ControlNet only for the conditional batch.
                # To apply the output of ControlNet to both the unconditional and conditional batches,
                # add 0 to the unconditional batch to keep it unchanged.
                down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]

            # noise_pred = self.unet(
            #     latent_model_input, tt, encoder_hidden_states=embeddings
            # ).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            grad = w * (noise_pred - noise)
            if mask is not None:
                grad = grad * mask
            grad = torch.nan_to_num(grad)


        # downscale_depth.retain_grad()
        # loss = 0
        # for i in range(4):
        #     loss = loss + torch.diagonal(downscale_depth[0, i, ...]).sum()
        # loss.backward()

        depth = F.interpolate(depth, (64, 64), mode="bilinear", align_corners=False)
        # depth = self.controlnet.controlnet_cond_embedding(depth)
        target = (depth - grad).detach()
        loss = 0.5 * F.mse_loss(depth, target, reduction='sum') / latents.shape[0]

        return loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.c_loss = CoarseMatchingLoss(cfg)
        self.f_loss = FineMatchingLoss(cfg)
        self.weight_c_loss = cfg.loss.coarse_loss.weight
        self.weight_f_loss = cfg.loss.fine_loss.weight
        self.weight_sds_loss = cfg.loss.sds_loss.weight

        self.enable_diff_loss = cfg.model.diff_pnp_solver is not None

        self.deform_corr_point_l2_loss_weight = cfg.model.deformable_correspondences.point_l2_norm_weight
        self.deform_corr_point_residual_loss_weight = cfg.model.deformable_correspondences.point_residual_weight
        self.deform_corr_pixel_l2_loss_weight = cfg.model.deformable_correspondences.pixel_l2_norm_weight
        self.deform_corr_pixel_residual_loss_weight = cfg.model.deformable_correspondences.pixel_residual_weight

        self.translation_loss_weight = cfg.model.diff_pnp_solver.translation_loss_weight
        self.rotation_loss_weight = cfg.model.diff_pnp_solver.rotation_loss_weight

        self.sds_loss = SdsLoss(cfg)

    def forward(self, data_dict, output_dict):
        c_loss = self.c_loss(output_dict)
        f_loss, f_recall = self.f_loss(data_dict, output_dict)

        c_loss = c_loss * self.weight_c_loss
        f_loss = f_loss * self.weight_f_loss




        # if "diff_loss" in output_dict:
        #     loss = c_loss + f_loss + output_dict["diff_loss"]
        # else:
        #     if self.enable_diff_loss:
        #         print("No diff loss")
        #     loss = c_loss + f_loss
        loss = c_loss + f_loss

        # sds_loss = 0
        if "depth" in output_dict:
            image_path = f'../../data/7Scenes/data/{data_dict["image_file"]}'
            image = Image.open(image_path).convert('RGB')  # 确保是RGB格式
            # 应用转换
            transform = transforms.Compose([
                transforms.ToTensor(),  # 转换为张量，像素值会在[0, 1]之间
            ])
            image_tensor = transform(image)  # 变成 (C, H, W)
            image = image_tensor.unsqueeze(0).cuda()  # 变成 (1, C, H, W)

            depth = output_dict["depth"]

            depth = depth.unsqueeze(0).repeat(1, 4, 1, 1)

            depth = transforms.functional.resize(depth, (512, 512))

            gt_depth = output_dict['gt_depth']
            gt_depth = gt_depth.unsqueeze(0).unsqueeze(0)
            mask = transforms.functional.resize(gt_depth, (64, 64)) > 0

            sds_loss = self.sds_loss(image, depth, mask=mask)
            sds_loss = sds_loss * self.weight_sds_loss
            loss += sds_loss



        if "loss_coord" in output_dict and self.deform_corr_point_l2_loss_weight > 0.:
            loss += output_dict["loss_coord"] * self.deform_corr_point_l2_loss_weight

        if "loss_coord_residual" in output_dict:
            loss += output_dict["loss_coord_residual"] * self.deform_corr_point_residual_loss_weight

        if "loss_pixel" in output_dict and self.deform_corr_pixel_l2_loss_weight > 0.:
            loss += output_dict["loss_pixel"] * self.deform_corr_pixel_l2_loss_weight

        if "loss_pixel_residual" in output_dict:
            loss += output_dict["loss_pixel_residual"] * self.deform_corr_pixel_residual_loss_weight

        if "translation_loss" in output_dict and self.translation_loss_weight > 0.:
            loss += output_dict["translation_loss"] * self.translation_loss_weight

        if "rotation_loss" in output_dict and self.rotation_loss_weight > 0.:
            loss += output_dict["rotation_loss"] * self.rotation_loss_weight

        # return {"loss": loss, "c_loss": c_loss, "f_loss": f_loss, "sds_loss": sds_loss, "f_recall": f_recall}
        return {"loss": loss, "c_loss": c_loss, "f_loss": f_loss, "f_recall": f_recall}

class EvalFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold

    @torch.no_grad()
    def evaluate_coarse_matching(self, output_dict):
        img_length_c = output_dict["img_num_nodes"]
        pcd_length_c = output_dict["pcd_num_nodes"]
        gt_node_corr_min_overlaps = output_dict["gt_node_corr_min_overlaps"]
        gt_img_node_corr_indices = output_dict["gt_img_node_corr_indices"]
        gt_pcd_node_corr_indices = output_dict["gt_pcd_node_corr_indices"]
        img_node_corr_indices = output_dict["img_node_corr_indices"]
        pcd_node_corr_indices = output_dict["pcd_node_corr_indices"]

        masks = torch.gt(gt_node_corr_min_overlaps, self.acceptance_overlap)
        gt_img_node_corr_indices = gt_img_node_corr_indices[masks]
        gt_pcd_node_corr_indices = gt_pcd_node_corr_indices[masks]
        gt_node_corr_mat = torch.zeros(img_length_c, pcd_length_c).cuda()
        gt_node_corr_mat[gt_img_node_corr_indices, gt_pcd_node_corr_indices] = 1.0

        precision = gt_node_corr_mat[img_node_corr_indices, pcd_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine_matching(self, data_dict, output_dict):
        transform = data_dict["transform"]
        img_corr_points = output_dict["img_corr_points"]
        pcd_corr_points = output_dict["pcd_corr_points"]
        # only evaluate the correspondences with depth
        corr_masks = torch.gt(img_corr_points[..., -1], 0.0)
        img_corr_points = img_corr_points[corr_masks]
        pcd_corr_points = pcd_corr_points[corr_masks]
        pcd_corr_points = apply_transform(pcd_corr_points, transform)
        corr_distances = torch.linalg.norm(pcd_corr_points - img_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean().nan_to_num_()
        return precision

    @torch.no_grad()
    def evaluate_registration(self, data_dict, output_dict):
        transform = data_dict["transform"]
        est_transform = output_dict["estimated_transform"]
        pcd_points = output_dict["pcd_points"]

        rre, rte = compute_isotropic_transform_error(transform, est_transform)

        realignment_transform = torch.matmul(torch.linalg.inv(transform), est_transform)
        realigned_pcd_points_f = apply_transform(pcd_points, realignment_transform)
        rmse = torch.linalg.norm(realigned_pcd_points_f - pcd_points, dim=1).mean()
        recall = torch.lt(rmse, self.acceptance_rmse).float()

        return rre, rte, rmse, recall

    def forward(self, data_dict, output_dict):
        c_precision = self.evaluate_coarse_matching(output_dict)
        f_precision = self.evaluate_fine_matching(data_dict, output_dict)

        return {"PIR": c_precision, "IR": f_precision}
