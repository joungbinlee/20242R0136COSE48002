#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from time import time 
    
    
def render_from_batch(viewpoint_cameras, pc : GaussianModel, pipe, random_color= False, scaling_modifier = 1.0, stage="fine", batch_size=1, visualize_attention=False, only_infer = False, canonical_tri_plane_factor_list = None, iteration=None):
    if only_infer:
        time1 = time()
        batch_size = len(viewpoint_cameras)
    means3D = pc.get_xyz.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 3]
    opacity = pc._opacity.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 1]
    shs = pc.get_features.unsqueeze(0).repeat(batch_size, 1, 1, 1) # [B, N, 16, 3]
    scales = pc._scaling.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 3]
    rotations = pc._rotation.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 4] 
    attention = None
    colors_precomp = None
    cov3D_precomp = None
     
    aud_features = []
    eye_features = []
    rasterizers = []
    gt_imgs = []
    viewspace_point_tensor_list = []
    means2Ds = []
    lips_list = []
    bg_w_torso_list = []
    gt_masks = []
    gt_w_bg = []
    cam_features = []
    
    for viewpoint_camera in viewpoint_cameras:
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2Ds.append(screenspace_points)
        viewspace_point_tensor_list.append(screenspace_points)
        
        if random_color:
            background = torch.rand((3,), dtype=torch.float32, device="cuda")
            bg_image = background[:, None, None] * torch.ones((1, viewpoint_camera.image_height, viewpoint_camera.image_width), device=background.device)
        elif only_infer:
            bg_image = viewpoint_camera.bg_w_torso.to('cuda')
        else: 
            white_or_black = torch.randint(2, (1,)).item()
            background = torch.full((3,), white_or_black, dtype=torch.float32, device="cuda")
            bg_image = background[:, None, None] * torch.ones((1, viewpoint_camera.image_height, viewpoint_camera.image_width), device=background.device)
        
        aud_features.append(viewpoint_camera.aud_f.unsqueeze(0).to(means3D.device))
        eye_features.append(torch.from_numpy(np.array([viewpoint_camera.eye_f])).unsqueeze(0).to(means3D.device))
        cam_features.append(torch.from_numpy(np.concatenate((viewpoint_camera.R.reshape(-1), viewpoint_camera.T.reshape(-1))).reshape(1,-1)).to(means3D.device))
        bg_w_torso_list.append(viewpoint_camera.bg_w_torso.cpu())
        
        bg_image = viewpoint_camera.bg_w_torso.to('cuda')
        
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
            
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx= tanfovx,
            tanfovy= tanfovy,
            bg=bg_image,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        rasterizers.append(GaussianRasterizer(raster_settings=raster_settings))

        bg_mask = viewpoint_camera.head_mask 
        bg_mask = torch.from_numpy(bg_mask).to("cuda")
        gt_image = viewpoint_camera.original_image.cuda()
        gt_w_bg.append(gt_image.unsqueeze(0))
        gt_image = gt_image * bg_mask + bg_image * (~ bg_mask)
        gt_imgs.append(gt_image.unsqueeze(0).cuda())
        lips_list.append(viewpoint_camera.lips_rect)
        bg_mask = bg_mask.to(torch.float).unsqueeze(0).unsqueeze(0)
        gt_masks.append(bg_mask)
    
    if stage == "coarse":
        aud_features, eye_features, cam_features = None, None, None 
        means3D_final, scales_temp, rotations_temp, opacity_temp, shs_temp = pc._deformation(means3D, scales, rotations, opacity, shs, aud_features, eye_features, cam_features)
        if "scales" in canonical_tri_plane_factor_list:
            scales_temp = scales_temp-2
            scales_final = scales_temp
        else: 
            scales_final = scales
            scales_temp = None
        if "rotations" in canonical_tri_plane_factor_list:
            rotations_final = rotations_temp
        else: 
            rotations_final = rotations
            rotations_temp = None
        if "opacity" in canonical_tri_plane_factor_list:
            opacity_final = opacity_temp
        else:
            opacity_final = opacity
            opacity_temp = None
        if "shs" in canonical_tri_plane_factor_list:
            shs_final = shs_temp
        else:
            shs_final = shs
            shs_temp = None
            
        pc.replace_gaussian(scales_temp, rotations_temp, opacity_temp, shs_temp)

        scales_final = pc.scaling_activation(scales_final)
        rotations_final = torch.nn.functional.normalize(rotations_final,dim=2) 
        opacity_final = pc.opacity_activation(opacity_final)

    elif stage == "fine":
        aud_features = torch.cat(aud_features,dim=0)
        eye_features = torch.cat(eye_features,dim=0)
        cam_features = torch.cat(cam_features,dim=0)

        means3D_final, scales_final, rotations_final, opacity_final, shs_final, attention = pc._deformation(means3D, scales, rotations, opacity, shs, aud_features, eye_features,cam_features)
                                                                                                    
        scales_final = pc.scaling_activation(scales_final)
        rotations_final = torch.nn.functional.normalize(rotations_final,dim=2) 
        opacity_final = pc.opacity_activation(opacity_final)
        
    
    rendered_image_list = []
    radii_list = []
    depth_list = []
    visibility_filter_list = []
    audio_image_list = []
    eye_image_list = []
    cam_image_list = []
    null_image_list = []
    rendered_lips = []
    gt_lips = []
    
    for idx, rasterizer in enumerate(rasterizers):
        colors_precomp = None
        rendered_image, radii, depth = rasterizer(
            means3D = means3D_final[idx],
            means2D = means2Ds[idx],
            shs = shs_final[idx],
            colors_precomp = colors_precomp,
            opacities = opacity_final[idx],
            scales = scales_final[idx],
            rotations = rotations_final[idx],
            cov3D_precomp = cov3D_precomp,)

        rendered_image_list.append(rendered_image.unsqueeze(0))
        radii_list.append(radii.unsqueeze(0))
        depth_list.append(depth.unsqueeze(0))
        visibility_filter_list.append((radii > 0).unsqueeze(0))
        
        if not only_infer:
            y1,y2,x1,x2 = lips_list[idx]
            lip_crop = rendered_image[:,y1:y2,x1:x2]
            gt_lip_crop = gt_imgs[idx][:,:,y1:y2,x1:x2]
            rendered_lips.append(lip_crop.flatten())
            gt_lips.append(gt_lip_crop.flatten())
            
        audio_image, eye_image ,cam_image, null_image = None, None, None, None
        if visualize_attention:
            colors_precomp = attention.mean(dim=1)[idx,:,0].unsqueeze(1).repeat(1, 3)
            min_val = colors_precomp.min()
            max_val = colors_precomp.max()
            colors_precomp = (colors_precomp - min_val) / (max_val - min_val)

            audio_image, radii, depth = rasterizer(
                means3D = means3D_final[idx],
                means2D = means2Ds[idx],
                shs = None,
                colors_precomp = colors_precomp,
                opacities = opacity_final[idx], 
                scales = scales_final[idx],
                rotations = rotations_final[idx],
                cov3D_precomp = cov3D_precomp,)
                    
            colors_precomp = attention.mean(dim=1)[idx,:,1].unsqueeze(1).repeat(1, 3)
            min_val = colors_precomp.min()
            max_val = colors_precomp.max()
            colors_precomp = (colors_precomp - min_val) / (max_val - min_val)
            
            eye_image, radii, depth = rasterizer(
                means3D = means3D_final[idx],
                means2D = means2Ds[idx],
                shs = None,
                colors_precomp = colors_precomp,
                opacities = opacity_final[idx], 
                scales = scales_final[idx],
                rotations = rotations_final[idx],
                cov3D_precomp = cov3D_precomp,)
            
            colors_precomp = attention.mean(dim=1)[idx,:,2].unsqueeze(1).repeat(1, 3)
            min_val = colors_precomp.min()
            max_val = colors_precomp.max()
            colors_precomp = (colors_precomp - min_val) / (max_val - min_val)
            
            cam_image, radii, depth = rasterizer(
                means3D = means3D_final[idx],
                means2D = means2Ds[idx],
                shs = None,
                colors_precomp = colors_precomp,
                opacities = opacity_final[idx], 
                scales = scales_final[idx],
                rotations = rotations_final[idx],
                cov3D_precomp = cov3D_precomp,)
            
            colors_precomp = attention.mean(dim=1)[idx,:,3].unsqueeze(1).repeat(1, 3)
            min_val = colors_precomp.min()
            max_val = colors_precomp.max()
            colors_precomp = (colors_precomp - min_val) / (max_val - min_val)
            
            null_image, radii, depth = rasterizer(
                means3D = means3D_final[idx],
                means2D = means2Ds[idx],
                shs = None,
                colors_precomp = colors_precomp,
                opacities = opacity_final[idx], 
                scales = scales_final[idx],
                rotations = rotations_final[idx],
                cov3D_precomp = cov3D_precomp,)

            audio_image_list.append(audio_image.unsqueeze(dim=0))
            eye_image_list.append(eye_image.unsqueeze(dim=0))
            cam_image_list.append(cam_image.unsqueeze(dim=0))
            null_image_list.append(null_image.unsqueeze(dim=0))
        
    radii = torch.cat(radii_list,0).max(dim=0).values
    visibility_filter_tensor = torch.cat(visibility_filter_list).any(dim=0)
    rendered_image_tensor = torch.cat(rendered_image_list,0)
    gt_tensor = torch.cat(gt_imgs,0)
    depth_tensor = torch.cat(depth_list,dim=0)
    gt_masks_tensor = torch.cat(gt_masks,dim=0)
    gt_w_bg_tensor = torch.cat(gt_w_bg,dim=0)
    
    audio_image_tensor, eye_image_tensor, null_image_tensor, cam_image_tensor = None, None, None, None
    if visualize_attention:
        audio_image_tensor = torch.cat(audio_image_list,0)
        eye_image_tensor = torch.cat(eye_image_list,0)
        cam_image_tensor = torch.cat(cam_image_list, 0)
        null_image_tensor = torch.cat(null_image_list,0)
    
    rendered_lips_tensor ,gt_lips_tensor, rendered_w_bg_tensor = None, None, None
    inference_time = None
    
    if not only_infer:
        rendered_lips_tensor = torch.cat(rendered_lips,0)
        gt_lips_tensor = torch.cat(gt_lips,0)
    if only_infer:
        inference_time = time()-time1
        
        
    return {"rendered_image_tensor": rendered_image_tensor,
        "gt_tensor":gt_tensor,
        "viewspace_points": screenspace_points,
        "visibility_filter_tensor" : visibility_filter_tensor,
        "viewspace_point_tensor_list" : viewspace_point_tensor_list,
        "radii": radii,
        "depth_tensor": depth_tensor,
        "audio_attention": audio_image_tensor,
        "eye_attention": eye_image_tensor,
        "cam_attention" : cam_image_tensor,
        "null_attention": null_image_tensor,
        "rendered_lips_tensor":rendered_lips_tensor,
        "gt_lips_tensor":gt_lips_tensor,
        "rendered_w_bg_tensor":rendered_w_bg_tensor,
        "inference_time":inference_time,
        "gt_masks_tensor":gt_masks_tensor,
        "gt_w_bg_tensor":gt_w_bg_tensor,
        }


def vae_render_from_batch(viewpoint_cameras, pc : GaussianModel, pipe, vae, random_color= False, scaling_modifier = 1.0, stage="fine", batch_size=1, visualize_attention=False, only_infer = False, canonical_tri_plane_factor_list = None, iteration=None):
    if only_infer:
        time1 = time()
        batch_size = len(viewpoint_cameras)
    means3D = pc.get_xyz.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 3]
    opacity = pc._opacity.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 1]
    shs = pc.get_features.unsqueeze(0).repeat(batch_size, 1, 1, 1) # [B, N, 16, 3]
    scales = pc._scaling.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 3]
    rotations = pc._rotation.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 4] 
    attention = None
    colors_precomp = None
    cov3D_precomp = None
     
    aud_features = []
    eye_features = []
    rasterizers = []
    gt_imgs = []
    viewspace_point_tensor_list = []
    means2Ds = []
    lips_list = []
    bg_w_torso_list = []
    gt_masks = []
    gt_w_bg = []
    cam_features = []
    projmatrixs = []
    
    for viewpoint_camera in viewpoint_cameras:
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2Ds.append(screenspace_points)
        viewspace_point_tensor_list.append(screenspace_points)
        
        if random_color:
            background = torch.rand((3,), dtype=torch.float32, device="cuda")
            bg_image = background[:, None, None] * torch.ones((1, viewpoint_camera.image_height, viewpoint_camera.image_width), device=background.device)
        elif only_infer:
            bg_image = viewpoint_camera.bg_w_torso.to('cuda')
        else: 
            white_or_black = torch.randint(2, (1,)).item()
            background = torch.full((3,), white_or_black, dtype=torch.float32, device="cuda")
            bg_image = background[:, None, None] * torch.ones((1, viewpoint_camera.image_height, viewpoint_camera.image_width), device=background.device)
        
        aud_features.append(viewpoint_camera.aud_f.unsqueeze(0).to(means3D.device))
        eye_features.append(torch.from_numpy(np.array([viewpoint_camera.eye_f])).unsqueeze(0).to(means3D.device))
        cam_features.append(torch.from_numpy(np.concatenate((viewpoint_camera.R.reshape(-1), viewpoint_camera.T.reshape(-1))).reshape(1,-1)).to(means3D.device))
        bg_w_torso_list.append(viewpoint_camera.bg_w_torso.cpu())
        
        bg_image = viewpoint_camera.bg_w_torso.to('cuda')
        
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        
            
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx= tanfovx,
            tanfovy= tanfovy,
            bg=bg_image,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        rasterizers.append(GaussianRasterizer(raster_settings=raster_settings))

        bg_mask = viewpoint_camera.head_mask 
        bg_mask = torch.from_numpy(bg_mask).to("cuda")
        gt_image = viewpoint_camera.original_image.cuda()
        gt_w_bg.append(gt_image.unsqueeze(0))
        gt_image = gt_image * bg_mask + bg_image * (~ bg_mask)
        gt_imgs.append(gt_image.unsqueeze(0).cuda())
        lips_list.append(viewpoint_camera.lips_rect)
        bg_mask = bg_mask.to(torch.float).unsqueeze(0).unsqueeze(0)
        gt_masks.append(bg_mask)
        projmatrixs.append(viewpoint_camera.full_proj_transform.unsqueeze(dim=0))
        
    gt_tensor = torch.cat(gt_imgs,0)
    # posterior = vae.module.encode(gt_tensor*2-1).latent_dist
    posterior = vae.encode(gt_tensor*2-1).latent_dist
    z = posterior.sample()
    projmatrixs = torch.cat(projmatrixs,0).to(device=means3D.device)
    
    ones = torch.ones(means3D.shape[0], means3D.shape[1], 1, device=means3D.device)  # Shape: [4, 34650, 1]
    means3D_h = torch.cat([means3D, ones], dim=-1)  # Shape: [4, 34650, 4]
    projmatrixs = projmatrixs.unsqueeze(1)  # Shape: [4, 1, 4, 4]
    transformed_h = torch.matmul(means3D_h.unsqueeze(-2), projmatrixs).squeeze(-2)  # Shape: [4, 34650, 4]
    ndc_coordinates = transformed_h[..., :3] / transformed_h[..., 3:4]  # Shape: [4, 34650, 3]
    ndc_coordinates[:,:,1:] = -ndc_coordinates[:,:,1:]
    # ndc_coordinates = ndc_coordinates.cuda()
    # visualize_ndc_video(ndc_coordinates.detach(), batch_idx=0, point_sample=10000)
    # visualize_ndc_with_gt(ndc_coordinates.detach(), gt_tensor.detach(), batch_idx=1, point_sample=10000)

    if stage == "coarse":
        aud_features, eye_features, cam_features = None, None, None 
        _, mu_temp, scales_temp, rotations_temp, opacity_temp, shs_temp = pc._deformation(
            z, ndc_coordinates, scales, rotations, opacity, shs, aud_features, eye_features, cam_features)
        if "mu" in canonical_tri_plane_factor_list:
            means3D_final = means3D + mu_temp
        else: 
            means3D_final = means3D
            mu_temp=None
        if "scales" in canonical_tri_plane_factor_list:
            scales_temp = scales_temp-2
            scales_final = scales_temp
        else: 
            scales_final = scales
            scales_temp = None
        if "rotations" in canonical_tri_plane_factor_list:
            rotations_final = rotations_temp
        else: 
            rotations_final = rotations
            rotations_temp = None
        if "opacity" in canonical_tri_plane_factor_list:
            opacity_final = opacity_temp
        else:
            opacity_final = opacity
            opacity_temp = None
        if "shs" in canonical_tri_plane_factor_list:
            shs_final = shs_temp
        else:
            shs_final = shs
            shs_temp = None
            
        # pc.replace_gaussian(scales_temp, rotations_temp, opacity_temp, shs_temp)

        scales_final = pc.scaling_activation(scales_final)
        rotations_final = torch.nn.functional.normalize(rotations_final,dim=2) 
        opacity_final = pc.opacity_activation(opacity_final)

    elif stage == "fine":
        aud_features = torch.cat(aud_features,dim=0)
        eye_features = torch.cat(eye_features,dim=0)
        cam_features = torch.cat(cam_features,dim=0)

        means3D_final, scales_final, rotations_final, opacity_final, shs_final, attention = pc._deformation(
            z, ndc_coordinates, scales, rotations, opacity, shs, aud_features, eye_features, cam_features)
                                                                                                    
        scales_final = pc.scaling_activation(scales_final)
        rotations_final = torch.nn.functional.normalize(rotations_final,dim=2) 
        opacity_final = pc.opacity_activation(opacity_final)
        
    
    rendered_image_list = []
    radii_list = []
    depth_list = []
    visibility_filter_list = []
    audio_image_list = []
    eye_image_list = []
    cam_image_list = []
    null_image_list = []
    rendered_lips = []
    gt_lips = []
    
    for idx, rasterizer in enumerate(rasterizers):
        colors_precomp = None
        rendered_image, radii, depth = rasterizer(
            means3D = means3D_final[idx],
            means2D = means2Ds[idx],
            shs = shs_final[idx],
            colors_precomp = colors_precomp,
            opacities = opacity_final[idx],
            scales = scales_final[idx],
            rotations = rotations_final[idx],
            cov3D_precomp = cov3D_precomp,)

        rendered_image_list.append(rendered_image.unsqueeze(0))
        radii_list.append(radii.unsqueeze(0))
        depth_list.append(depth.unsqueeze(0))
        visibility_filter_list.append((radii > 0).unsqueeze(0))
        
        if not only_infer:
            y1,y2,x1,x2 = lips_list[idx]
            lip_crop = rendered_image[:,y1:y2,x1:x2]
            gt_lip_crop = gt_imgs[idx][:,:,y1:y2,x1:x2]
            rendered_lips.append(lip_crop.flatten())
            gt_lips.append(gt_lip_crop.flatten())

        audio_image, eye_image ,cam_image, null_image = None, None, None, None
        if visualize_attention:
            colors_precomp = attention.mean(dim=1)[idx,:,0].unsqueeze(1).repeat(1, 3)
            min_val = colors_precomp.min()
            max_val = colors_precomp.max()
            colors_precomp = (colors_precomp - min_val) / (max_val - min_val)
            
            audio_image, radii, depth = rasterizer(
                means3D = means3D_final[idx],
                means2D = means2Ds[idx],
                shs = None,
                colors_precomp = colors_precomp,
                opacities = opacity_final[idx], 
                scales = scales_final[idx],
                rotations = rotations_final[idx],
                cov3D_precomp = cov3D_precomp,)
                    
            colors_precomp = attention.mean(dim=1)[idx,:,1].unsqueeze(1).repeat(1, 3)
            min_val = colors_precomp.min()
            max_val = colors_precomp.max()
            colors_precomp = (colors_precomp - min_val) / (max_val - min_val)
            
            eye_image, radii, depth = rasterizer(
                means3D = means3D_final[idx],
                means2D = means2Ds[idx],
                shs = None,
                colors_precomp = colors_precomp,
                opacities = opacity_final[idx], 
                scales = scales_final[idx],
                rotations = rotations_final[idx],
                cov3D_precomp = cov3D_precomp,)
            
            colors_precomp = attention.mean(dim=1)[idx,:,2].unsqueeze(1).repeat(1, 3)
            min_val = colors_precomp.min()
            max_val = colors_precomp.max()
            colors_precomp = (colors_precomp - min_val) / (max_val - min_val)
            
            cam_image, radii, depth = rasterizer(
                means3D = means3D_final[idx],
                means2D = means2Ds[idx],
                shs = None,
                colors_precomp = colors_precomp,
                opacities = opacity_final[idx], 
                scales = scales_final[idx],
                rotations = rotations_final[idx],
                cov3D_precomp = cov3D_precomp,)
            
            colors_precomp = attention.mean(dim=1)[idx,:,3].unsqueeze(1).repeat(1, 3)
            min_val = colors_precomp.min()
            max_val = colors_precomp.max()
            colors_precomp = (colors_precomp - min_val) / (max_val - min_val)
            
            null_image, radii, depth = rasterizer(
                means3D = means3D_final[idx],
                means2D = means2Ds[idx],
                shs = None,
                colors_precomp = colors_precomp,
                opacities = opacity_final[idx], 
                scales = scales_final[idx],
                rotations = rotations_final[idx],
                cov3D_precomp = cov3D_precomp,)

            audio_image_list.append(audio_image.unsqueeze(dim=0))
            eye_image_list.append(eye_image.unsqueeze(dim=0))
            cam_image_list.append(cam_image.unsqueeze(dim=0))
            null_image_list.append(null_image.unsqueeze(dim=0))
        
    radii = torch.cat(radii_list,0).max(dim=0).values
    visibility_filter_tensor = torch.cat(visibility_filter_list).any(dim=0)
    rendered_image_tensor = torch.cat(rendered_image_list,0)
    gt_tensor = torch.cat(gt_imgs,0)
    depth_tensor = torch.cat(depth_list,dim=0)
    gt_masks_tensor = torch.cat(gt_masks,dim=0)
    gt_w_bg_tensor = torch.cat(gt_w_bg,dim=0)
    
    audio_image_tensor, eye_image_tensor, null_image_tensor, cam_image_tensor = None, None, None, None
    if visualize_attention:
        audio_image_tensor = torch.cat(audio_image_list,0)
        eye_image_tensor = torch.cat(eye_image_list,0)
        cam_image_tensor = torch.cat(cam_image_list, 0)
        null_image_tensor = torch.cat(null_image_list,0)
    
    rendered_lips_tensor, gt_lips_tensor, rendered_w_bg_tensor = None, None, None
    inference_time = None
    
    if not only_infer:
        rendered_lips_tensor = torch.cat(rendered_lips,0)
        gt_lips_tensor = torch.cat(gt_lips,0)
    if only_infer:
        inference_time = time()-time1
        
        
    return {"rendered_image_tensor": rendered_image_tensor,
        "gt_tensor":gt_tensor,
        "viewspace_points": screenspace_points,
        "visibility_filter_tensor" : visibility_filter_tensor,
        "viewspace_point_tensor_list" : viewspace_point_tensor_list,
        "radii": radii,
        "depth_tensor": depth_tensor,
        "audio_attention": audio_image_tensor,
        "eye_attention": eye_image_tensor,
        "cam_attention" : cam_image_tensor,
        "null_attention": null_image_tensor,
        "rendered_lips_tensor":rendered_lips_tensor,
        "gt_lips_tensor":gt_lips_tensor,
        "rendered_w_bg_tensor":rendered_w_bg_tensor,
        "inference_time":inference_time,
        "gt_masks_tensor":gt_masks_tensor,
        "gt_w_bg_tensor":gt_w_bg_tensor,
        "posterior":posterior,
        }



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import imageio
import os

def visualize_ndc_video(ndc_coordinates, batch_idx=0, point_sample=1000, save_path="ndc_visualization.mp4", fps=5):
    """
    Visualize NDC coordinates for a given batch, with Y and Z swapped, saving frames as a video.
    
    Args:
        ndc_coordinates (torch.Tensor): Tensor of shape [batch_size, num_points, 3].
        batch_idx (int): Batch index to visualize.
        point_sample (int): Number of points to sample for visualization.
        save_path (str): Path to save the output video.
        fps (int): Frames per second for the video.
    """
    # Select batch
    ndc_points = ndc_coordinates[batch_idx].cpu().numpy()  # Shape: [num_points, 3]
    
    # Optionally sample points for visualization
    if point_sample < ndc_points.shape[0]:
        sampled_indices = torch.randperm(ndc_points.shape[0])[:point_sample]
        ndc_points = ndc_points[sampled_indices]
    
    # Extract x, y, z with Y and Z swapped
    x, z, y = ndc_points[:, 0], ndc_points[:, 1], ndc_points[:, 2]
    
    # Prepare temporary directory for frames
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create frames by rotating the plot
    frame_paths = []
    for angle in range(0, 360, 5):  # Rotate from 0 to 360 degrees in 5-degree steps
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', s=1)
        
        # Set axis labels (with swapped Y and Z)
        ax.set_xlabel('X')
        ax.set_ylabel('Z (swapped)')
        ax.set_zlabel('Y (swapped)')
        ax.set_title(f'NDC Visualization (Batch {batch_idx}, Y and Z Swapped)')
        fig.colorbar(scatter, ax=ax, label='Depth (Z)')
        
        # Rotate the view
        ax.view_init(elev=30, azim=angle)
        
        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{angle:03d}.png")
        plt.savefig(frame_path)
        frame_paths.append(frame_path)
        plt.close(fig)
    
    # Create video from frames
    with imageio.get_writer(save_path, fps=fps) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))
    
    # Clean up temporary frames
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(temp_dir)
    
    print(f"Video saved to {save_path}")




def visualize_ndc_with_gt(ndc_coordinates, gt_tensor, batch_idx=0, point_sample=1000):
    """
    Visualize NDC points projected onto the XY plane with a GT image as background.
    
    Args:
        ndc_coordinates (torch.Tensor): NDC coordinates of shape [batch_size, num_points, 3].
        gt_tensor (torch.Tensor): Ground truth images of shape [batch_size, 3, H, W].
        batch_idx (int): Batch index to visualize.
        point_sample (int): Number of points to sample for visualization.
    """
    # Select batch
    ndc_points = ndc_coordinates[batch_idx].cpu().numpy()  # Shape: [num_points, 3]
    gt_image = gt_tensor[batch_idx].permute(1, 2, 0).cpu().numpy()  # Shape: [H, W, 3]
    
    # Optionally sample points for visualization
    if point_sample < ndc_points.shape[0]:
        sampled_indices = torch.randperm(ndc_points.shape[0])[:point_sample]
        ndc_points = ndc_points[sampled_indices]
    
    # Project NDC points onto the XY plane
    x, y = ndc_points[:, 0], ndc_points[:, 1]
    
    # Normalize XY points to match image dimensions
    H, W, _ = gt_image.shape
    x_normalized = ((x + 1) / 2 * W).clip(0, W - 1)  # Normalize to [0, W]
    y_normalized = ((1 - (y + 1) / 2) * H).clip(0, H - 1)  # Normalize to [0, H]
    
    # Create plot
    plt.figure(figsize=(10, 10))
    plt.imshow(gt_image, origin='upper')  # Display GT image as background
    plt.scatter(x_normalized, y_normalized, c='red', s=1, label='NDC Points')  # Overlay NDC points
    plt.title(f'NDC Points on GT Image (Batch {batch_idx})')
    plt.axis('off')
    plt.legend()
    # plt.show()
    plt.savefig("ndc_with_gt.png")