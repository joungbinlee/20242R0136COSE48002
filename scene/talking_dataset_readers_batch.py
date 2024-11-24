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

import os
import sys
from PIL import Image
from scene.cameras import Camera

from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.hyper_loader import Load_hyper_data, format_hyper_data
import torchvision.transforms as transforms
import copy
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm
from scene.utils import get_audio_features
import pandas as pd

import cv2

from scene.dataset_readers import read_timeline, getNerfppNorm

class CameraInfo(NamedTuple):
    path: str
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    full_image: np.array
    full_image_path: str
    image_name: str
    width: int
    height: int
    torso_image: np.array
    torso_image_path: str
    bg_image: np.array
    bg_image_path: str
    mask: np.array
    mask_path: str
    trans: np.array
    face_rect: list
    lhalf_rect: list
    aud_f: torch.FloatTensor
    eye_f: np.array
    eye_rect: list
    lips_rect: list
   
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    custom_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int

def read_and_extract_head(ori_image_path):
    ori_image = cv2.imread(ori_image_path, cv2.IMREAD_UNCHANGED)
    seg = cv2.imread(ori_image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
    head_mask = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
    
    # Create an empty image with the same shape as the original image
    # head_image = np.zeros_like(ori_image)

    # Apply the mask to the original image to extract the head part
    # head_image[head_mask] = ori_image[head_mask]
    
    return ori_image, head_mask
    

def generateCamerasFromTransforms(path, template_transformsfile, extension, maxtime):
    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()
    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w
    cam_infos = []
    # generate render poses and times
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)
    render_times = torch.linspace(0,maxtime,render_poses.shape[0])
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file)
        try:
            fovx = template_json["camera_angle_x"]
        except:
            fovx = focal2fov(template_json["fl_x"], template_json['w'])
    print("hello!!!!")
    # breakpoint()
    # load a single image to get image info.
    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        image = PILtoTorch(image,(800,800))
        break
    # format information
    for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
        time = time/maxtime
        matrix = np.linalg.inv(np.array(poses))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        # fovy = fovx
        FovY = fovy 
        FovX = fovx
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
    return cam_infos

def euler2rot2(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    rot_x = torch.cat(
        (
            torch.cat((one, zero, zero), 1),
            torch.cat((zero, theta.cos(), theta.sin()), 1),
            torch.cat((zero, -theta.sin(), theta.cos()), 1),
        ),
        2,
    )
    rot_y = torch.cat(
        (
            torch.cat((phi.cos(), zero, -phi.sin()), 1),
            torch.cat((zero, one, zero), 1),
            torch.cat((phi.sin(), zero, phi.cos()), 1),
        ),
        2,
    )
    rot_z = torch.cat(
        (
            torch.cat((psi.cos(), -psi.sin(), zero), 1),
            torch.cat((psi.sin(), psi.cos(), zero), 1),
            torch.cat((zero, zero, one), 1),
        ),
        2,
    )
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))

def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
    zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))



def load_eye_features(path):
    au_blink_info=pd.read_csv(os.path.join(path, 'au.csv'))
    try:eye_features = au_blink_info[' AU45_r'].values
    except:eye_features = au_blink_info['AU45_r'].values
    return eye_features

def load_audio_features(path):
    aud_features = np.load(os.path.join(path, 'aud_ds.npy'))
    aud_features = torch.from_numpy(aud_features)

    # support both [N, 16] labels and [N, 16, K] logits
    if len(aud_features.shape) == 3:
        aud_features = aud_features.float().permute(0, 2, 1) # [N, 16, 29] --> [N, 29, 16]    
    else:
        raise NotImplementedError(f'[ERROR] aud_features.shape {aud_features.shape} not supported')
    return aud_features

import os

def validate_dataset_structure(path):
    required_files = [
        "transforms_train.json",
        "transforms_val.json",
        "au.csv",
        "aud_ds.npy",
        "aud_novel.wav",
        "aud_train.wav",
        "aud.wav",
        "bc.jpg",
    ]
    required_dirs = [
        "gt_imgs",
        "ori_imgs",
        "parsing",
        "torso_imgs",
    ]

    valid_datasets = []

    # Check for all dataset directories
    dataset_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for dataset_name in dataset_dirs:
        dataset_path = os.path.join(path, dataset_name)
        
        # Check required files
        if not all(os.path.exists(os.path.join(dataset_path, file)) for file in required_files):
            continue
        
        # Check required directories and their contents
        if not all(
            os.path.isdir(os.path.join(dataset_path, dir_name)) and 
            os.listdir(os.path.join(dataset_path, dir_name))  # Ensure directory is not empty
            for dir_name in required_dirs
        ):
            continue
        
        # Check for specific file types in directories
        def check_images_and_landmarks(dir_name, img_ext=".jpg", lms_ext=".lms"):
            dir_path = os.path.join(dataset_path, dir_name)
            if dir_name == "ori_imgs":  # Check both images and .lms files
                images = [f for f in os.listdir(dir_path) if f.endswith(img_ext)]
                landmarks = [f for f in os.listdir(dir_path) if f.endswith(lms_ext)]
                return len(images) > 0 and len(landmarks) > 0 and len(images) == len(landmarks)
            else:  # Check only images for other directories
                return any(f.endswith(img_ext) for f in os.listdir(dir_path))

        # if not all(check_images_and_landmarks(dir_name) for dir_name in required_dirs):
            # continue

        # Check for .mp4 file with the dataset name
        if not os.path.exists(os.path.join(dataset_path, f"{dataset_name}.mp4")):
            continue

        # Check for track_params.pt
        if not os.path.exists(os.path.join(dataset_path, "track_params.pt")):
            continue

        # If all checks passed, add dataset to valid list
        valid_datasets.append(dataset_name)
        # print(f"Valid dataset: {dataset_name}")

    return valid_datasets



def readTalkingPortraitDatasetInfo_batch(path, white_background, eval, extension=".jpg",custom_aud=None):
    # Audio Information
    # aud_features = load_audio_features(path)    
    # eye_features = load_eye_features(path)
    
    ply_path = os.path.join(path, "fused.ply")
    mesh_path = os.path.join(path, "track_params.pt")
    
    
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTracksTransforms_batch(path, "track_params.pt", "transforms_train.json", extension, preload = False, short_load = True)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTracksTransforms_batch(path, "track_params.pt", "transforms_val.json", extension, short_load = True)
    print("Generating Video Transforms")
    video_cam_infos = None 

    if custom_aud:
        aud_features = np.load(os.path.join(path, custom_aud))
        aud_features = torch.from_numpy(aud_features)
        if len(aud_features.shape) == 3:
            aud_features = aud_features.float().permute(0, 2, 1) # [N, 16, 29] --> [N, 29, 16]    
        else:
            raise NotImplementedError(f'[ERROR] aud_features.shape {aud_features.shape} not supported')
        print("Reading Custom Transforms")
        custom_cam_infos = readCamerasFromTracksTransforms_batch(path, "track_params.pt", "transforms_val.json", aud_features, extension
                                                                 ,custom_aud=custom_aud)
    else:
        custom_cam_infos=None
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if not os.path.exists(ply_path):
        facial_mesh = torch.load(mesh_path)["vertices"]
        average_facial_mesh = torch.mean(facial_mesh, dim=0)
        xyz = average_facial_mesh.cpu().numpy()
        shs = np.random.random((xyz.shape[0], 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))        
    else:
        raise NotImplementedError("No ply file found!")

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           custom_cameras=custom_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=None
                           )
    return scene_info




def readCamerasFromTracksTransforms_batch(path_origin, meshfile, transformsfile, aud_features=None, eye_features=None, 
                                    extension=".jpg", mapper = {}, preload=False, custom_aud =None, short_load = False):
    cam_infos = []
    path_list = validate_dataset_structure(path_origin)
    for i, name in enumerate(tqdm(path_list, desc="Processing datasets")):
        if short_load and i > 10:break
        path = os.path.join(path_origin, name)
        
        eye_features = load_eye_features(path)    
        aud_features = load_audio_features(path)
        
        mesh_path = os.path.join(path, meshfile)
        track_params = torch.load(mesh_path)
        trans_infos = torch.load(mesh_path)["trans"]
        
        with open(os.path.join(path, transformsfile)) as json_file:
            contents = json.load(json_file)
            
        contents["fl_x"] = contents["focal_len"]
        contents["fl_y"] = contents["focal_len"]
        contents["w"] = contents["cx"] * 2
        contents["h"] = contents["cy"] * 2

        fovx = focal2fov(contents['fl_x'],contents['w'])
        fovy = focal2fov(contents['fl_y'],contents['h'])
        frames = contents["frames"]
        f_path = os.path.join(path, "ori_imgs")
        
        FovY = fovy 
        FovX = fovx
        bg_image_path = os.path.join(path, "bc.jpg")
        if custom_aud:
            auds = aud_features
        else:    
            auds = [aud_features[min(frame['aud_id'], aud_features.shape[0] - 1)] for frame in frames]
            auds = torch.stack(auds, dim=0)
            
        for idx, frame in enumerate(frames): # len(frames): 7272
            
            cam_name = os.path.join(f_path, str(frame["img_id"]) + extension)
            # aud_feature = get_audio_features(auds, att_mode = 2, index = idx)     
            
            # Camera Codes
            euler = track_params["euler"][frame["img_id"]]
            R = euler2rot(euler.unsqueeze(0))
            
            flip_rot = torch.tensor(
                [[-1,  0,  0],  # This flips the X axis
                [ 0,  1,  0],  # Y axis remains the same
                [ 0,  0, 1]], # This flips the Z axis, maintaining the right-hand rule
                dtype=R.dtype,
                device=R.device
            ).view(1, 3, 3)
            # flip_rot = flip_rot.expand_as(R)  # Make sure it has the same batch size as R

            # Apply the flip rotation by matrix multiplication
            # Depending on your convention, you might need to apply the flip before or after the original rotation.
            # Use torch.matmul(flip_rot, R) if the flip should be applied globally first,
            # or torch.matmul(R, flip_rot) if the flip should be applied in the camera's local space.
            R = torch.matmul(flip_rot, R)
            R = R.squeeze(0).cpu().numpy()
            T = track_params["trans"][frame["img_id"]].unsqueeze(0).cpu().numpy()
            
            R = -np.transpose(R)
            T = -T
            T[:, 0] = -T[:, 0] 

            # Get Iamges for Facial 
            image_name = Path(cam_name).stem

            full_image_path = cam_name
            torso_image_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
            mask_path = cam_name.replace('ori_imgs', 'parsing').replace('.jpg', '.png')
            
            # Landmark and extract face
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            face_rect = [xmin, xmax, ymin, ymax]
            lhalf_rect = [lh_xmin, lh_xmax, ymin, ymax]
            
            # Eye Area and Eye Rect
            eye_area = eye_features[frame['img_id']]
            eye_area = np.clip(eye_area, 0, 2) / 2
            
            xmin, xmax = int(lms[36:48, 1].min()), int(lms[36:48, 1].max())
            ymin, ymax = int(lms[36:48, 0].min()), int(lms[36:48, 0].max())
            eye_rect = [xmin, xmax, ymin, ymax]
            
            # Finetune Lip Area
            lips = slice(48, 60)
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = max(0, cx - l)
            xmax = min(contents["h"], cx + l)
            ymin = max(0, cy - l)
            ymax = min(contents["w"], cy + l)

            lips_rect = [xmin, xmax, ymin, ymax]
            
            ori_image = None
            torso_img = None
            head_mask = None
            seg = None
            bg_img = None
            
            cam_infos.append(CameraInfo(path=path, uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, full_image=ori_image, full_image_path=full_image_path,
                            image_name=image_name, width=contents["w"], height=contents["h"],
                            torso_image=torso_img, torso_image_path=torso_image_path, bg_image=bg_img, bg_image_path=bg_image_path,
                            mask=seg, mask_path=mask_path, trans=trans_infos[frame["img_id"]],
                            face_rect=face_rect, lhalf_rect=lhalf_rect, aud_f=auds, eye_f=eye_area, eye_rect=eye_rect, lips_rect=lips_rect))
    return cam_infos     


