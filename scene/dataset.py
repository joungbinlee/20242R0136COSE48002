from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
from scene.utils import get_audio_features


import cv2
import os
import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def create_transformation_matrix(R, T):
    T_homogeneous = np.hstack((R, T.reshape(-1, 1)))  # Concatenate R and T horizontally
    T_homogeneous = np.vstack((T_homogeneous, [0, 0, 0, 1]))  # Add the last row for homogeneous coordinates
    return T_homogeneous

def resize_image_to_512(image):
    if image.shape[0] != 512 or image.shape[1] != 512:
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    return image

def resize_rect(rect, original_size, target_size=(512, 512)):
    oy, ox = original_size
    ty, tx = target_size
    
    x1, y1, x2, y2 = rect
    x1 = int(x1 * tx / ox)
    x2 = int(x2 * tx / ox)
    y1 = int(y1 * ty / oy)
    y2 = int(y2 * ty / oy)
    
    return (x1, y1, x2, y2)



class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type,
        aud=None
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        if self.dataset_type != "PanopticSports":

            caminfo = self.dataset[index]
            R = caminfo.R  # (3, 3)
            T = caminfo.T
            FovX = caminfo.FovX
            FovY = caminfo.FovY
            trans = caminfo.trans.cpu().numpy()

            mask = caminfo.mask
            
            full_image = caminfo.full_image
            if full_image is None:
                full_image = cv2.imread(caminfo.full_image_path, cv2.IMREAD_UNCHANGED)
                full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
                full_image = resize_image_to_512(full_image)
                full_image = torch.from_numpy(full_image).permute(2,0,1).float() / 255.0
                
            torso_image = caminfo.torso_image
            if torso_image is None:
                torso_image = cv2.imread(caminfo.torso_image_path, cv2.IMREAD_UNCHANGED) # [H, W, 4]
                torso_image = cv2.cvtColor(torso_image, cv2.COLOR_BGRA2RGBA)
                torso_image = resize_image_to_512(torso_image)
                torso_image = torso_image.astype(np.float32) / 255 # [H, W, 3/4]
                torso_image = torch.from_numpy(torso_image).permute(2, 0, 1) # [3/4, H, W]
                
            bg_image = caminfo.bg_image
            if bg_image is None:
                bg_img = cv2.imread(caminfo.bg_image_path, cv2.IMREAD_UNCHANGED)
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
                bg_img = resize_image_to_512(bg_img)
                bg_image = torch.from_numpy(bg_img).permute(2,0,1).float() / 255.0
            
            seg = caminfo.mask
            if seg is None:
                seg = cv2.imread(caminfo.mask_path)
            seg = resize_image_to_512(seg)
            head_mask = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
            bg_w_torso = torso_image[:3,...] * torso_image[3:,...] + bg_image * (1-torso_image[3:,...])
            
            aud_feature = get_audio_features(caminfo.aud_f, att_mode = 2, index = caminfo.uid)  
            
            original_size = (caminfo.height, caminfo.width)
                        
            face_rect = resize_rect(caminfo.face_rect, original_size)
            lhalf_rect = resize_rect(caminfo.lhalf_rect, original_size)
            eye_rect = resize_rect(caminfo.eye_rect, original_size)
            lips_rect = resize_rect(caminfo.lips_rect, original_size)
            
            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,gt_image=full_image, head_mask=head_mask, bg_image = bg_image,
                    image_name=f"{index}",uid=index,data_device=torch.device("cuda"), #trans=trans,
                    aud_f = aud_feature, eye_f = caminfo.eye_f,
                    face_rect=face_rect, lhalf_rect=lhalf_rect, eye_rect=eye_rect, lips_rect=lips_rect, bg_w_torso = bg_w_torso)
            
        else:
            return self.dataset[index]
    
    def __len__(self):
        
        return len(self.dataset)



class FourDGSdataset_test(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type,
        aud=None
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        if self.dataset_type != "PanopticSports":

            caminfo = self.dataset[index]
            R = caminfo.R  # (3, 3)
            T = caminfo.T
            FovX = caminfo.FovX
            FovY = caminfo.FovY
            trans = caminfo.trans.cpu().numpy()

            mask = caminfo.mask
            
            full_image = caminfo.full_image
            if full_image is None:
                full_image = cv2.imread(caminfo.full_image_path, cv2.IMREAD_UNCHANGED)
                full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
                full_image = torch.from_numpy(full_image).permute(2,0,1).float() / 255.0
                
            torso_image = caminfo.torso_image
            if torso_image is None:
                torso_image = cv2.imread(caminfo.torso_image_path, cv2.IMREAD_UNCHANGED) # [H, W, 4]
                torso_image = cv2.cvtColor(torso_image, cv2.COLOR_BGRA2RGBA)
                torso_image = torso_image.astype(np.float32) / 255 # [H, W, 3/4]
                torso_image = torch.from_numpy(torso_image) # [3/4, H, W]
                torso_image = torso_image.permute(2, 0, 1)

                
            bg_image = caminfo.bg_image
            if bg_image is None:
                bg_img = cv2.imread(caminfo.bg_image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
                if bg_img.shape[0] != caminfo.height or bg_img.shape[1] != caminfo.width:
                    bg_img = cv2.resize(bg_img, (caminfo.width, caminfo.height), interpolation=cv2.INTER_AREA)
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
                bg_image = torch.from_numpy(bg_img).permute(2,0,1).float() / 255.0
            
            seg = caminfo.mask
            if seg is None:
                seg = cv2.imread(caminfo.mask_path)
            head_mask = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
            bg_w_torso = torso_image[:3,...] * torso_image[3:,...] + bg_image * (1-torso_image[3:,...])
                        
            face_rect = caminfo.face_rect
            lhalf_rect = caminfo.lhalf_rect
            eye_rect = caminfo.eye_rect
            lips_rect = caminfo.lips_rect
            
            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,gt_image=full_image, head_mask=head_mask, bg_image = bg_image,
                    image_name=f"{index}",uid=index,data_device=torch.device("cuda"), #trans=trans,
                    aud_f = caminfo.aud_f, eye_f = caminfo.eye_f,
                    face_rect=face_rect, lhalf_rect=lhalf_rect, eye_rect=eye_rect, lips_rect=lips_rect, bg_w_torso = bg_w_torso)
            
        else:
            return self.dataset[index]
    
    def __len__(self):
        
        return len(self.dataset)

def load_audio_features(path):
    aud_features = np.load(os.path.join(path, 'aud_ds.npy'))
    aud_features = torch.from_numpy(aud_features)

    # support both [N, 16] labels and [N, 16, K] logits
    if len(aud_features.shape) == 3:
        aud_features = aud_features.float().permute(0, 2, 1) # [N, 16, 29] --> [N, 29, 16]    
    else:
        raise NotImplementedError(f'[ERROR] aud_features.shape {aud_features.shape} not supported')
    return aud_features