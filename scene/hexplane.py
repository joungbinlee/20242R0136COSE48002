import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
class CustomConvModel(nn.Module):
    def __init__(self, input_dim, output_dims=[64, 128], hidden_dim=32, kernel_size=3):
        super(CustomConvModel, self).__init__()
        
        self.conv_blocks = nn.ModuleList()
        
        for output_size in output_dims:
            blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(output_size)
                ) for _ in range(3)
            ])
            self.conv_blocks.append(blocks)

    def forward(self, x):
        # results = []
        # B,_,_,_ = x.shape
        # for b in range(B):
        #     batch_results = []
        #     for block_group in self.conv_blocks:
        #         group_results = []
        #         for conv_block in block_group:
        #             output = conv_block(x[b:b+1])
        #             group_results.append(output)
        #         batch_results.append(group_results)
        #     results.append(batch_results)
        
        results = []
        B, _, _, _ = x.shape  # Get batch size

        # Iterate over block groups
        for block_group in self.conv_blocks:
            group_results = []  # To store results for the current group
            
            # Process entire batch for each block in the group
            for conv_block in block_group:
                output = conv_block(x)  # Process the entire batch at once
                group_results.append(output)  # Append the batch output
            
            results.append(group_results)

        # Transpose results to group by batch
        # Current results: [G][N][B, C, H, W]
        # Desired results: [B][G][N][C, H, W]
        batch_results = [
            [[results[g][n][b].unsqueeze(dim=0) for n in range(len(results[g]))] for g in range(len(results))]
            for b in range(B)
        ]

        return batch_results
        
    
def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0
def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id,  grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = ( #N, 16
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .reshape(-1, feature_dim)
                # .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


class HexPlaneField(nn.Module):
    def __init__(
        self,
        
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)
        # print(f"Initialized model grids: {self.grids}")
        print("feature_dim:",self.feat_dim)
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)


    def get_density(self, pts: torch.Tensor):
        """Computes and returns the densities."""

        pts = normalize_aabb(pts, self.aabb) # N, 3
        pts = pts.reshape(-1, pts.shape[-1]) # 3, N
        features = interpolate_ms_features(
            pts, ms_grids=self.grids, 
            grid_dimensions=self.grid_config[0]["grid_dimensions"], 
            concat_features=self.concat_features, num_levels=None) 
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)


        return features
    
    
    def forward(self, posterior: torch.Tensor, pts: torch.Tensor):
        features = self.get_density(pts)

        return features




class HexPlaneField_Conv(nn.Module):
    def __init__(
        self,
        
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True
        
        self.conv_in = CustomConvModel(
            4,[64,128],
        )

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)
        # print(f"Initialized model grids: {self.grids}")
        print("feature_dim:",self.feat_dim)
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)


    def get_density(self, posterior: torch.Tensor, pts: torch.Tensor):
        """Computes and returns the densities."""
        B,_,_,_ = posterior.shape
        grids = self.conv_in(posterior)
        features = []
        for idx in range(B):
            pts_ = pts[idx]
            pts_ = normalize_aabb(pts_, self.aabb) # N, 3
            pts_ = pts_.reshape(-1, pts_.shape[-1]) # 3, N
            feature = interpolate_ms_features(
                # pts_, ms_grids=self.grids, 
                pts_, ms_grids=grids[idx], 
                grid_dimensions=self.grid_config[0]["grid_dimensions"], 
                concat_features=self.concat_features, num_levels=None) 
            if len(feature) < 1:
                feature = torch.zeros((0, 1)).to(feature.device)
            features.append(feature)
        features = torch.stack(features)
        return features
    
    
    def forward(self, posterior: torch.Tensor, pts: torch.Tensor):
        features = self.get_density(posterior, pts)

        return features
