import torch
from typing import NamedTuple, Optional, Tuple, Union, List
from collections import namedtuple
import raster


class RasterizePoints(torch.autograd.Function):
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx,
        points,  # (P, 3)
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size: Union[List[int], Tuple[int, int]] = (256, 256),
        radius: Union[float, torch.Tensor] = 0.01,
        points_per_pixel: int = 8,
        bin_size: int = 0,
        max_points_per_bin: int = 0,
    ):
        # TODO: Add better error handling for when there are more than
        # max_points_per_bin in any bin.
        
        args = (
            points.type(torch.float32),
            cloud_to_packed_first_idx.type(torch.long),
            num_points_per_cloud.type(torch.long),
            image_size,
            radius.type(torch.float32),
            points_per_pixel,
            bin_size,
            max_points_per_bin,
        )
        
        # import pdb; pdb.set_trace()
        # pyre-fixme[16]: Module `pytorch3d` has no attribute `_C`.
        idx, zbuf, dists = raster.rasterize_points(*args)
        # ctx.save_for_backward(points, idx)
        # ctx.mark_non_differentiable(idx)
        return idx, zbuf, dists
