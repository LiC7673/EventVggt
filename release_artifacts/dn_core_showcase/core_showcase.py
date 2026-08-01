                                                                   

                                                                              
                                                                       
                                                                          
   
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def signed_temporal_voxel(split_voxel, bin_count=5, count_ceiling=3.0,
                          bin_age=None, decay_time=0.0015):

    if split_voxel.shape[-3] != 2 * bin_count:
        raise ValueError("expected separate positive/negative temporal bins")
    scale = torch.log1p(torch.as_tensor(count_ceiling, device=split_voxel.device))
    mass = torch.log1p(split_voxel.abs().clamp_max(count_ceiling)) / scale
    sign = mass.new_ones(2 * bin_count)
    sign[bin_count:] = -1
    representation = mass * sign.view(*([1] * (mass.ndim - 3)), -1, 1, 1)
    if bin_age is not None:
        temporal_weight = torch.exp(-bin_age / decay_time).clamp(0, 1)
        temporal_weight = torch.cat((temporal_weight, temporal_weight), dim=-1)
        representation = representation * temporal_weight[..., None, None]
    return representation


class EventNormalDerivativeHead(nn.Module):
                                                                         

    def __init__(self, channels, derivative_limit=0.25):
        super().__init__()
        self.limit = float(derivative_limit)
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 6, 1),
        )

    def forward(self, event_feature):
        raw = self.decoder(event_feature)
        bounded = self.limit * torch.tanh(raw / self.limit)
        return bounded.reshape(raw.shape[0], 2, 3, *raw.shape[-2:])


def forward_normal_difference(normal):
                                                                    
    dx = torch.zeros_like(normal)
    dy = torch.zeros_like(normal)
    dx[..., :, :-1, :] = normal[..., :, 1:, :] - normal[..., :, :-1, :]
    dy[..., :-1, :, :] = normal[..., 1:, :, :] - normal[..., :-1, :, :]
    return torch.cat((dx, dy), dim=-1)


def detail_balanced_loss(error, target_magnitude, valid, quantile=0.70,
                         weak_weight=0.10, floor=1.0e-4):
                                                                                
    values = target_magnitude[valid]
    threshold = (torch.quantile(values.detach(), quantile) if values.numel()
                 else target_magnitude.new_tensor(float("inf")))
    strong = valid & (target_magnitude >= threshold) & (target_magnitude > floor)
    weak = valid & ~strong
    strong_term = (error * strong).sum() / strong.sum().clamp_min(1)
    weak_term = (error * weak).sum() / weak.sum().clamp_min(1)
    return strong_term + weak_weight * weak_term


def normalized_local_mean(value, valid, kernel_size=9):
                                                                                
    padding = kernel_size // 2
    weight = F.avg_pool2d(valid.float(), kernel_size, 1, padding)
    total = F.avg_pool2d(value * valid.float(), kernel_size, 1, padding)
    return total / weight.clamp_min(1.0e-6)


def high_frequency_log_depth_target(base_depth, gt_depth, valid, kernel_size=9):
                                                                       
    residual = torch.log(gt_depth.clamp_min(1.0e-6)) - torch.log(
        base_depth.clamp_min(1.0e-6)
    )
    low_frequency = normalized_local_mean(residual, valid, kernel_size)
    return (residual - low_frequency).detach()


class EventConditionedHDRAdapter(nn.Module):
                                                                          

    def __init__(self, token_dim, bottleneck):
        super().__init__()
        self.rgb_norm = nn.LayerNorm(token_dim)
        self.event_norm = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.rgb_context = nn.Linear(token_dim, bottleneck)
        self.event_modulation = nn.Linear(token_dim, bottleneck, bias=False)
        self.output = nn.Linear(bottleneck, token_dim)

    def forward(self, ldr_token, selected_event_token):
        rgb = F.gelu(self.rgb_context(self.rgb_norm(ldr_token)))
        event = torch.tanh(self.event_modulation(
            self.event_norm(selected_event_token)
        ))
        token_residual = self.output(rgb * event)
        return ldr_token + token_residual, token_residual


class PixelGeometryRefiner(nn.Module):
                                                                                    

    def __init__(self, event_channels, hidden=64, update_limit=0.30):
        super().__init__()
                                                                             
        self.network = nn.Sequential(
            nn.Conv2d(event_channels + 9, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.limit = float(update_limit)

    def forward(self, event_feature, base_depth, current_normal,
                event_normal_target, confidence):
        log_base = torch.log(base_depth.clamp_min(1.0e-6))
        refine_input = torch.cat((event_feature, log_base, log_base,
                                  current_normal, event_normal_target,
                                  confidence), dim=1)
        neutral_input = torch.cat((event_feature, log_base, log_base,
                                   current_normal, current_normal,
                                   confidence), dim=1)
                                                                              
                                                                
        raw = self.network(refine_input) - self.network(neutral_input)
        log_update = self.limit * torch.tanh(raw / self.limit)
        final_depth = torch.exp(log_base + log_update)
        return final_depth, log_update


def differential_geometry_losses(event_derivative, final_depth, gt_depth,
                                 intrinsics, valid, event_support,
                                 depth_to_normals):
                                                                      
    gt_normal = F.normalize(depth_to_normals(gt_depth, intrinsics), dim=-1)
    final_normal = F.normalize(depth_to_normals(final_depth, intrinsics), dim=-1)
    target = forward_normal_difference(gt_normal).detach()
    prediction = event_derivative.movedim(-3, -1).flatten(-2)
    final_derivative = forward_normal_difference(final_normal)
    magnitude = target.norm(dim=-1)
    live = valid & event_support

    event_error = F.smooth_l1_loss(
        prediction, target, beta=0.01, reduction="none"
    ).mean(-1)
    depth_error = F.smooth_l1_loss(
        final_derivative, target, beta=0.01, reduction="none"
    ).mean(-1)
    event_dn_loss = detail_balanced_loss(event_error, magnitude, live)
    depth_dn_loss = detail_balanced_loss(depth_error, magnitude, live)
    return event_dn_loss, depth_dn_loss


def project_patch_centers(depth_patch, source_intrinsics, target_intrinsics,
                          source_to_world, world_to_target, patch_size):
                                                                                  
    height, width = depth_patch.shape[-2:]
    y = (torch.arange(height, device=depth_patch.device) + 0.5) * patch_size - 0.5
    x = (torch.arange(width, device=depth_patch.device) + 0.5) * patch_size - 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    z = depth_patch
    xyz1 = torch.stack((
        (xx - source_intrinsics[0, 2]) * z / source_intrinsics[0, 0],
        (yy - source_intrinsics[1, 2]) * z / source_intrinsics[1, 1],
        z, torch.ones_like(z),
    ), dim=-1)
    world = torch.einsum("ij,hwj->hwi", source_to_world, xyz1)
    target = torch.einsum("ij,hwj->hwi", world_to_target, world)[..., :3]
    u = target_intrinsics[0, 0] * target[..., 0] / target[..., 2] + target_intrinsics[0, 2]
    v = target_intrinsics[1, 1] * target[..., 1] / target[..., 2] + target_intrinsics[1, 2]
    return u, v, target[..., 2]


def cross_view_derivative_consistency(source_magnitude, warped_target_magnitude,
                                      overlap, source_confidence,
                                      target_confidence):
                                                                           
    active = overlap & (source_magnitude > 1.0e-5) & (
        warped_target_magnitude > 1.0e-5
    )
    if active.any():
        calibration = torch.median(
            warped_target_magnitude[active].detach()
            / source_magnitude[active].detach().clamp_min(1.0e-6)
        ).clamp(0.25, 4.0)
    else:
        calibration = source_magnitude.new_tensor(1.0)
    error = F.smooth_l1_loss(
        source_magnitude * calibration,
        warped_target_magnitude,
        beta=0.01,
        reduction="none",
    )
    weight = overlap.float() * torch.sqrt(
        (source_confidence * target_confidence).clamp_min(0)
    )
    return (error * weight).sum() / weight.sum().clamp_min(1.0e-6)


def compose_core_objective(base_geometry_loss, hdr_alignment_loss,
                           event_dn_loss, depth_dn_loss, hf_refiner_loss,
                           cross_view_dn_loss, weights):
                                                                               
    return (
        base_geometry_loss
        + weights.hdr * hdr_alignment_loss
        + weights.event_dn * event_dn_loss
        + weights.depth_dn * depth_dn_loss
        + weights.high_frequency * hf_refiner_loss
        + weights.cross_view * cross_view_dn_loss
    )
