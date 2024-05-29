import torch
import torch.nn as nn

import numpy as np

def none_safe_collate(batch):
    batch = [x for x in batch if x is not None]
    if batch:
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)
    else:
        return {}

def init_weights(m):
    """ Initialize weights according to the FlowNet2-pytorch from nvidia """
    if isinstance(m, nn.Linear):
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-.0001, b=0.0001)
        nn.init.xavier_uniform_(m.weight, gain=0.001)

    if isinstance(m, nn.Conv1d):
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-.001, b=0.001)
        nn.init.xavier_uniform_(m.weight, gain=0.01)

    if isinstance(m, nn.Conv2d):
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-.001, b=0.001)
        nn.init.xavier_uniform_(m.weight, gain=0.01)

    if isinstance(m, nn.ConvTranspose2d):
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-.001, b=0.001)
        nn.init.xavier_uniform_(m.weight, gain=0.01)

def num_trainable_parameters(module):
    trainable_parameters = filter(lambda p: p.requires_grad,
                                  module.parameters())
    return sum([np.prod(p.size()) for p in trainable_parameters])


def num_parameters(network):
    n_params = 0
    modules = list(network.modules())

    for mod in modules:
        parameters = mod.parameters()
        n_params += sum([np.prod(p.size()) for p in parameters])
    return n_params

def calc_floor_ceil_delta(x): 
    x_fl = torch.floor(x + 1e-8)
    x_ce = torch.ceil(x - 1e-8)
    x_ce_fake = torch.floor(x) + 1

    dx_ce = x - x_fl
    dx_fl = x_ce_fake - x
    return [x_fl.long(), dx_fl], [x_ce.long(), dx_ce]

def create_update(x, y, t, dt, p, vol_size, device="cpu"):
    assert (x>=0).byte().all() 
    assert (x<vol_size[2]).byte().all()
    assert (y>=0).byte().all()
    assert (y<vol_size[1]).byte().all()
    assert (t>=0).byte().all() 
    #assert (t<vol_size[0] // 2).byte().all()

    if not (t < vol_size[0] // 2).byte().all():
        print(t[t >= vol_size[0] // 2])
        print(vol_size)
        raise AssertionError()

    vol_mul = torch.where(p < 0,
                          torch.ones(p.shape, dtype=torch.long).to(device) * vol_size[0] // 2,
                          torch.zeros(p.shape, dtype=torch.long).to(device))

    inds = (vol_size[1]*vol_size[2]) * (t + vol_mul)\
         + (vol_size[2])*y\
         + x

    vals = dt

    return inds, vals

def gen_discretized_event_volume(events, vol_size, device="cpu"):
    # volume is [t, x, y]
    # events are Nx4
    npts = events.shape[0]
    volume = events.new_zeros(vol_size)

    x = events[:, 0].long()
    y = events[:, 1].long()
    t = events[:, 2]

    t_min = t.min()
    t_max = t.max()
    t_scaled = (t-t_min) * ((vol_size[0] // 2-1) / (t_max-t_min + 1e-6))

    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())
    
    inds_fl, vals_fl = create_update(x, y,
                                     ts_fl[0], ts_fl[1],
                                     events[:, 3],
                                     vol_size,
                                     device=device)
        
    volume.view(-1).put_(inds_fl, vals_fl, accumulate=True)

    inds_ce, vals_ce = create_update(x, y,
                                     ts_ce[0], ts_ce[1],
                                     events[:, 3],
                                     vol_size,
                                     device=device)
    volume.view(-1).put_(inds_ce, vals_ce, accumulate=True)
    return volume

def normalize_event_volume(event_volume):
    event_volume_flat = event_volume.view(-1)
    nonzero = torch.nonzero(event_volume_flat)
    nonzero_values = event_volume_flat[nonzero]
    if nonzero_values.shape[0]:
        lower = torch.kthvalue(nonzero_values,
                               max(int(0.02 * nonzero_values.shape[0]), 1),
                               dim=0)[0][0]
        upper = torch.kthvalue(nonzero_values,
                               max(int(0.98 * nonzero_values.shape[0]), 1),
                               dim=0)[0][0]
        max_val = max(abs(lower), upper)
        event_volume = torch.clamp(event_volume, -max_val, max_val)
        event_volume /= max_val
    return event_volume

def create_update_xyt(x, y, t, dx, dy, dt, p, vol_size, device="cpu"):
    assert (x>=0).byte().all() 
    assert (x<vol_size[2]).byte().all()
    assert (y>=0).byte().all() 
    assert (y<vol_size[1]).byte().all()
    assert (t>=0).byte().all() 
    assert (t<vol_size[0]).byte().all()

    #vol_mul = torch.where(p < 0,
    #                      torch.ones(p.shape, dtype=torch.long).to(device) * vol_size[0] // 2,
    #                      torch.zeros(p.shape, dtype=torch.long).to(device))
    
    # only look at positive events
    vol_mul = 0
    inds = (vol_size[1]*vol_size[2]) * (t)\
         + (vol_size[2])*y\
         + x

    vals = dx * dy * dt
    return inds, vals

def gen_discretized_event_volume_xyt(events, vol_size, weight=None, device="cpu"):
    # volume is [t, x, y]
    # events are Nx4
    npts = events.shape[0]
    volume = events.new_zeros(vol_size)

    x = events[:, 0]
    y = events[:, 1]
    t = events[:, 2]

    # scale t
    t_min = t.min()
    t_max = t.max()
    t_scaled = (t-t_min) * ((vol_size[0]-1) / (t_max-t_min + 1e-7))

    # scale x and y
    x_scaled = (x + 1e-8) * (vol_size[2] - 1)
    y_scaled = (y + 1e-8) * (vol_size[1] - 1)

    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())
    xs_fl, xs_ce = calc_floor_ceil_delta(x_scaled.squeeze())
    ys_fl, ys_ce = calc_floor_ceil_delta(y_scaled.squeeze())

    all_ts_options = [ts_fl, ts_ce]
    all_xs_options = [xs_fl, xs_ce]
    all_ys_options = [ys_fl, ys_ce]

    # interpolate in all three directions
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # choose a set of index and a set of interpolants
                x, y, t = all_xs_options[i][0], all_ys_options[j][0], all_ts_options[k][0]
                dx, dy, dt = all_xs_options[i][1], all_ys_options[j][1], all_ts_options[k][1]
                inds, vals = create_update_xyt(x, y, t, dx, dy, dt, 
                        events[..., 3], vol_size, device=device)
                if weight is None:
                    volume.view(-1).put_(inds, vals, accumulate=True)
                else:
                    volume.view(-1).put_(inds, vals*weight, accumulate=True)
    return volume
 

"""
 Network output is BxHxWxNx4, all between -1 and 1. Each 4-tuple is [x, y, t, p], 
 where [x, y] are relative to the center of the grid cell at that hxw pixel.
 This function scales this output to values in the range:
 [[0, volume_size[0]], [0, volume_size[1]], [0, volume_size[2]], [-1, 1]]
"""
def scale_events(events, volume_size, device='cuda'):
    # Compute the center of each grid cell.
    scale = volume_size[0] / events.shape[1]
    x_range = torch.arange(events.shape[2]).to(device) * scale + scale / 2
    y_range = torch.arange(events.shape[1]).to(device) * scale + scale / 2
    x_offset, y_offset = torch.meshgrid(x_range, y_range)
    
    t_scale = (volume_size[2] - 1) / 2.
    # Offset the timestamps from [-1, 1] to [0, 2].
    t_offset = torch.ones(x_offset.shape).to(device) * t_scale
    p_offset = torch.zeros(x_offset.shape).to(device)
    offset = torch.stack((x_offset.float(), y_offset.float(), t_offset, p_offset), dim=-1)
    offset = offset[None, ..., None, :]

    # Scale the [x, y] values to [-scale/2, scale/2] and
    # t to [-volume_size[2] / 2, volume_size[2] / 2].
    output_scale = torch.tensor((scale / 2, scale / 2, t_scale, 1))\
                        .to(device).reshape((1, 1, 1, 1, -1))

    # Scale the network output
    events *= output_scale

    # Offset the network output
    events += offset

    events = torch.reshape(events, (events.shape[0], -1, 4))

    return events

def generate_random_samples(batch_size, radius, num_points, dim, inside=True, device="cpu"):
    assert dim >= 1
    points = torch.empty(batch_size, num_points, dim).normal_()
    directions = points / points.norm(2, dim=2, keepdim=True) 
    if inside:
        dist_center = torch.empty(batch_size, num_points, 1).uniform_() * radius
    else:
        dist_center = torch.ones(batch_size, num_points, 1)
    samples = directions * dist_center
    return samples.to(device)

if __name__ == "__main__":
    
    '''
    events = torch.rand(3, 1000, 4).cuda()
    events[:, :, 3] = torch.rand(3, 1000).cuda() - 0.5
    vol_size = [3, 9, 100, 100]

    gen_batch_discretized_event_volume(events, vol_size)
    '''
    events = torch.rand(1000, 4, requires_grad=True) * 100
    events[:, 2] = torch.arange(1000).float() / 1000
    events[:, 3] = torch.rand(1000) - 0.5
    events = torch.nn.parameter.Parameter(events)
    vol_size = [9, 100, 100]

    '''
    import time
    t0 = time.time()
    print(time.time() - t0)

    mean = events.mean()
    mean.backward()
    print(events.grad)
    '''
    optimizer = torch.optim.Adam([events], lr=0.1)
    for j in range(100):
        total_volume = []
        for i in range(8):
            event_volume = gen_discretized_event_volume(events, vol_size)
            total_volume.append(event_volume)
        total_volume = torch.stack(total_volume, axis=0)
        mean = events.mean()
        mean.backward()

        optimizer.step()
        optimizer.zero_grad()
        print(mean.item())


def gen_event_images(event_volume, prefix, device="cuda", clamp_val=2., normalize_events=True, signed=True):
    
    if signed:
        n_bins = int(event_volume.shape[1] / 2)
        time_range = torch.tensor(np.linspace(0.1, 1, n_bins), dtype=torch.float32).to(device)
        time_range = torch.reshape(time_range, (1, n_bins, 1, 1))
        
        pos_event_image = torch.sum(
            event_volume[:, :n_bins, ...] * time_range / \
            (torch.sum(event_volume[:, :n_bins, ...], dim=1, keepdim=True) + 1e-5),
            dim=1, keepdim=True)
        neg_event_image = torch.sum(
            event_volume[:, n_bins:, ...] * time_range / \
            (torch.sum(event_volume[:, n_bins:, ...], dim=1, keepdim=True) + 1e-5),
            dim=1, keepdim=True)
        event_time_image = (pos_event_image + neg_event_image) / 2.
    else:
        n_bins = event_volume.shape[1]
        time_range = torch.tensor(np.linspace(0.1, 1, n_bins), dtype=torch.float32).to(device)
        time_range = torch.reshape(time_range, (1, n_bins, 1, 1))

        event_time_image = torch.sum(
                event_volume[:, :, ...] * time_range / \
                (torch.sum(event_volume[:, :, ...], dim=1, keepdim=True) + 1e-5),
            dim=1, keepdim=True)

    
    return event_time_image