import math
import torch

def gaussian2d(shape, sigma, device):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = torch.meshgrid(
        torch.arange(-m, m + 1, device=device),
        torch.arange(-n, n + 1, device=device),
        indexing="ij"
    )
    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return h

def draw_gaussian(heatmap, center_x, center_y, radius):
    # heatmap: (H, W)
    H, W = heatmap.shape
    x, y = center_x, center_y
    r = radius
    if r <= 0:
        if 0 <= x < W and 0 <= y < H:
            heatmap[y, x] = torch.maximum(heatmap[y, x], heatmap.new_tensor(1.0))
        return

    diameter = 2 * r + 1
    sigma = diameter / 6.0  # common choice
    g = gaussian2d((diameter, diameter), sigma, heatmap.device)

    left = min(x, r)
    right = min(W - x - 1, r)
    top = min(y, r)
    bottom = min(H - y - 1, r)

    masked_hm = heatmap[y - top:y + bottom + 1, x - left:x + right + 1]
    masked_g = g[r - top:r + bottom + 1, r - left:r + right + 1]
    torch.maximum(masked_hm, masked_g, out=masked_hm)

def radius_from_size(w_cells, l_cells, min_radius=1, max_radius=10):
    # Simple, stable heuristic (good enough for KITTI):
    # larger objects -> larger gaussian
    r = int(max(min(w_cells, l_cells) / 2.0, min_radius))
    return min(r, max_radius)

@torch.no_grad()
def build_centerpoint_targets(
    batched_gt_bboxes,
    batched_gt_labels,
    feature_map_hw,          # (H, W) = (248, 216)
    voxel_size_xy,           # (vx, vy) original pillar size, e.g. (0.16, 0.16)
    pc_range_xy,             # (x_min, y_min)
    out_stride=2,            # IMPORTANT in your net
    num_classes=3
):
    """
    Returns dict of batched targets:
      hm:   (B, C, H, W)
      off:  (B, 2, H, W)
      z:    (B, 1, H, W)
      dims: (B, 3, H, W)   (w,l,h)
      rot:  (B, 2, H, W)   (sin,cos)
      mask: (B, 1, H, W)   1 at centers
      ind:  (B, max_objs)  flattened indices (optional for gather-based losses)
    """
    H, W = feature_map_hw
    vx, vy = voxel_size_xy
    x_min, y_min = pc_range_xy

    B = len(batched_gt_bboxes)
    device = batched_gt_bboxes[0].device

    hm = torch.zeros((B, num_classes, H, W), device=device)
    off = torch.zeros((B, 2, H, W), device=device)
    z = torch.zeros((B, 1, H, W), device=device)
    dims = torch.zeros((B, 3, H, W), device=device)
    rot = torch.zeros((B, 2, H, W), device=device)
    mask = torch.zeros((B, 1, H, W), device=device)

    cell_x = vx * out_stride
    cell_y = vy * out_stride

    for b in range(B):
        gt = batched_gt_bboxes[b]
        labels = batched_gt_labels[b]
        if gt.numel() == 0:
            continue

        for box, cls in zip(gt, labels):
            cls = int(cls.item())
            x, y, zz, w, l, h, yaw = box

            # map to feature map coordinates
            u = (x - x_min) / cell_x
            v = (y - y_min) / cell_y
            ix = int(torch.floor(u).item())
            iy = int(torch.floor(v).item())
            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                continue

            # gaussian radius in cells
            w_cells = float(w.item() / cell_x)
            l_cells = float(l.item() / cell_y)
            r = radius_from_size(w_cells, l_cells)

            draw_gaussian(hm[b, cls], ix, iy, r)

            off[b, :, iy, ix] = torch.tensor([u - ix, v - iy], device=device)
            z[b, 0, iy, ix] = zz
            dims[b, :, iy, ix] = torch.tensor([w, l, h], device=device)
            rot[b, :, iy, ix] = torch.tensor([math.sin(float(yaw)), math.cos(float(yaw))], device=device)
            mask[b, 0, iy, ix] = 1.0

    return {"hm": hm, "off": off, "z": z, "dims": dims, "rot": rot, "mask": mask}
