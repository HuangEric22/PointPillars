import math
import torch

@torch.no_grad()
def decode_centerpoint(
    preds,
    voxel_size_xy,
    pc_range_xy,
    out_stride=2,
    K=50,
    score_thr=0.1
):
    hm = preds["hm"]      # (B,C,H,W)
    off = preds["off"]    # (B,2,H,W)
    z = preds["z"]        # (B,1,H,W)
    dims = preds["dims"]  # (B,3,H,W)
    rot = preds["rot"]    # (B,2,H,W)

    B, C, H, W = hm.shape
    vx, vy = voxel_size_xy
    x_min, y_min = pc_range_xy
    cell_x = vx * out_stride
    cell_y = vy * out_stride

    results = []
    for b in range(B):
        heat = hm[b].reshape(C, -1)  # (C, H*W)
        scores, inds = torch.topk(heat, K, dim=1)  # per class topk

        bboxes = []
        labels = []
        confs = []

        for cls in range(C):
            for s, ind in zip(scores[cls], inds[cls]):
                if float(s) < score_thr:
                    continue
                ind = int(ind.item())
                iy = ind // W
                ix = ind % W

                ox, oy = off[b, :, iy, ix]
                cx = (ix + ox) * cell_x + x_min
                cy = (iy + oy) * cell_y + y_min

                cz = z[b, 0, iy, ix]
                w, l, h = dims[b, :, iy, ix]
                sin_yaw, cos_yaw = rot[b, :, iy, ix]
                yaw = math.atan2(float(sin_yaw), float(cos_yaw))

                bboxes.append(torch.tensor([cx, cy, cz, w, l, h, yaw], device=hm.device))
                labels.append(cls)
                confs.append(float(s))

        if len(bboxes) == 0:
            results.append({"lidar_bboxes": [], "labels": [], "scores": []})
            continue

        bboxes = torch.stack(bboxes, dim=0)
        results.append({
            "lidar_bboxes": bboxes.detach().cpu().numpy(),
            "labels": torch.tensor(labels, dtype=torch.long).cpu().numpy(),
            "scores": torch.tensor(confs, dtype=torch.float32).cpu().numpy()
        })

    return results
