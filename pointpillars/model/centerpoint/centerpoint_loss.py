import torch
import torch.nn.functional as F

# ----------------------------
# Heatmap focal loss (AMP-safe)
# ----------------------------
def focal_loss_centerpoint(pred, gt):
    # pred, gt: (B,C,H,W), pred in [0,1]
    eps = 1e-6
    pred = pred.clamp(min=eps, max=1.0 - eps)

    pos = gt.eq(1).float()
    neg = gt.lt(1).float()
    neg_weights = (1 - gt).pow(4)

    pos_loss = -torch.log(pred) * (1 - pred).pow(2) * pos
    neg_loss = -torch.log(1 - pred) * pred.pow(2) * neg * neg_weights

    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


# ----------------------------
# Masked L1 (stable)
# ----------------------------
def masked_l1(pred, gt, mask):
    mask = mask.expand_as(pred).float()
    loss = F.l1_loss(pred * mask, gt * mask, reduction="sum")
    denom = mask.sum().clamp(min=1.0)
    return loss / denom


# ----------------------------
# CenterPoint loss
# ----------------------------
def centerpoint_loss(
    preds, targets,
    w_hm=1.0, w_off=1.0, w_z=1.0, w_dims=1.0, w_rot=1.0
):
    # Heatmap in FP32 for numerical safety
    loss_hm = focal_loss_centerpoint(
        preds["hm"].float(),
        targets["hm"].float(),
    )

    loss_off  = masked_l1(preds["off"],  targets["off"],  targets["mask"])
    loss_z    = masked_l1(preds["z"],    targets["z"],    targets["mask"])
    loss_dims = masked_l1(preds["dims"], targets["dims"], targets["mask"])
    loss_rot  = masked_l1(preds["rot"],  targets["rot"],  targets["mask"])

    total = (
        w_hm   * loss_hm +
        w_off  * loss_off +
        w_z    * loss_z +
        w_dims * loss_dims +
        w_rot  * loss_rot
    )

    # Safety guard
    if not torch.isfinite(total):
        raise RuntimeError("NaN/Inf detected in CenterPoint loss")

    return total, {
        "hm": loss_hm.detach(),
        "off": loss_off.detach(),
        "z": loss_z.detach(),
        "dims": loss_dims.detach(),
        "rot": loss_rot.detach(),
        "total": total.detach(),
    }
