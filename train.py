import argparse
import os
import torch
from tqdm import tqdm

from pointpillars.utils import setup_seed
from pointpillars.dataset import Kitti, get_dataloader
from pointpillars.model import PointPillars
from torch.utils.tensorboard import SummaryWriter


def save_summary(writer, loss_dict, global_step, tag, lr=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f"{tag}/{k}", v, global_step)
    if lr is not None:
        writer.add_scalar("lr", lr, global_step)


def main(args):
    setup_seed()

    # -------------------------
    # Dataset / Dataloader
    # -------------------------
    train_dataset = Kitti(data_root=args.data_root, split="train")
    val_dataset = Kitti(data_root=args.data_root, split="val")

    train_dataloader = get_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    val_dataloader = get_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # -------------------------
    # Model
    # -------------------------
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model = PointPillars(nclasses=args.nclasses).to(device)

    # AMP scaler
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    # -------------------------
    # Optimizer / Scheduler
    # -------------------------
    max_iters = len(train_dataloader) * args.max_epoch

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.init_lr,
        betas=(0.95, 0.99),
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.init_lr * 10,
        total_steps=max_iters,
        pct_start=0.4,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.95 * 0.895,
        max_momentum=0.95,
        div_factor=10,
    )

    # -------------------------
    # Logging / Checkpoints
    # -------------------------
    log_dir = os.path.join(args.saved_path, "summary")
    ckpt_dir = os.path.join(args.saved_path, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    # =========================
    # Training Loop
    # =========================
    global_step = 0

    for epoch in range(args.max_epoch):
        print("=" * 20, f"Epoch {epoch}", "=" * 20)

        # -------- Train --------
        model.train()
        for data_dict in tqdm(train_dataloader):
            # Move batch to device
            for key in data_dict:
                for i, item in enumerate(data_dict[key]):
                    if torch.is_tensor(item):
                        data_dict[key][i] = item.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            batched_pts = data_dict["batched_pts"]
            batched_gt_bboxes = data_dict["batched_gt_bboxes"]
            batched_labels = data_dict["batched_labels"]

            with torch.amp.autocast(
                device_type=device,
                enabled=(device == "cuda"),
            ):
                loss, loss_dict = model(
                    batched_pts=batched_pts,
                    mode="train",
                    batched_gt_bboxes=batched_gt_bboxes,
                    batched_gt_labels=batched_labels,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if global_step % args.log_freq == 0:
                save_summary(
                    writer,
                    loss_dict,
                    global_step,
                    tag="train",
                    lr=optimizer.param_groups[0]["lr"],
                )

            global_step += 1

        # -------- Checkpoint --------
        if (epoch + 1) % args.ckpt_freq_epoch == 0:
            torch.save(
                model.state_dict(),
                os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pth"),
            )

        # -------- Validation (loss only) --------
        model.eval()
        with torch.no_grad():
            for data_dict in tqdm(val_dataloader):
                for key in data_dict:
                    for i, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][i] = item.to(device, non_blocking=True)

                loss, loss_dict = model(
                    batched_pts=data_dict["batched_pts"],
                    mode="train",  # train mode = compute loss
                    batched_gt_bboxes=data_dict["batched_gt_bboxes"],
                    batched_gt_labels=data_dict["batched_labels"],
                )

                if global_step % args.log_freq == 0:
                    save_summary(writer, loss_dict, global_step, tag="val")

                global_step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CenterPoint Training")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--saved_path", type=str, default="pillar_logs")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--nclasses", type=int, default=3)
    parser.add_argument("--init_lr", type=float, default=2.5e-4)
    parser.add_argument("--max_epoch", type=int, default=160)
    parser.add_argument("--log_freq", type=int, default=5)
    parser.add_argument("--ckpt_freq_epoch", type=int, default=20)
    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()
    main(args)
