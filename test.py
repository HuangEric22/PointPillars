import argparse
import cv2
import numpy as np
import os
import torch

from pointpillars.utils import (
    read_points,
    read_calib,
    read_label,
    keep_bbox_from_image_range,
    keep_bbox_from_lidar_range,
    vis_pc,
    vis_img_3d,
    bbox3d2corners_camera,
    points_camera2image,
    bbox_camera2lidar,
)
from pointpillars.model import PointPillars


def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    mask = (
        (pts[:, 0] > point_range[0])
        & (pts[:, 1] > point_range[1])
        & (pts[:, 2] > point_range[2])
        & (pts[:, 0] < point_range[3])
        & (pts[:, 1] < point_range[4])
        & (pts[:, 2] < point_range[5])
    )
    return pts[mask]


def main(args):
    CLASSES = {
        "Pedestrian": 0,
        "Cyclist": 1,
        "Car": 2,
    }

    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    # -------------------------
    # Model
    # -------------------------
    if not args.no_cuda:
        model = PointPillars(nclasses=len(CLASSES)).cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillars(nclasses=len(CLASSES))
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device("cpu"))
        )

    model.eval()

    # -------------------------
    # Load point cloud
    # -------------------------
    if not os.path.exists(args.pc_path):
        raise FileNotFoundError(args.pc_path)

    pc = read_points(args.pc_path)
    pc = point_range_filter(pc)
    pc_torch = torch.from_numpy(pc).float()

    if not args.no_cuda:
        pc_torch = pc_torch.cuda()

    # -------------------------
    # Optional inputs
    # -------------------------
    calib_info = read_calib(args.calib_path) if args.calib_path else None
    gt_label = read_label(args.gt_path) if args.gt_path else None
    img = cv2.imread(args.img_path) if args.img_path else None

    # -------------------------
    # Inference (CenterPoint)
    # -------------------------
    with torch.no_grad():
        results = model(batched_pts=[pc_torch], mode="test")

    assert len(results) == 1
    result = results[0]

    # result keys:
    # {
    #   "lidar_bboxes": (N, 7),
    #   "labels": (N,),
    #   "scores": (N,)
    # }

    # -------------------------
    # Post-filtering
    # -------------------------
    if calib_info is not None and img is not None:
        tr_velo_to_cam = calib_info["Tr_velo_to_cam"].astype(np.float32)
        r0_rect = calib_info["R0_rect"].astype(np.float32)
        P2 = calib_info["P2"].astype(np.float32)

        image_shape = img.shape[:2]
        result = keep_bbox_from_image_range(
            result, tr_velo_to_cam, r0_rect, P2, image_shape
        )

    result = keep_bbox_from_lidar_range(result, pcd_limit_range)

    lidar_bboxes = result["lidar_bboxes"]
    labels = result["labels"]
    scores = result["scores"]

    # -------------------------
    # Visualize LiDAR
    # -------------------------
    vis_pc(pc, bboxes=lidar_bboxes, labels=labels)

    # -------------------------
    # Visualize Image (optional)
    # -------------------------
    if calib_info is not None and img is not None and len(lidar_bboxes) > 0:
        bboxes_camera = result["camera_bboxes"]
        corners = bbox3d2corners_camera(bboxes_camera)
        image_points = points_camera2image(corners, P2)
        img = vis_img_3d(img, image_points, labels, rt=True)
        cv2.imshow("CenterPoint 3D Detection", img)
        cv2.waitKey(0)

    # -------------------------
    # Compare with GT (optional)
    # -------------------------
    if calib_info is not None and gt_label is not None:
        tr_velo_to_cam = calib_info["Tr_velo_to_cam"].astype(np.float32)
        r0_rect = calib_info["R0_rect"].astype(np.float32)

        dims = gt_label["dimensions"]
        loc = gt_label["location"]
        ry = gt_label["rotation_y"]

        gt_boxes_cam = np.concatenate([loc, dims, ry[:, None]], axis=-1)
        gt_boxes_lidar = bbox_camera2lidar(
            gt_boxes_cam, tr_velo_to_cam, r0_rect
        )

        combined_boxes = np.concatenate([lidar_bboxes, gt_boxes_lidar], axis=0)
        combined_labels = np.concatenate(
            [labels, -np.ones(len(gt_boxes_lidar))]
        )

        vis_pc(pc, combined_boxes, labels=combined_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CenterPoint Test")

    parser.add_argument("--ckpt", required=True, help="model checkpoint")
    parser.add_argument("--pc_path", required=True, help="point cloud path")
    parser.add_argument("--calib_path", default="", help="calib file")
    parser.add_argument("--gt_path", default="", help="ground truth label")
    parser.add_argument("--img_path", default="", help="image path")
    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()
    main(args)
