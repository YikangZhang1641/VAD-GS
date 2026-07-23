"""Run VGGT-Omega before VADT and inspect its cross-view visibility.

The saved NPZ deliberately keeps VGGT geometry in VGGT coordinates.  VADT uses
it only to transfer visibility labels through matching image pixels; it never
uses these camera parameters as Waymo calibration.
"""
from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image


REQUIRED_OUTPUTS = (
    "image_names", "images", "depth", "depth_conf", "extrinsic", "intrinsic",
    "world_points_from_depth",
)

INVALID_VOXEL_ID = np.iinfo(np.uint32).max


def build_pixel_transforms(image_names, image_resolution, target_height, target_width, patch_size=16):
    """Describe VGGT's crop, balanced resize and common-size center padding."""
    transforms = []
    for name in image_names:
        with Image.open(name) as image:
            width, height = image.size
        crop_left = crop_top = 0
        crop_width, crop_height = width, height
        aspect = height / max(width, 1)
        if aspect < 0.5:
            crop_width = min(width, max(1, int(round(height / 0.5))))
            crop_left = max((width - crop_width) // 2, 0)
        elif aspect > 2.0:
            crop_height = min(height, max(1, int(round(width * 2.0))))
            crop_top = max((height - crop_height) // 2, 0)
        aspect = crop_height / max(crop_width, 1)
        tokens = (image_resolution // patch_size) ** 2
        width_patches = max(1, int(np.round(np.sqrt(tokens / aspect))))
        height_patches = max(1, int(np.round(tokens / width_patches)))
        resized_width = width_patches * patch_size
        resized_height = height_patches * patch_size
        pad_left = (target_width - resized_width) // 2
        pad_top = (target_height - resized_height) // 2
        transforms.append((crop_left, crop_top, crop_width, crop_height,
                           resized_width, resized_height, pad_left, pad_top))
    return np.asarray(transforms, dtype=np.float64)


def unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic):
    depth = depth_map[..., 0] if depth_map.ndim == 4 else depth_map
    num_views, height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    xx = np.broadcast_to(xx[None], (num_views, height, width))
    yy = np.broadcast_to(yy[None], (num_views, height, width))
    fx, fy = intrinsic[:, 0, 0, None, None], intrinsic[:, 1, 1, None, None]
    cx, cy = intrinsic[:, 0, 2, None, None], intrinsic[:, 1, 2, None, None]
    camera_points = np.stack(
        ((xx - cx) * depth / fx, (yy - cy) * depth / fy, depth), axis=-1
    )
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "nij,nhwj->nhwi", np.swapaxes(rotation, 1, 2),
        camera_points - translation[:, None, None, :],
    )


def depth_edge(depth, rtol=0.03, kernel_size=3):
    depth = np.asarray(depth)
    original_shape = depth.shape
    flat = depth.reshape(-1, *depth.shape[-2:])
    pad = kernel_size // 2
    padded = np.pad(flat, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    local_min = np.full_like(flat, np.inf)
    local_max = np.full_like(flat, -np.inf)
    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y:y + flat.shape[1], x:x + flat.shape[2]]
            local_min = np.minimum(local_min, window)
            local_max = np.maximum(local_max, window)
    jump = (local_max - local_min) / np.maximum(np.abs(flat), 1e-6)
    return (jump > rtol).reshape(original_shape)


def confidence_mask(depth_conf, depth, percentile, edge_rtol):
    mask = np.isfinite(depth_conf) & (depth_conf > 1e-5)
    if edge_rtol > 0:
        mask &= ~depth_edge(depth[..., 0] if depth.ndim == 4 else depth, edge_rtol)
    if percentile > 0 and np.any(mask):
        mask &= depth_conf >= np.percentile(depth_conf[mask], percentile)
    return mask


def build_vggt_voxel_point_cloud(predictions, voxel_size, conf_percentile, edge_rtol):
    """Build VGGT-Omega's filtered cloud and a compact pixel->voxel->views index.

    Point/view membership is stored as CSR arrays rather than an object array of
    Python lists.  Pixel entries rejected by VGGT's point-cloud filter contain
    ``INVALID_VOXEL_ID``.
    """
    if voxel_size <= 0:
        raise ValueError("--point-cloud-voxel-size must be positive")

    points_map = np.asarray(predictions["world_points_from_depth"])
    num_views, height, width = points_map.shape[:3]
    valid = confidence_mask(
        np.asarray(predictions["depth_conf"]), predictions["depth"],
        conf_percentile, edge_rtol,
    )
    if valid.ndim == 4 and valid.shape[-1] == 1:
        valid = valid[..., 0]
    valid &= np.isfinite(points_map).all(axis=-1)
    if not np.any(valid):
        raise RuntimeError("No VGGT points remain after confidence/depth-edge filtering")

    points = points_map[valid].astype(np.float32, copy=False)
    voxel_keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(
        voxel_keys, axis=0, return_inverse=True, return_counts=True,
    )
    num_voxels = len(counts)
    if num_voxels >= INVALID_VOXEL_ID:
        raise OverflowError("VGGT voxel count does not fit in the uint32 pixel index")

    point_sum = np.zeros((num_voxels, 3), dtype=np.float64)
    np.add.at(point_sum, inverse, points)
    voxel_points = (point_sum / counts[:, None]).astype(np.float32)

    pixel_voxel_ids = np.full((num_views, height, width), INVALID_VOXEL_ID, dtype=np.uint32)
    pixel_voxel_ids[valid] = inverse.astype(np.uint32, copy=False)

    source_views = np.broadcast_to(
        np.arange(num_views, dtype=np.int64)[:, None, None], valid.shape,
    )[valid]
    # Unique (voxel, source view) pairs, ordered by voxel, form CSR directly.
    pairs = np.unique(inverse.astype(np.int64) * num_views + source_views)
    pair_voxels = pairs // num_views
    source_view_ids = (pairs % num_views).astype(
        np.uint16 if num_views <= np.iinfo(np.uint16).max else np.uint32,
    )
    source_offsets = np.zeros(num_voxels + 1, dtype=np.uint64)
    source_offsets[1:] = np.cumsum(
        np.bincount(pair_voxels, minlength=num_voxels), dtype=np.uint64,
    )

    return {
        "vggt_voxel_points": voxel_points,
        "vggt_voxel_ids": np.arange(num_voxels, dtype=np.uint32),
        "vggt_voxel_source_offsets": source_offsets,
        "vggt_voxel_source_view_ids": source_view_ids,
        "vggt_pixel_voxel_ids": pixel_voxel_ids,
        "vggt_invalid_voxel_id": np.asarray(INVALID_VOXEL_ID, dtype=np.uint32),
        "vggt_voxel_size": np.asarray(voxel_size, dtype=np.float32),
    }


def project_points(points, extrinsic, intrinsic):
    cam = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    projected = cam @ intrinsic.T
    z = cam[:, 2]
    return projected[:, 0] / np.maximum(z, 1e-8), projected[:, 1] / np.maximum(z, 1e-8), z


def sample_visibility(predictions, conf_percentile, edge_rtol, pixel_stride,
                      max_points, rel_tol, abs_tol):
    points_map = predictions["world_points_from_depth"]
    depth = predictions["depth"]
    depth_3d = depth[..., 0] if depth.ndim == 4 else depth
    mask = confidence_mask(predictions["depth_conf"], depth, conf_percentile, edge_rtol)
    yy, xx = np.meshgrid(np.arange(points_map.shape[1]), np.arange(points_map.shape[2]), indexing="ij")
    mask &= (yy % max(pixel_stride, 1) == 0) & (xx % max(pixel_stride, 1) == 0)
    source_view = np.broadcast_to(np.arange(len(points_map))[:, None, None], mask.shape)[mask]
    points = points_map[mask]
    colors = predictions["images"]
    if colors.ndim == 4 and colors.shape[1] == 3:
        colors = np.transpose(colors, (0, 2, 3, 1))
    colors = colors[mask]
    finite = np.isfinite(points).all(axis=1)
    points, colors, source_view = points[finite], colors[finite], source_view[finite]
    if max_points > 0 and len(points) > max_points:
        keep = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points, colors, source_view = points[keep], colors[keep], source_view[keep]

    visibility = np.zeros((len(points), len(points_map)), dtype=bool)
    visibility[np.arange(len(points)), source_view] = True
    height, width = depth_3d.shape[1:]
    for view in range(len(points_map)):
        u, v, z = project_points(points, predictions["extrinsic"][view], predictions["intrinsic"][view])
        px, py = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
        inside = (z > 1e-6) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
        idx = np.flatnonzero(inside & (source_view != view))
        if len(idx) == 0:
            continue
        reference = depth_3d[view, py[idx], px[idx]]
        tolerance = np.maximum(abs_tol, rel_tol * np.maximum(reference, 1e-6))
        good = np.isfinite(reference) & (reference > 0) & (np.abs(z[idx] - reference) <= tolerance)
        visibility[idx[good], view] = True
    return points, np.clip(colors, 0, 1), visibility


def voxelize_visibility(points, colors, visibility, voxel_size, max_voxels):
    if voxel_size <= 0:
        raise ValueError("--voxel-size must be positive")
    voxel_index = np.floor(points / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(voxel_index, axis=0, return_inverse=True, return_counts=True)
    centers = np.zeros((len(counts), 3), dtype=np.float64)
    voxel_colors = np.zeros((len(counts), 3), dtype=np.float64)
    voxel_visibility = np.zeros((len(counts), visibility.shape[1]), dtype=bool)
    np.add.at(centers, inverse, points)
    np.add.at(voxel_colors, inverse, colors)
    np.logical_or.at(voxel_visibility, inverse, visibility)
    centers /= counts[:, None]
    voxel_colors /= counts[:, None]
    if max_voxels > 0 and len(counts) > max_voxels:
        keep = np.argsort(counts)[-max_voxels:]
        centers, voxel_colors, voxel_visibility = (
            centers[keep], voxel_colors[keep], voxel_visibility[keep]
        )
    return centers, np.clip(voxel_colors, 0, 1), voxel_visibility


def visualize_visibility(points, colors, visibility, extrinsic, max_voxels,
                         max_rays_per_voxel, ray_length):
    import open3d as o3d

    if max_voxels > 0 and len(points) > max_voxels:
        keep = np.linspace(0, len(points) - 1, max_voxels).astype(np.int64)
        points, colors, visibility = points[keep], colors[keep], visibility[keep]
    rotation, translation = extrinsic[:, :3, :3], extrinsic[:, :3, 3]
    centers = -(np.swapaxes(rotation, 1, 2) @ translation[..., None])[..., 0]
    line_points, lines, line_colors = [], [], []
    for point, visible in zip(points, visibility):
        view_ids = np.flatnonzero(visible)
        if max_rays_per_voxel > 0 and len(view_ids) > max_rays_per_voxel:
            pick = np.linspace(0, len(view_ids) - 1, max_rays_per_voxel).astype(np.int64)
            view_ids = view_ids[pick]
        for view in view_ids:
            direction = centers[view] - point
            norm = np.linalg.norm(direction)
            if norm <= 1e-8:
                continue
            start = len(line_points)
            line_points.extend((point, point + direction / norm * ray_length))
            lines.append((start, start + 1))
            line_colors.append((0.1, 0.85, 1.0))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    geometries = [cloud]
    if lines:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.asarray(line_points))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(line_colors))
        geometries.append(line_set)
    o3d.visualization.draw_geometries(geometries)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the VGGT visibility prior consumed by VADT Waymo.")
    parser.add_argument("--images", default="", help="Glob for input Waymo images, quoted in the shell.")
    parser.add_argument("--image-list", default=None, help="Optional ordered newline-separated image list.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vggt-root", default="", help="VGGT-Omega checkout if it is not installed.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--conf-percentile", type=float, default=20.0)
    parser.add_argument("--depth-edge-rtol", type=float, default=0.03)
    parser.add_argument("--depth-rel-tol", type=float, default=0.05)
    parser.add_argument("--depth-abs-tol", type=float, default=0.05)
    parser.add_argument("--pixel-stride", type=int, default=6)
    parser.add_argument("--max-vis-points", type=int, default=100000)
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--point-cloud-voxel-size", type=float, default=0.08,
                        help="VGGT-space voxel size for the saved pixel->voxel->views index.")
    parser.add_argument("--max-vis-voxels", type=int, default=5000)
    parser.add_argument("--max-rays-per-point", type=int, default=6)
    parser.add_argument("--ray-length", type=float, default=0.25)
    parser.add_argument("--no-vis", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.vggt_root:
        sys.path.insert(0, os.path.abspath(args.vggt_root))
    import torch
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    # if args.image_list:
    #     with open(args.image_list, "r", encoding="utf-8") as handle:
    #         image_names = [line.strip() for line in handle if line.strip()]
    # else:
    #     image_names = sorted(glob(args.images))
    # if not image_names:
    #     raise ValueError(f"No images match {args.images!r}")

    # zyk: for debugging
    image_names = sorted(glob("/home/zyk/Ext/Projects/vadt/data/waymo/training/021/images/0000*_[0-2].png"))[:183]


    model = VGGTOmega().to(args.device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    images = load_and_preprocess_images(image_names, image_resolution=args.image_resolution).to(args.device)
    with torch.inference_mode():
        raw = model(images)
        extrinsic, intrinsic = encoding_to_camera(raw["pose_enc"], raw["images"].shape[-2:])
    raw["extrinsic"], raw["intrinsic"] = extrinsic, intrinsic
    predictions = {}
    for key, value in raw.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions[key] = value
    predictions["image_names"] = np.asarray([str(Path(name).resolve()) for name in image_names])
    target_height, target_width = predictions["depth"].shape[1:3]
    predictions["pixel_transform"] = build_pixel_transforms(
        image_names, args.image_resolution, target_height, target_width
    )
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
    )
    voxel_cache = build_vggt_voxel_point_cloud(
        predictions, args.point_cloud_voxel_size, args.conf_percentile,
        args.depth_edge_rtol,
    )
    predictions.update(voxel_cache)
    missing = set(REQUIRED_OUTPUTS).difference(predictions)
    if missing:
        raise KeyError(f"VGGT output is missing required fields: {sorted(missing)}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # Keep the on-disk interface intentionally small.  Raw VGGT tokens/features,
    # input images and the dense [V,H,W,3] point map are only generation-time data.
    saved_predictions = {
        "image_names": predictions["image_names"],
        "depth": predictions["depth"],
        "pixel_transform": predictions["pixel_transform"],
        **voxel_cache,
    }
    np.savez_compressed(args.output, **saved_predictions)
    source_counts = np.diff(predictions["vggt_voxel_source_offsets"])
    print(f"Saved {len(image_names)} VGGT views and "
          f"{len(predictions['vggt_voxel_points'])} filtered voxels to {args.output}; "
          f"source views/voxel median={np.median(source_counts):.1f}, "
          f"mean={source_counts.mean():.2f}")

    # Visualize the exact cache written above, rather than constructing a second
    # sampled point cloud with different voxelization/visibility semantics.
    points = voxel_cache["vggt_voxel_points"]
    colors = np.full((len(points), 3), (0.72, 0.72, 0.72), dtype=np.float32)
    visibility = np.zeros((len(points), len(image_names)), dtype=np.bool_)
    offsets = voxel_cache["vggt_voxel_source_offsets"]
    source_ids = voxel_cache["vggt_voxel_source_view_ids"].astype(np.int64, copy=False)
    voxel_ids = np.repeat(
        np.arange(len(points), dtype=np.int64),
        np.diff(offsets).astype(np.int64, copy=False),
    )
    visibility[voxel_ids, source_ids] = True
    counts = visibility.sum(axis=1)
    print(f"Saved-cache visualization: {len(points)} voxels; source views/voxel "
          f"median={np.median(counts):.1f}, mean={counts.mean():.2f}, max={counts.max()}")
    if not args.no_vis:
        visualize_visibility(points, colors, visibility, predictions["extrinsic"],
                             args.max_vis_voxels, args.max_rays_per_point, args.ray_length)


if __name__ == "__main__":
    main()
