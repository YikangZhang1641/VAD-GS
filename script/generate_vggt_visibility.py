"""Generate the offline VGGT-Omega pixel visibility cache for VAD-GS.

The saved NPZ keeps VGGT geometry in its own coordinates and records
pixel-to-voxel-to-view indices. No dataset point cloud or online worker is used.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# Support both `python script/generate_vggt_visibility.py` and `python -m ...`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REQUIRED_OUTPUTS = (
    "image_names", "depth", "depth_conf", "extrinsic", "intrinsic",
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
        # Match VGGT-Omega's balanced loader: round both unrounded dimensions.
        width_patches_float = np.sqrt(tokens / aspect)
        height_patches_float = tokens / width_patches_float
        width_patches = max(1, int(np.round(width_patches_float)))
        height_patches = max(1, int(np.round(height_patches_float)))
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


def confidence_mask(
    depth_conf, depth, percentile, edge_rtol, sky_masks=None,
):
    depth_conf = np.asarray(depth_conf)
    depth = np.asarray(depth)
    if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
        depth_conf = depth_conf[..., 0]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth_conf.shape != depth.shape:
        raise ValueError("VGGT confidence and depth shapes must match")
    mask = np.isfinite(depth_conf) & (depth_conf > 1e-5)
    if sky_masks is not None:
        sky_masks = np.asarray(sky_masks, dtype=np.bool_)
        if sky_masks.ndim == 4 and sky_masks.shape[-1] == 1:
            sky_masks = sky_masks[..., 0]
        if sky_masks.shape != mask.shape:
            raise ValueError(
                f"sky mask shape {sky_masks.shape} does not match "
                f"confidence shape {mask.shape}"
            )
        # VADT masks use True for sky. VGGT's example helper instead
        # multiplies confidence by a non-sky/valid-region mask.
        mask &= ~sky_masks
    if edge_rtol > 0:
        mask &= ~depth_edge(depth, edge_rtol)
    if percentile > 0 and np.any(mask):
        mask &= depth_conf >= np.percentile(depth_conf[mask], percentile)
    return mask


def build_vggt_voxel_point_cloud(
    predictions, relative_voxel_size, conf_percentile, edge_rtol,
    sky_masks=None,
):
    """Build VGGT-Omega's filtered cloud and a compact pixel->voxel->views index.

    Point/view membership is stored as CSR arrays rather than an object array of
    Python lists.  Pixel entries rejected by VGGT's point-cloud filter contain
    ``INVALID_VOXEL_ID``.
    """
    if relative_voxel_size <= 0:
        raise ValueError("relative_voxel_size must be positive")

    points_map = np.asarray(predictions["world_points_from_depth"])
    depth = np.asarray(predictions["depth"])
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    num_views, height, width = points_map.shape[:3]
    valid = confidence_mask(
        np.asarray(predictions["depth_conf"]), predictions["depth"],
        conf_percentile, edge_rtol, sky_masks,
    )
    if valid.ndim == 4 and valid.shape[-1] == 1:
        valid = valid[..., 0]
    valid &= np.isfinite(points_map).all(axis=-1)
    valid &= np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise RuntimeError("No VGGT points remain after confidence/depth-edge filtering")

    # VGGT geometry has an arbitrary scene-dependent scale.  Express the
    # pseudo-voxel resolution relative to the median valid predicted depth so
    # the pixel/voxel visibility index has comparable density across scenes.
    vggt_scale = float(np.median(depth[valid]))
    if not np.isfinite(vggt_scale) or vggt_scale <= 0:
        raise RuntimeError("Cannot estimate a positive VGGT depth scale")
    voxel_size = float(relative_voxel_size) * vggt_scale

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
        "vggt_scale": np.asarray(vggt_scale, dtype=np.float32),
        "vggt_relative_voxel_size": np.asarray(
            relative_voxel_size, dtype=np.float32,
        ),
    }


def load_prediction_sky_masks(
    image_names, source_path, pixel_transforms, target_height, target_width,
    explicit_mask_paths=None, explicit_mask_kind="binary",
):
    """Load VADT sky masks and map them through VGGT's image transform."""
    sky_dir = None if explicit_mask_paths is not None else next(
        (source_path / name for name in ("sky_mask", "sky_masks")
         if (source_path / name).is_dir()), None,
    )
    if explicit_mask_paths is None and sky_dir is None:
        raise FileNotFoundError(
            f"Sky-mask directory not found under {source_path}; expected "
            "sky_mask/ or sky_masks/"
        )

    nearest = getattr(Image, "Resampling", Image).NEAREST
    transformed_masks = []
    if explicit_mask_paths is None:
        mask_paths = []
        for image_name in image_names:
            image_path = Path(image_name)
            candidates = (
                sky_dir / f"{image_path.stem}.png",
                sky_dir / image_path.name,
                sky_dir / f"{image_path.stem}.jpg",
            )
            mask_paths.append(next((path for path in candidates if path.is_file()), None))
    else:
        mask_paths = list(map(Path, explicit_mask_paths))
        if len(mask_paths) != len(image_names):
            raise ValueError("Explicit sky-mask and image counts do not match")

    for image_name, mask_path, transform in zip(image_names, mask_paths, pixel_transforms):
        image_path = Path(image_name)
        if mask_path is None or not mask_path.is_file():
            raise FileNotFoundError(
                f"Sky mask not found for {image_path.name}: {mask_path}"
            )
        with Image.open(mask_path) as image:
            if explicit_mask_paths is None or explicit_mask_kind == "binary":
                sky = np.asarray(image.convert("L")) > 0
            elif explicit_mask_kind == "vkitti2_semantic":
                semantic = np.asarray(image.convert("RGB"))
                sky = np.all(semantic == np.array([90, 200, 255]), axis=-1)
            else:
                raise ValueError(
                    f"Unknown explicit sky-mask kind: {explicit_mask_kind}"
                )

        (crop_left, crop_top, crop_width, crop_height,
         resized_width, resized_height, pad_left, pad_top) = np.rint(
            transform
        ).astype(np.int64)
        sky = sky[
            crop_top:crop_top + crop_height,
            crop_left:crop_left + crop_width,
        ]
        sky = np.asarray(
            Image.fromarray(sky).resize(
                (int(resized_width), int(resized_height)), nearest,
            )
        ).astype(np.bool_)
        if (
            pad_left < 0 or pad_top < 0
            or pad_left + resized_width > target_width
            or pad_top + resized_height > target_height
        ):
            raise ValueError(
                "Sky-mask transform does not fit VGGT prediction shape: "
                f"content={(resized_height, resized_width)}, "
                f"padding={(pad_top, pad_left)}, "
                f"prediction={(target_height, target_width)}"
            )
        output = np.zeros((target_height, target_width), dtype=np.bool_)
        output[
            pad_top:pad_top + resized_height,
            pad_left:pad_left + resized_width,
        ] = sky
        transformed_masks.append(output)
    return np.stack(transformed_masks)


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


def _load_config(config_path):
    """Load an experiment YAML without importing train-time global config."""
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.get("parent_cfg")
    if not parent:
        return config
    # Match train.py/lib.utils.cfg_utils: parent_cfg is interpreted from the
    # process working directory, normally the VAD-GS repository root.  It is not
    # relative to the YAML containing the parent_cfg declaration.
    parent_path = Path(parent).expanduser().resolve()
    base = _load_config(parent_path)

    def merge(dst, src):
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                merge(dst[key], value)
            else:
                dst[key] = value

    merge(base, {key: value for key, value in config.items()
                 if key != "parent_cfg"})
    return base


def _resolve_path(path, workspace):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def _select_dataset_inputs(source_path, data):
    dataset_type = str(data["type"]).lower()
    if dataset_type == "waymo":
        from script.waymo.vggt_visibility_input import select_inputs
    elif dataset_type in {"drivestudio", "nuscenes"}:
        from script.nuscenes.vggt_visibility_input import select_inputs
    else:
        raise ValueError(f"Unsupported VGGT input data.type: {data['type']!r}")
    return select_inputs(source_path, data)


def load_generation_settings(config_path):
    config = _load_config(config_path)
    workspace = Path(config.get("workspace", os.getcwd())).expanduser().resolve()
    source_path = _resolve_path(config["source_path"], workspace)
    data = config["data"]
    generation = data.get("vggt_prediction_generation", {})
    checkpoint = generation.get("checkpoint")
    if not checkpoint:
        raise ValueError("Set data.vggt_prediction_generation.checkpoint")
    vggt_root = generation.get("vggt_root")
    output = data.get("vggt_predictions", "predictions.npz")
    if not output:
        raise ValueError("Set a nonempty data.vggt_predictions output path")
    resolution = int(generation.get("image_resolution", 512))
    if resolution <= 0 or resolution % 16:
        raise ValueError("image_resolution must be positive and divisible by 16")
    image_names, prediction_names, sky_mask_paths, sky_mask_kind = _select_dataset_inputs(source_path, data)
    return {
        "image_names": image_names,
        "prediction_names": prediction_names,
        "sky_mask_paths": sky_mask_paths,
        "sky_mask_kind": sky_mask_kind,
        "source_path": source_path,
        "checkpoint": str(_resolve_path(checkpoint, workspace)),
        "vggt_root": str(_resolve_path(vggt_root, workspace)) if vggt_root else "",
        "output": str(_resolve_path(output, source_path)),
        "device": generation.get("device", "cuda"),
        "image_resolution": resolution,
        "conf_percentile": float(generation.get("conf_percentile", 20.0)),
        "depth_edge_rtol": float(generation.get("depth_edge_rtol", 0.03)),
        "relative_voxel_size": float(generation.get("relative_voxel_size", 0.01)),
        "visualize": bool(generation.get("visualize", False)),
        "max_vis_voxels": int(generation.get("max_vis_voxels", 5000)),
        "max_rays_per_point": int(generation.get("max_rays_per_point", 6)),
        "ray_length": float(generation.get("ray_length", 0.025)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate offline VGGT-Omega pixel visibility for VAD-GS.")
    parser.add_argument("--config", required=True, help="Experiment YAML or a preprocessing YAML inheriting it.")
    parser.add_argument("--dry-run", action="store_true", help="Check input selection and paths without loading the model or writing output.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing prediction cache.")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_generation_settings(args.config)
    output_path = Path(settings["output"])
    if not Path(settings["checkpoint"]).is_file():
        raise FileNotFoundError(f"VGGT checkpoint not found: {settings['checkpoint']}")
    if settings["vggt_root"] and not (Path(settings["vggt_root"]) / "vggt_omega").is_dir():
        raise FileNotFoundError(f"VGGT-Omega source not found: {settings['vggt_root']}")
    print(f"VGGT offline input: {len(settings['image_names'])} images")
    print(f"checkpoint: {settings['checkpoint']}")
    print(f"output: {settings['output']}")
    if args.dry_run:
        print("Dry run complete; no inference or output writes.")
        return
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Prediction cache exists: {output_path}. Use --overwrite to replace it.")
    if settings["vggt_root"]:
        sys.path.insert(0, settings["vggt_root"])
    import torch
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    model = VGGTOmega().to(settings["device"]).eval()
    model.load_state_dict(torch.load(settings["checkpoint"], map_location="cpu"))
    images = load_and_preprocess_images(
        settings["image_names"], image_resolution=settings["image_resolution"],
    ).to(settings["device"])
    with torch.inference_mode():
        raw = model(images)
        extrinsic, intrinsic = encoding_to_camera(raw["pose_enc"], raw["images"].shape[-2:])
    raw["extrinsic"], raw["intrinsic"] = extrinsic, intrinsic
    predictions = {}
    for key in ("depth", "depth_conf", "extrinsic", "intrinsic"):
        value = raw[key].detach().float().cpu().numpy()
        # VGGT-Omega uses one scene batch: [1, views, ...].
        predictions[key] = value[0]
    del raw, images, model
    predictions["image_names"] = np.asarray(settings["prediction_names"])
    target_height, target_width = predictions["depth"].shape[1:3]
    predictions["pixel_transform"] = build_pixel_transforms(
        settings["image_names"], settings["image_resolution"], target_height, target_width,
    )
    sky_masks = load_prediction_sky_masks(
        settings["image_names"], settings["source_path"], predictions["pixel_transform"],
        target_height, target_width, settings["sky_mask_paths"], settings["sky_mask_kind"],
    )
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"],
    )
    voxel_cache = build_vggt_voxel_point_cloud(
        predictions, settings["relative_voxel_size"], settings["conf_percentile"],
        settings["depth_edge_rtol"], sky_masks,
    )
    rotation = predictions["extrinsic"][:, :3, :3]
    translation = predictions["extrinsic"][:, :3, 3]
    voxel_cache["vggt_camera_centers"] = (
        -(np.swapaxes(rotation, 1, 2) @ translation[..., None])[..., 0]
    ).astype(np.float32)
    missing = set(REQUIRED_OUTPUTS).difference(predictions)
    if missing:
        raise KeyError(f"VGGT output is missing required fields: {sorted(missing)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, image_names=predictions["image_names"], depth=predictions["depth"],
        pixel_transform=predictions["pixel_transform"], **voxel_cache,
    )
    counts = np.diff(voxel_cache["vggt_voxel_source_offsets"])
    print(f"Saved {len(settings['image_names'])} views, {len(counts)} voxels to {output_path}; "
          f"voxel size={float(voxel_cache['vggt_voxel_size']):.4g}, "
          f"source views/voxel median={np.median(counts):.1f}")
    if settings["visualize"]:
        # Only allocate visualization data when requested, and cap its size first.
        keep = np.arange(len(counts))
        limit = settings["max_vis_voxels"]
        if limit > 0 and len(keep) > limit:
            keep = np.linspace(0, len(keep) - 1, limit).astype(np.int64)
        visibility = np.zeros((len(keep), len(settings["image_names"])), dtype=np.bool_)
        offsets = voxel_cache["vggt_voxel_source_offsets"]
        sources = voxel_cache["vggt_voxel_source_view_ids"]
        for row, voxel_id in enumerate(keep):
            visibility[row, sources[int(offsets[voxel_id]):int(offsets[voxel_id + 1])]] = True
        points = voxel_cache["vggt_voxel_points"][keep]
        colors = np.full((len(points), 3), 0.72, dtype=np.float32)
        visualize_visibility(points, colors, visibility, predictions["extrinsic"], limit,
                             settings["max_rays_per_point"], settings["ray_length"])


if __name__ == "__main__":
    main()
