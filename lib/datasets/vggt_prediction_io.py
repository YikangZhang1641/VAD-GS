"""Offline pixel -> VGGT pseudo-voxel -> source views, ported from VADT 98540b1."""

import os

import numpy as np


def _image_stem(name):
    if isinstance(name, bytes):
        name = name.decode()
    return os.path.splitext(os.path.basename(str(name)))[0]


def load_vggt_predictions(source_path, image_filenames, predictions_path="predictions.npz"):
    """Load the compact lookup without dataset point clouds, poses or inference.

    Returns per-image pixel indices and a shared voxel/CSR table. Scene view IDs
    follow image_filenames order, including all selected frames and cameras.
    """
    if not predictions_path:
        return {}, None
    if not os.path.isabs(predictions_path):
        predictions_path = os.path.join(source_path, predictions_path)
    if not os.path.isfile(predictions_path):
        print(f"VGGT pixel visibility unavailable: {predictions_path}")
        return {}, None

    with np.load(predictions_path, allow_pickle=False) as predictions:
        voxel_keys = {
            "vggt_voxel_points", "vggt_voxel_source_offsets",
            "vggt_voxel_source_view_ids", "vggt_invalid_voxel_id",
        }
        required = voxel_keys | {"image_names", "vggt_pixel_voxel_ids"}
        missing = required.difference(predictions.files)
        if missing:
            raise KeyError(f"VGGT pixel visibility index missing fields: {sorted(missing)}")
        names = [_image_stem(name) for name in predictions["image_names"]]
        pixel_ids = predictions["vggt_pixel_voxel_ids"]
        transforms = predictions["pixel_transform"] if "pixel_transform" in predictions.files else None
        voxel_info = {key: predictions[key] for key in voxel_keys}
        for key in ("vggt_voxel_ids", "vggt_voxel_size", "vggt_camera_centers"):
            if key in predictions.files:
                voxel_info[key] = predictions[key]

    scene_names = [_image_stem(filename) for filename in image_filenames]
    if len(names) != len(set(names)) or len(scene_names) != len(set(scene_names)):
        raise ValueError("VGGT and scene image filename stems must be unique")
    if pixel_ids.ndim != 3 or len(pixel_ids) != len(names):
        raise ValueError("VGGT pixel indices must have shape [prediction views, H, W]")
    if transforms is not None and transforms.shape != (len(names), 8):
        raise ValueError("VGGT pixel_transform must have shape [prediction views, 8]")
    offsets = voxel_info["vggt_voxel_source_offsets"]
    sources = voxel_info["vggt_voxel_source_view_ids"]
    if (len(offsets) != len(voxel_info["vggt_voxel_points"]) + 1
            or offsets[0] != 0 or offsets[-1] != len(sources)
            or np.any(offsets[1:] < offsets[:-1])):
        raise ValueError("Invalid VGGT voxel source offsets")
    if np.any(sources < 0) or np.any(sources >= len(names)):
        raise ValueError("VGGT source view IDs are outside the prediction view range")

    scene_by_name = {name: view_id for view_id, name in enumerate(scene_names)}
    voxel_info["prediction_to_scene_view"] = np.asarray(
        [scene_by_name.get(name, -1) for name in names], dtype=np.int32,
    )
    if not np.any(voxel_info["prediction_to_scene_view"] >= 0):
        raise ValueError("No VGGT image_names match the selected scene views")
    voxel_info["scene_image_names"] = scene_names
    prediction_by_name = {name: view_id for view_id, name in enumerate(names)}
    per_view = {}
    for name in scene_names:
        prediction_id = prediction_by_name.get(name)
        if prediction_id is not None:
            per_view[name] = {
                "voxel_ids": pixel_ids[prediction_id],
                "pixel_transform": None if transforms is None else transforms[prediction_id],
            }
    print(f"Loaded VGGT pixel visibility for {len(per_view)}/{len(scene_names)} scene views")
    return per_view, voxel_info


def add_vggt_camera_guidance(guidance, image_name, width, height, per_view, voxel_info):
    """Attach a raw VGGT pixel-index map to CameraInfo for resolution adaptation."""
    prediction = per_view.get(image_name)
    if prediction is not None:
        guidance["vggt_voxel_id"] = {
            **prediction,
            "invalid_id": int(voxel_info["vggt_invalid_voxel_id"]),
            "raw_width": width,
            "raw_height": height,
        }


def resize_vggt_voxel_ids(payload, resolution):
    """Map training-image pixels to cached VGGT IDs, honoring crop/resize/pad.

    Uses nearest pixel lookup and returns CPU int64 indices; cropped-out and
    filtered pixels are -1. resolution is (width, height).
    """
    ids = payload["voxel_ids"]
    raw_w, raw_h = payload["raw_width"], payload["raw_height"]
    width, height = resolution
    transform = payload["pixel_transform"]
    if transform is None:
        # Historical VGGT preprocessing crops aspect ratios outside [0.5, 2].
        crop_w = min(raw_w, max(1, int(round(raw_h / 0.5))))
        crop_h = min(raw_h, max(1, int(round(raw_w * 2.0))))
        left, top = (raw_w - crop_w) // 2, (raw_h - crop_h) // 2
        resized_h, resized_w = ids.shape
        pad_left = pad_top = 0
    else:
        left, top, crop_w, crop_h, resized_w, resized_h, pad_left, pad_top = transform

    raw_x = (np.arange(width) + 0.5) * raw_w / width - 0.5
    raw_y = (np.arange(height) + 0.5) * raw_h / height - 0.5
    px = np.floor((raw_x - left + 0.5) * resized_w / crop_w + pad_left).astype(np.int64)
    py = np.floor((raw_y - top + 0.5) * resized_h / crop_h + pad_top).astype(np.int64)
    valid_x = (raw_x >= left) & (raw_x < left + crop_w) & (px >= 0) & (px < ids.shape[1])
    valid_y = (raw_y >= top) & (raw_y < top + crop_h) & (py >= 0) & (py < ids.shape[0])
    restored = ids[np.clip(py, 0, ids.shape[0] - 1)[:, None],
                   np.clip(px, 0, ids.shape[1] - 1)[None, :]].astype(np.int64)
    restored[~(valid_y[:, None] & valid_x[None, :])] = -1
    restored[restored == payload["invalid_id"]] = -1
    return restored


def get_vggt_voxel_source_views(voxel_info, voxel_id):
    """Return selected scene view IDs contributing pixels to a VGGT voxel."""
    if voxel_info is None:
        return np.empty(0, dtype=np.int32)
    voxel_id = int(voxel_id)
    offsets = voxel_info["vggt_voxel_source_offsets"]
    if voxel_id < 0 or voxel_id >= len(offsets) - 1:
        return np.empty(0, dtype=np.int32)
    begin, end = int(offsets[voxel_id]), int(offsets[voxel_id + 1])
    prediction_views = voxel_info["vggt_voxel_source_view_ids"][begin:end].astype(np.int64)
    scene_views = voxel_info["prediction_to_scene_view"][prediction_views]
    return np.unique(scene_views[scene_views >= 0])


def get_vggt_pixel_source_views(viewpoint_cam, voxel_info, u, v):
    """Query (camera view, training-image pixel x/y) -> source scene view IDs.

    This only indexes offline VGGT pseudo-voxels, independently of metric point
    clouds and GrapeTrellis. Invalid/unpredicted pixels return an empty array.
    """
    ids = viewpoint_cam.guidance.get("vggt_voxel_id")
    if ids is None or not (0 <= u < ids.shape[1] and 0 <= v < ids.shape[0]):
        return np.empty(0, dtype=np.int32)
    return get_vggt_voxel_source_views(voxel_info, ids[v, u])
