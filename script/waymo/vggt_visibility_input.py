"""Waymo inputs for the dataset-neutral VGGT visibility generator."""

import re
from pathlib import Path


def select_inputs(source_path, data):
    """Select ``images/{frame}_{camera}`` views and dataset sky masks."""
    source_path = Path(source_path)
    cameras = {int(camera) for camera in data["cameras"]}
    selected_frames = data["selected_frames"]
    frame_range = tuple(map(int, selected_frames))
    image_dir = source_path / "images"
    images = []
    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        match = re.fullmatch(r"(\d+)_(\d+)", image_path.stem)
        if match is None:
            raise ValueError(f"Invalid Waymo image name: {image_path.name}")
        frame, camera = map(int, match.groups())
        if camera in cameras and frame_range[0] <= frame <= frame_range[1]:
            images.append(image_path.resolve())
    images = sorted(images)
    if not images:
        raise ValueError(
            f"No Waymo images selected from {image_dir} for "
            f"frames={selected_frames}, cameras={sorted(cameras)}"
        )
    image_names = [str(path) for path in images]
    sky_dir = next(
        (
            source_path / name for name in ("sky_mask", "sky_masks")
            if (source_path / name).is_dir()
        ),
        None,
    )
    if sky_dir is None:
        raise FileNotFoundError(
            f"Waymo sky-mask directory not found under {source_path}"
        )
    sky_masks = []
    for image_path in images:
        # Both original frame names and the legacy shortened names are in use.
        stems = [image_path.stem]
        if len(image_path.stem.split('_')[0]) == 6:
            stems.append(image_path.stem[3:])
        candidates = [sky_dir / f"{stem}.png" for stem in stems]
        mask_path = next((path for path in candidates if path.is_file()), None)
        if mask_path is None:
            raise FileNotFoundError(
                f"Waymo sky mask not found for {image_path.name}: {candidates}"
            )
        sky_masks.append(str(mask_path.resolve()))
    return image_names, image_names, sky_masks, "binary"
