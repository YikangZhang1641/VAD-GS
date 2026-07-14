import torch
from glob import glob
import matplotlib.pyplot as plt

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

checkpoint_path = "../checkpoints/vggt_omega_1b_512.pt"
# image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]
# image_names = sorted(glob("/home/zyk/Ext/Projects/vadt/data/waymo/training/021/images/*_0.*"))[:60]
# image_names = sorted(glob("/home/zyk/Ext/Projects/vadt/data/waymo/training/021/images/00000[0-9]_[0-2].png"))
image_names = sorted(glob("/home/zyk/Ext/Projects/vadt/data/waymo/training/021/images/0000*_[0-2].png"))[:183]
# image_names = sorted(glob("../examples/sculpture/*"))


model = VGGTOmega().to("cuda").eval()
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

images = load_and_preprocess_images(image_names, image_resolution=512).to("cuda")

with torch.inference_mode():
    predictions = model(images)

extrinsics, intrinsics = encoding_to_camera(
    predictions["pose_enc"],
    predictions["images"].shape[-2:],
)

depth = predictions["depth"]
depth_conf = predictions["depth_conf"]
camera_and_register_tokens = predictions["camera_and_register_tokens"]
camera_tokens = camera_and_register_tokens[:, :, :1]
registers = camera_and_register_tokens[:, :, 1:]