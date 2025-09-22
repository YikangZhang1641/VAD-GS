import os 
import torch
import patchmatch_cuda
import numpy as np 
import cv2
import struct 
import matplotlib.pyplot as plt 

N = 5
H, W = 640, 960
device = "cuda"
cache_path = "/home/zyk/Projects/_DL/_Reconstruction/street_gaussians_new_start/cache/"

def readDepthDmb(file_path):
    inimage = open(file_path, "rb")
    if not inimage:
        print("Error opening file", file_path)
        return -1

    type = -1

    type = struct.unpack("i", inimage.read(4))[0]
    h = struct.unpack("i", inimage.read(4))[0]
    w = struct.unpack("i", inimage.read(4))[0]
    nb = struct.unpack("i", inimage.read(4))[0]

    if type != 1:
        inimage.close()
        return -1

    dataSize = h * w * nb

    depth = np.zeros((h, w), dtype=np.float32)
    depth_data = np.frombuffer(inimage.read(dataSize * 4), dtype=np.float32)
    depth_data = depth_data.reshape((h, w))
    np.copyto(depth, depth_data)

    inimage.close()
    return depth


# 伪造输入
image_tensors = []
for i in range(5):
    path = cache_path + "images/" + str(i) + ".jpg"
    img = cv2.imread(path) / 255.0
    img = np.mean(img, axis=2)
    image_tensors.append(torch.from_numpy(img).float().contiguous().to("cuda"))


mask_tensors = []
for i in range(5):
    path = cache_path + "masks/" + str(i) + ".png"
    msk = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    msk[msk<128] = 0
    msk[msk>=128]=255
    mask_tensors.append(torch.from_numpy(msk).byte().contiguous().to("cuda"))

camera_tensors = []
for i in range(5):
    path = cache_path + "cams/" + str(i) + ".txt"
    with open(path, "r") as f:
        lines = f.readlines()
        Rt = np.vstack([np.array(lines[j][:-2].split(" "), dtype=np.float32) for j in [1,2,3]])
        src_ext = torch.from_numpy(Rt)
        # R = Rt[:3,:3]
        # t = Rt[:3, 3]

        K = np.vstack([np.array(lines[j][:-2].split(" "), dtype=np.float32) for j in [7,8,9]])
        src_K = torch.from_numpy(K)

        img_H, img_W = 640, 960
        depth_min, depth_max = 0.1, 80

        row = torch.concat([src_ext[:3,:3].reshape(-1), src_ext[:3,3].reshape(-1), src_K.reshape(-1), torch.tensor([img_H, img_W, depth_min, depth_max])])
        camera_tensors.append(row.float().unsqueeze(0).contiguous().to("cpu"))
        print(row)

path = cache_path + "depths/0.dmb"
depth = readDepthDmb(path)
depth_tensor = torch.from_numpy(depth).to(device)

anchor_depth_tensor = torch.zeros((H, W), device=device, dtype=torch.float32)

path = cache_path + "normals/0.png"
normal = cv2.imread(path) / 255 * 2 - 1

normal_guess_tensor = torch.from_numpy(normal).float().to(device)


d = torch.load("/home/zyk/Projects/_DL/_Reconstruction/street_gaussians_new_start/text.pth")
image_tensors_d = d["image_tensors"]
mask_tensors_d = d["mask_tensors"]
camera_tensors_d = d["camera_tensors"]
depth_tensor_d = d["depth_tensor"]
anchor_depth_tensor_d = d["anchor_depth_tensor"]
normal_guess_tensor_d = d["normal_guess_tensor"]

# # 调用
# depth_out, normal_out, cost_out = patchmatch_cuda.run_propagation(
cnt = 0
while True:
    plane_out, cost_out = patchmatch_cuda.run_propagation(
        image_tensors_d,
        mask_tensors_d,
        camera_tensors_d,
        depth_tensor_d,
        anchor_depth_tensor_d * 0,
        normal_guess_tensor_d
    )
    cnt += 1
    print("iter", cnt)

depth_out = plane_out[..., 3]
normal_out = plane_out[..., :3]

m = torch.any(torch.abs(normal_out) > 1, dim=2)
normal_out[m] = 0
plt.imshow(normal_out.cpu().detach()/2+0.5)


print("depth_tensor device:", depth_tensor.device)



# print("Depth output shape:", depth_out.shape)
print("depth device: ", depth_out.device)
# print("depth dtype: ", depth_out.dtype)

# print("Normal output shape:", normal_out.shape)
print("normal device: ", normal_out.device)
# print("normal dtype: ", normal_out.dtype)

cv2.imshow("depth_out", depth_out.cpu().detach().numpy() / depth_out.max().item())
cv2.imshow("normal_out", normal_out.cpu().detach().numpy()/2+0.5)
key = cv2.waitKey(0)