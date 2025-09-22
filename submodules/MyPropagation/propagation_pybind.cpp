#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// #include "main.h"
#include "Propagation.h"
#include <iostream>
#include <sstream>
#include <cstdarg>

#include <stdio.h>

// void UploadTensorToCudaArray(torch::Tensor tensor, cudaArray_t* dst_array) {
//     TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA device");
//     TORCH_CHECK(tensor.dtype() == torch::kFloat32, "Expected float tensor");
//     TORCH_CHECK(tensor.dim() == 2, "Expected 2D tensor");

//     int rows = tensor.size(0);
//     int cols = tensor.size(1);
//     cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
//     cudaMallocArray(dst_array, &channelDesc, cols, rows);
//     cudaMemcpy2DToArray(*dst_array, 0, 0, tensor.data_ptr(), cols * sizeof(float), cols * sizeof(float), rows, cudaMemcpyDeviceToDevice);
// }

Camera ConvertCameraTensor(torch::Tensor cam_tensor) {
    TORCH_CHECK(cam_tensor.sizes() == std::vector<int64_t>({1, 25}), "Camera tensor must have shape [1, R9+t3+K9+D4]");
    if (!cam_tensor.device().is_cuda()){
        cam_tensor = cam_tensor.cuda();
    }
    // TORCH_CHECK(cam_tensor.device().is_cuda(), "Camera tensor must be on CUDA");

    float temp[25];
    cudaMemcpy(temp, cam_tensor.data_ptr<float>(), sizeof(float) * 25, cudaMemcpyDeviceToHost);

    Camera cam;
    std::copy(temp + 0, temp + 9, cam.R);
    std::copy(temp + 9, temp + 12, cam.t);
    std::copy(temp + 12, temp + 21, cam.K);
    cam.height = static_cast<int>(temp[21]);
    cam.width = static_cast<int>(temp[22]);
    cam.depth_min = temp[23];
    cam.depth_max = temp[24];
    return cam;
}

std::tuple<torch::Tensor, torch::Tensor> MyRunPropagation(
    std::vector<torch::Tensor>& image_tensors,    // [N][H,W] float32, cuda
    std::vector<torch::Tensor>& mask_tensors,     // [N][H,W] uint8, cuda
    std::vector<torch::Tensor>& camera_tensors,   // [N][1,25] float32, cpu
    torch::Tensor& depth_tensor,                  // [H,W] float32, cuda
    torch::Tensor& anchor_depth_tensor,           // [H,W] float32, cuda
    torch::Tensor& normal_guess_tensor            // [H,W,3] float32, cuda
) {
    Propagation pro;
    int N = image_tensors.size();
    pro.num_images = N;

    int H = depth_tensor.size(0);
    int W = depth_tensor.size(1);

    // std::cout << "enter propagation" << std::endl;

    // Upload to cudaArray and bind texture
    for (int i = 0; i < N; ++i) {

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaMallocArray(&pro.cuArray[i], &channelDesc, W, H);

        cudaMemcpyKind copy_kind = image_tensors[i].device().is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
        cudaMemcpy2DToArray(pro.cuArray[i], 0, 0, image_tensors[i].data_ptr<float>(), W*sizeof(float), W*sizeof(float), H, copy_kind);

        // UploadTensorToCudaArray(image_tensors[i], &pro.cuArray[i]);

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = pro.cuArray[i];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaCreateTextureObject(&(pro.texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
    }

    cudaMalloc((void**)&pro.texture_objects_cuda, sizeof(cudaTextureObjects));
    cudaMemcpy(pro.texture_objects_cuda, &pro.texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

    // Camera
    pro.cameras.resize(N);
    for (int i = 0; i < N; ++i) {
        pro.cameras[i] = ConvertCameraTensor(camera_tensors[i]);
    }
    
    // std::cout << pro.cameras[0].K[0] << " " << pro.cameras[0].K[2] << " " << std::endl;

    pro.params.depth_min = pro.cameras[0].depth_min * 0.6f;
    pro.params.depth_max = pro.cameras[0].depth_max * 1.2f;
    if (pro.params.depth_max < 1.0) {
        pro.params.depth_max = 80;
    }

    pro.params.num_images = N;
    pro.params.disparity_min = pro.cameras[0].K[0] * pro.params.baseline / pro.params.depth_max;
    pro.params.disparity_max = pro.cameras[0].K[0] * pro.params.baseline / pro.params.depth_min;

    cudaMalloc((void**)&pro.cameras_cuda, sizeof(Camera) * N);
    cudaMemcpy(pro.cameras_cuda, pro.cameras.data(), sizeof(Camera) * N, cudaMemcpyHostToDevice);

    // std::cout << "2" << std::endl;

    // Other direct inputs
    // int H = depth_tensor.size(0);
    // int W = depth_tensor.size(1);
    int numel = H * W;

    pro.plane_hypotheses_host = new float4[pro.cameras[0].height * pro.cameras[0].width];
    cudaMalloc((void**)&pro.plane_hypotheses_cuda, sizeof(float4) * (pro.cameras[0].height * pro.cameras[0].width));

    pro.costs_host = new float[pro.cameras[0].height * pro.cameras[0].width];
    cudaMalloc((void**)&pro.costs_cuda, sizeof(float) * (pro.cameras[0].height * pro.cameras[0].width));

    cudaMalloc((void**)&pro.rand_states_cuda, sizeof(curandState) * (pro.cameras[0].height * pro.cameras[0].width));
    cudaMalloc((void**)&pro.selected_views_cuda, sizeof(unsigned int) * (pro.cameras[0].height * pro.cameras[0].width));

    cudaMalloc((void**)&pro.depths_cuda, sizeof(float) * numel);
    cudaMemcpyKind depth_copy_kind = depth_tensor.device().is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    cudaMemcpy(pro.depths_cuda, depth_tensor.data_ptr(), sizeof(float) * numel, depth_copy_kind);

    cudaMalloc((void**)&pro.anchor_depth_cuda, sizeof(float) * numel);
    cudaMemcpyKind anchor_copy_kind = anchor_depth_tensor.device().is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    cudaMemcpy(pro.anchor_depth_cuda, anchor_depth_tensor.data_ptr(), sizeof(float) * numel, anchor_copy_kind);

    cudaMalloc((void**)&pro.normals_guess_cuda, sizeof(float3) * numel);
    cudaMemcpyKind normal_copy_kind = normal_guess_tensor.device().is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    cudaMemcpy(pro.normals_guess_cuda, normal_guess_tensor.data_ptr(), sizeof(float3) * numel, normal_copy_kind);

    // std::cout << "3" << std::endl;

    // masks
    pro.h_mask_array = new unsigned char*[N];
    for (int i = 0; i < N; ++i) {
        int rows = mask_tensors[i].size(0);
        int cols = mask_tensors[i].size(1);

        unsigned char* d_mask = NULL;
        cudaMalloc((void**)&d_mask, sizeof(unsigned char) * (rows * cols));
        cudaMemcpyKind mask_copy_kind = mask_tensors[i].device().is_cuda() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
        cudaMemcpy(d_mask, mask_tensors[i].data_ptr(), sizeof(unsigned char) * rows * cols, mask_copy_kind);

        pro.h_mask_array[i] = d_mask;
    }
    cudaMalloc((void**)&pro.d_mask_array, sizeof(unsigned char*) * N);
    cudaMemcpy(pro.d_mask_array, pro.h_mask_array, sizeof(unsigned char*) * N, cudaMemcpyHostToDevice);

    // std::cout << "4" << std::endl;
    // run patchmatch
    pro.RunPatchMatch();


    // torch::Tensor out_depth_normals = torch::zeros({H, W, 4}, torch::kFloat32);  // 最后一维为 4
    // torch::Tensor out_costs = torch::zeros({H, W}, torch::kFloat32);
    
    // auto depth_normal_ptr = reinterpret_cast<float4*>(out_depth_normals.data_ptr<float>());  // 直接操作 float4
    // auto cost_ptr = out_costs.data_ptr<float>();
   

    // for (int row = 0; row < H; ++row) {
    //     for (int col = 0; col < W; ++col) {
    //         int center = row * W + col;
    //         float4 plane = pro.GetPlaneHypothesis(center);
    //         float cost = pro.GetCost(center);
    
    //         // 直接赋值 float4
    //         depth_normal_ptr[center] = plane;
    //         cost_ptr[center] = cost;
    //     }
    // }
    // std::cout << "leave propagation" << std::endl;

    // torch::Tensor dn = out_depth_normals.clone().detach();
    // torch::Tensor c = out_costs.clone().detach();

    // return std::make_tuple(dn.cuda(), c.cuda());

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    torch::Tensor out_depth_normals = torch::from_blob(
        pro.plane_hypotheses_cuda,         // float4* device ptr
        {H, W, 4},                          // shape
        options
    );

    torch::Tensor out_costs = torch::from_blob(
        pro.costs_cuda,                    // float* device ptr
        {H, W},
        options
    );

    return std::make_tuple(out_depth_normals.clone(), out_costs.clone());

}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_propagation", &MyRunPropagation, "Run PatchMatch Propagation with torch inputs");
}
