#include "Propagation.h"

#include <cstdarg>

void StringAppendV(std::string* dst, const char* format, va_list ap) {
  // First try with a small fixed size buffer.
  static const int kFixedBufferSize = 1024;
  char fixed_buffer[kFixedBufferSize];

  // It is possible for methods that use a va_list to invalidate
  // the data in it upon use.  The fix is to make a copy
  // of the structure before using it and use that copy instead.
  va_list backup_ap;
  va_copy(backup_ap, ap);
  int result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
  va_end(backup_ap);

  if (result < kFixedBufferSize) {
    if (result >= 0) {
      // Normal case - everything fits.
      dst->append(fixed_buffer, result);
      return;
    }

#ifdef _MSC_VER
    // Error or MSVC running out of space.  MSVC 8.0 and higher
    // can be asked about space needed with the special idiom below:
    va_copy(backup_ap, ap);
    result = vsnprintf(nullptr, 0, format, backup_ap);
    va_end(backup_ap);
#endif

    if (result < 0) {
      // Just an error.
      return;
    }
  }

  // Increase the buffer size to the size requested by vsnprintf,
  // plus one for the closing \0.
  const int variable_buffer_size = result + 1;
  std::unique_ptr<char> variable_buffer(new char[variable_buffer_size]);

  // Restore the va_list before we use it again.
  va_copy(backup_ap, ap);
  result =
      vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
  va_end(backup_ap);

  if (result >= 0 && result < variable_buffer_size) {
    dst->append(variable_buffer.get(), result);
  }
}

std::string StringPrintf(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  std::string result;
  StringAppendV(&result, format, ap);
  va_end(ap);
  return result;
}

void CudaSafeCall(const cudaError_t error, const std::string& file,
                  const int line) {
  if (error != cudaSuccess) {
    std::cerr << StringPrintf("%s in %s at line %i", cudaGetErrorString(error),
                              file.c_str(), line)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

void CudaCheckError(const char* file, const int line) {
  cudaError error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << StringPrintf("cudaCheckError() failed at %s:%i : %s", file,
                              line, cudaGetErrorString(error))
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  error = cudaDeviceSynchronize();
  if (cudaSuccess != error) {
    std::cerr << StringPrintf("cudaCheckError() with sync failed at %s:%i : %s",
                              file, line, cudaGetErrorString(error))
              << std::endl;
    std::cerr
        << "This error is likely caused by the graphics card timeout "
           "detection mechanism of your operating system. Please refer to "
           "the FAQ in the documentation on how to solve this problem."
        << std::endl;
    exit(EXIT_FAILURE);
  }
}

Propagation::Propagation() {}

Propagation::~Propagation()
{
    delete[] plane_hypotheses_host;
    delete[] costs_host;

    for (int i = 0; i < num_images; ++i) {
        cudaDestroyTextureObject(texture_objects_host.images[i]);
        cudaFreeArray(cuArray[i]);
    }
    cudaFree(texture_objects_cuda);
    cudaFree(cameras_cuda);
    cudaFree(plane_hypotheses_cuda);
    cudaFree(costs_cuda);
    cudaFree(rand_states_cuda);
    cudaFree(selected_views_cuda);
    cudaFree(depths_cuda); 
    cudaFree(anchor_depth_cuda); 
    cudaFree(normals_guess_cuda); 

    for (int i = 0; i < num_images; ++i) {
        cudaFree(h_mask_array[i]); 
    }
    delete[] h_mask_array;
    cudaFree(d_mask_array); 
    
    if (params.geom_consistency) {
        for (int i = 0; i < num_images; ++i) {
            cudaDestroyTextureObject(texture_depths_host.images[i]);
            cudaFreeArray(cuDepthArray[i]);
        }
        cudaFree(texture_depths_cuda);
    }
}

Camera ReadCamera(const std::string &cam_path)
{
    Camera camera;
    std::ifstream file(cam_path);

    std::string line;
    file >> line;

    for (int i = 0; i < 3; ++i) {
        file >> camera.R[3 * i + 0] >> camera.R[3 * i + 1] >> camera.R[3 * i + 2] >> camera.t[i];
    }

    float tmp[4];
    file >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3];
    file >> line;

    for (int i = 0; i < 3; ++i) {
        file >> camera.K[3 * i + 0] >> camera.K[3 * i + 1] >> camera.K[3 * i + 2];
    }

    float depth_num;
    float interval;
    file >> camera.depth_min >> interval >> depth_num >> camera.depth_max;

    return camera;
}

float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera)
{
    float3 pointX;
    float3 tmpX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    // Rotation
    tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
    tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
    tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;

    // Transformation
    float3 C;
    C.x = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
    C.y = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
    C.z = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
    pointX.x = tmpX.x + C.x;
    pointX.y = tmpX.y + C.y;
    pointX.z = tmpX.z + C.z;

    return pointX;
}

void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth)
{
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

// int readDepthDmb(const std::string file_path, cv::Mat_<float> &depth)
// {
//     FILE *inimage;
//     inimage = fopen(file_path.c_str(), "rb");
//     if (!inimage){
//         std::cout << "Error opening file " << file_path << std::endl;
//         return -1;
//     }

//     int32_t type, h, w, nb;

//     type = -1;

//     fread(&type,sizeof(int32_t),1,inimage);
//     fread(&h,sizeof(int32_t),1,inimage);
//     fread(&w,sizeof(int32_t),1,inimage);
//     fread(&nb,sizeof(int32_t),1,inimage);

//     if (type != 1) {
//         fclose(inimage);
//         return -1;
//     }

//     int32_t dataSize = h*w*nb;

//     depth = cv::Mat::zeros(h,w,CV_32F);
//     fread(depth.data,sizeof(float),dataSize,inimage);

//     fclose(inimage);
//     return 0;
// }

// int writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth)
// {
//     FILE *outimage;
//     outimage = fopen(file_path.c_str(), "wb");
//     if (!outimage) {
//         std::cout << "Error opening file " << file_path << std::endl;
//     }

//     int32_t type = 1;
//     int32_t h = depth.rows;
//     int32_t w = depth.cols;
//     int32_t nb = 1;

//     fwrite(&type,sizeof(int32_t),1,outimage);
//     fwrite(&h,sizeof(int32_t),1,outimage);
//     fwrite(&w,sizeof(int32_t),1,outimage);
//     fwrite(&nb,sizeof(int32_t),1,outimage);

//     float* data = (float*)depth.data;

//     int32_t datasize = w*h*nb;
//     fwrite(data,sizeof(float),datasize,outimage);

//     fclose(outimage);
//     return 0;
// }

// int readNormalDmb (const std::string file_path, cv::Mat_<cv::Vec3f> &normal)
// {
//     FILE *inimage;
//     inimage = fopen(file_path.c_str(), "rb");
//     if (!inimage){
//         std::cout << "Error opening file " << file_path << std::endl;
//         return -1;
//     }

//     int32_t type, h, w, nb;

//     type = -1;

//     fread(&type,sizeof(int32_t),1,inimage);
//     fread(&h,sizeof(int32_t),1,inimage);
//     fread(&w,sizeof(int32_t),1,inimage);
//     fread(&nb,sizeof(int32_t),1,inimage);

//     if (type != 1) {
//         fclose(inimage);
//         return -1;
//     }

//     int32_t dataSize = h*w*nb;

//     normal = cv::Mat::zeros(h,w,CV_32FC3);
//     fread(normal.data,sizeof(float),dataSize,inimage);

//     fclose(inimage);
//     return 0;
// }

// int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal)
// {
//     FILE *outimage;
//     outimage = fopen(file_path.c_str(), "wb");
//     if (!outimage) {
//         std::cout << "Error opening file " << file_path << std::endl;
//     }

//     int32_t type = 1;
//     int32_t h = normal.rows;
//     int32_t w = normal.cols;
//     int32_t nb = 3;

//     fwrite(&type,sizeof(int32_t),1,outimage);
//     fwrite(&h,sizeof(int32_t),1,outimage);
//     fwrite(&w,sizeof(int32_t),1,outimage);
//     fwrite(&nb,sizeof(int32_t),1,outimage);

//     float* data = (float*)normal.data;

//     int32_t datasize = w*h*nb;
//     fwrite(data,sizeof(float),datasize,outimage);

//     fclose(outimage);
//     return 0;
// }

void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc)
{
    std::cout << "store 3D points to ply file" << std::endl;

    FILE *outputPly;
    outputPly=fopen(plyFilePath.c_str(), "wb");

    /*write header*/
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n",pc.size());
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
    fprintf(outputPly, "property float nx\n");
    fprintf(outputPly, "property float ny\n");
    fprintf(outputPly, "property float nz\n");
    fprintf(outputPly, "property uchar red\n");
    fprintf(outputPly, "property uchar green\n");
    fprintf(outputPly, "property uchar blue\n");
    fprintf(outputPly, "end_header\n");

    //write data
#pragma omp parallel for
    for(size_t i = 0; i < pc.size(); i++) {
        const PointList &p = pc[i];
        float3 X = p.coord;
        const float3 normal = p.normal;
        const float3 color = p.color;
        const char b_color = (int)color.x;
        const char g_color = (int)color.y;
        const char r_color = (int)color.z;

        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
#pragma omp critical
        {
            fwrite(&X.x,      sizeof(X.x), 1, outputPly);
            fwrite(&X.y,      sizeof(X.y), 1, outputPly);
            fwrite(&X.z,      sizeof(X.z), 1, outputPly);
            fwrite(&normal.x, sizeof(normal.x), 1, outputPly);
            fwrite(&normal.y, sizeof(normal.y), 1, outputPly);
            fwrite(&normal.z, sizeof(normal.z), 1, outputPly);
            fwrite(&r_color,  sizeof(char), 1, outputPly);
            fwrite(&g_color,  sizeof(char), 1, outputPly);
            fwrite(&b_color,  sizeof(char), 1, outputPly);
        }

    }
    fclose(outputPly);
}

static float GetDisparity(const Camera &camera, const int2 &p, const float &depth)
{
    float point3D[3];
    point3D[0] = depth * (p.x - camera.K[2]) / camera.K[0];
    point3D[1] = depth * (p.y - camera.K[5]) / camera.K[4];
    point3D[2] = depth;

    return std::sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1] + point3D[2] * point3D[2]);
}

void Propagation::SetGeomConsistencyParams()
{
    params.geom_consistency = true;
    params.max_iterations = 2;
}

// void Propagation::InuputInitialization(const std::string &dense_folder, const Problem &problem)
// {
//     // std::cout << "test " << std::endl;

//     images.clear();
//     cameras.clear();
//     masks.clear();
//     normals.clear();
//     depths.clear();

//     // RGB images
//     std::string image_folder = dense_folder + std::string("/images");
//     std::stringstream image_path;
//     image_path << image_folder << "/" << problem.ref_image_id << ".jpg";
//     cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
//     cv::Mat image_float;
//     image_uint.convertTo(image_float, CV_32FC1);
//     images.push_back(image_float);


//     // zyk: uint8 mask here
//     std::string mask_folder = dense_folder + std::string("/masks"); // zyk 
//     std::stringstream mask_path;
//     mask_path << mask_folder << "/" << problem.ref_image_id << ".png";
//     cv::Mat_<uint8_t> mask = cv::imread(mask_path.str(), cv::IMREAD_GRAYSCALE);
//     masks.push_back(mask);


//     // Cam intrinsic & extrinsic
//     std::string cam_folder = dense_folder + std::string("/cams");
//     std::stringstream cam_path;
//     cam_path << cam_folder << "/" << problem.ref_image_id << ".txt";
//     Camera camera = ReadCamera(cam_path.str());
//     camera.height = image_float.rows;
//     camera.width = image_float.cols;
//     cameras.push_back(camera);


//     // Depth
//     std::string depth_folder = dense_folder + std::string("/depths");
//     std::stringstream depth_path;
//     depth_path << depth_folder << "/" << problem.ref_image_id << ".dmb";
//     cv::Mat_<float> ref_depth;
//     readDepthDmb(depth_path.str(), ref_depth);

//     // cv::Mat ref_depth = cv::imread(depth_path.str(), cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
//     // ref_depth.convertTo(ref_depth, CV_32FC1);
//     // ref_depth = ref_depth / 100.;  //scale to metric m // need or not?
//     depths.push_back(ref_depth); // rendered depth, dense but not accurate.

//     std::stringstream anchor_path; // zyk: my anchor here
//     anchor_path << depth_folder << "/" << problem.ref_image_id << ".raw";
//     std::ifstream file(anchor_path.str(), std::ios::binary);
//     if (!file.is_open()) {
//         throw std::runtime_error("Could not open raw file!");
//     }
//     cv::Mat anchor_depth = cv::Mat::zeros(camera.height, camera.width, CV_32FC1);
//     file.read(reinterpret_cast<char*>(anchor_depth.data), camera.height * camera.width * sizeof(float)); // sparse but absolute prior, without occlusion
//     depths.push_back(anchor_depth);

//     // zyk: Normal for initial guess 
//     std::string normal_folder = dense_folder + std::string("/normals");
//     std::stringstream normal_path;
//     normal_path << normal_folder << "/" << problem.ref_image_id << ".png";
//     // cv::Mat normal_rgb = cv::Mat::zeros(camera.height, camera.width, CV_8UC3); // = cv::imread(normal_path.str(), cv::IMREAD_COLOR);
//     cv::Mat normal_rgb = cv::imread(normal_path.str(), cv::IMREAD_COLOR);
//     // cv::cvtColor(normal_rgb, normal_rgb, cv::COLOR_RGB2BGR);
//     cv::Mat normal_float;
//     normal_rgb.convertTo(normal_float, CV_32FC3);

//     normal_float = normal_float / 255 * 2 - 1;

//     // cv::imshow("image", normal_float);
//     // cv::waitKey(0);    
//     // normal_float = -normal_float; // 0->1
//     normals.push_back(normal_float); // zyk: todo: 坐标方向似乎不对
//     // std::cout << "normals " << normal_float.ptr<float>()[100] << std::endl;

//     size_t num_src_images = problem.src_image_ids.size();
//     for (size_t i = 0; i < num_src_images; ++i) {
//         std::stringstream image_path;
//         image_path << image_folder << "/" << problem.src_image_ids[i] << ".jpg";
//         cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
//         cv::Mat image_float;
//         image_uint.convertTo(image_float, CV_32FC1);
//         images.push_back(image_float);
//         std::stringstream cam_path;
//         cam_path << cam_folder << "/" << problem.src_image_ids[i] << ".txt";
//         Camera camera = ReadCamera(cam_path.str());
//         camera.height = image_float.rows;
//         camera.width = image_float.cols;
//         cameras.push_back(camera);

//         // zyk: uint8 mask here
//         std::stringstream mask_path;
//         mask_path << mask_folder << "/" << problem.src_image_ids[i] << ".png";
//         cv::Mat_<uint8_t> mask = cv::imread(mask_path.str(), cv::IMREAD_GRAYSCALE);
//         masks.push_back(mask);
//     }

//     // Scale cameras and images
//     for (size_t i = 0; i < images.size(); ++i) {
//         if (images[i].cols <=  params.max_image_size && images[i].rows <= params.max_image_size) {
//             continue;
//         }

//         const float factor_x = static_cast<float>(params.max_image_size) / images[i].cols;
//         const float factor_y = static_cast<float>(params.max_image_size) / images[i].rows;
//         const float factor = std::min(factor_x, factor_y);

//         const int new_cols = std::round(images[i].cols * factor);
//         const int new_rows = std::round(images[i].rows * factor);

//         const float scale_x = new_cols / static_cast<float>(images[i].cols);
//         const float scale_y = new_rows / static_cast<float>(images[i].rows);

//         cv::Mat_<float> scaled_image_float;
//         cv::resize(images[i], scaled_image_float, cv::Size(new_cols,new_rows), 0, 0, cv::INTER_LINEAR);
//         images[i] = scaled_image_float.clone();

//         cameras[i].K[0] *= scale_x;
//         cameras[i].K[2] *= scale_x;
//         cameras[i].K[4] *= scale_y;
//         cameras[i].K[5] *= scale_y;
//         cameras[i].height = scaled_image_float.rows;
//         cameras[i].width = scaled_image_float.cols;
//     }

//     params.depth_min = cameras[0].depth_min * 0.6f;
//     params.depth_max = cameras[0].depth_max * 1.2f;
//     if (params.depth_max < 1.0) {
//         params.depth_max = 80;
//     }

//     // std::cout << "depth range: " << params.depth_min << " " << params.depth_max << std::endl;
//     params.num_images = (int)images.size();
//     // std::cout << "num images: " << params.num_images << std::endl;
//     params.disparity_min = cameras[0].K[0] * params.baseline / params.depth_max;
//     params.disparity_max = cameras[0].K[0] * params.baseline / params.depth_min;

// }



// void Propagation::CudaSpaceInitialization()
// {
//     num_images = (int)images.size();

//     for (int i = 0; i < num_images; ++i) {
//         int rows = images[i].rows;
//         int cols = images[i].cols;
//         // std::cout << "image shape: " << rows << " " << rows << std::endl;

//         cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
//         cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);
//         cudaMemcpy2DToArray(cuArray[i], 0, 0, images[i].ptr<float>(), images[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice);

//         struct cudaResourceDesc resDesc;
//         memset(&resDesc, 0, sizeof(cudaResourceDesc));
//         resDesc.resType = cudaResourceTypeArray;
//         resDesc.res.array.array = cuArray[i];

//         struct cudaTextureDesc texDesc;
//         memset(&texDesc, 0, sizeof(cudaTextureDesc));
//         texDesc.addressMode[0] = cudaAddressModeWrap;
//         texDesc.addressMode[1] = cudaAddressModeWrap;
//         texDesc.filterMode = cudaFilterModeLinear;
//         texDesc.readMode  = cudaReadModeElementType;
//         texDesc.normalizedCoords = 0;

//         cudaCreateTextureObject(&(texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
//     }

//     cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects));
//     cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

//     cudaMalloc((void**)&cameras_cuda, sizeof(Camera) * (num_images));
//     cudaMemcpy(cameras_cuda, &cameras[0], sizeof(Camera) * (num_images), cudaMemcpyHostToDevice);

//     plane_hypotheses_host = new float4[cameras[0].height * cameras[0].width];
//     cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4) * (cameras[0].height * cameras[0].width));

//     costs_host = new float[cameras[0].height * cameras[0].width];
//     cudaMalloc((void**)&costs_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));

//     cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * (cameras[0].height * cameras[0].width));
//     cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int) * (cameras[0].height * cameras[0].width));

//     cudaMalloc((void**)&depths_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));
//     cudaMemcpy(depths_cuda, depths[0].ptr<float>(),  sizeof(float) * cameras[0].height * cameras[0].width, cudaMemcpyHostToDevice);
    
//     cudaMalloc((void**)&anchor_depth_cuda, sizeof(float) * (cameras[0].height * cameras[0].width)); // zyk: depth from lidar: absolute, sparse, view specified.
//     cudaMemcpy(anchor_depth_cuda, depths[1].ptr<float>(),  sizeof(float) * cameras[0].height * cameras[0].width, cudaMemcpyHostToDevice);

//     cudaMalloc((void**)&normals_guess_cuda, sizeof(float3) * (cameras[0].height * cameras[0].width)); // zyk: depth from lidar: absolute, sparse, view specified.
//     cudaMemcpy(normals_guess_cuda, normals[0].ptr<float>(),  sizeof(float3) * cameras[0].height * cameras[0].width, cudaMemcpyHostToDevice);

//     // std::cout << "test " << depths[0].ptr<float>()[1000] << std::endl;


//     // zyk: new mask channel 
//     h_mask_array = new unsigned char*[num_images];
//     for (int i = 0; i < num_images; ++i) {
//         int rows = masks[i].rows;
//         int cols = masks[i].cols;

//         unsigned char* d_mask = NULL;
//         cudaMalloc((void**)&d_mask, sizeof(unsigned char) * (rows * cols)); // zyk: depth from lidar: absolute, sparse, view specified.
//         cudaMemcpy(d_mask, masks[i].data, sizeof(unsigned char) * rows * cols, cudaMemcpyHostToDevice);

//         h_mask_array[i] = d_mask;
//     }

//     cudaMalloc((void**)&d_mask_array, sizeof(unsigned char*) * num_images);
//     cudaMemcpy(d_mask_array, h_mask_array, sizeof(unsigned char*) * num_images, cudaMemcpyHostToDevice); // temp trial
//     delete[] h_mask_array;


//     // // zyk: visualize 是否读取正确
// 	// unsigned char** output = new unsigned char*[num_images];
// 	// cudaMemcpy(output, d_mask_array, sizeof(unsigned char*)*num_images, cudaMemcpyDeviceToHost);
//     // for (int i = 0; i < num_images; ++i) {
//     //     int rows = masks[i].rows;
//     //     int cols = masks[i].cols;
//     //     cv::Mat dst0 = cv::Mat::zeros(rows, cols, CV_32FC2);
//     //     cudaMemcpy(dst0.data, output[i], sizeof(unsigned char)*rows*cols, cudaMemcpyDeviceToHost);
//     //     int tmp = 0;
//     //     unsigned char *input = (unsigned char*)(dst0.data);
//     //     for (int j = 0; j < rows*cols; j++) {
//     //         if (input[j] > 128) {
//     //             tmp ++;
//     //         }
//     //     }
//     //     std::cout << "mask " << i << " : " << tmp << std::endl;
//     // }
//     // delete[] output;
// }

int Propagation::GetReferenceImageWidth()
{
    return cameras[0].width;
}

int Propagation::GetReferenceImageHeight()
{
    return cameras[0].height;
}

// cv::Mat Propagation::GetReferenceImage()
// {
//     return images[0];
// }

float4 Propagation::GetPlaneHypothesis(const int index)
{
    return plane_hypotheses_host[index];
}

float Propagation::GetCost(const int index)
{
    return costs_host[index];
}

void Propagation::SetPatchSize(int patch_size)
{
    params.patch_size = patch_size;
}

int Propagation::GetPatchSize()
{
    return params.patch_size;
}


