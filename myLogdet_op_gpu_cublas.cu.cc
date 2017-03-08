/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/


#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#define MAX_T 1024

#include "myLogdet_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "/usr/local/cuda-8.0/include/cuda_runtime.h"
#include "/usr/local/cuda-8.0/include/cublas_v2.h"


namespace tensorflow {
    
    typedef Eigen::GpuDevice GPUDevice;
    
    /* compute log det */
    __global__ void ComputeLogdet(const int batch_size,
                                  const int mat_size,
                                  float* top_logdet,
                                  float* top_data)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < batch_size)
        {
            int offset = i * mat_size * mat_size;
            float* top_data_after_offset = top_data + offset;
            top_logdet[i] = 0.0;
            for (int j = 0; j < mat_size; ++j)
                top_logdet[i] += log(abs(top_data_after_offset[j + mat_size * j]));
        }
    }
    
    __global__ void AssignPointer(const int batch_size,
                                  const int mat_size,
                                  float* top_data,
                                  float** Aarray)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < batch_size)
        {
            int offset = i * mat_size * mat_size;
            Aarray[i] = top_data + offset;
        }
    }
    
    void MyLogdetKernelLauncher(const int batch_size,
                                const int mat_size,
                                const float* bottom_data,
                                float* top_logdet,
                                float* top_data, // top cholesky decomposed matrix
                                const GPUDevice& d)
    {
        cublasHandle_t cnpHandle;
        cublasStatus_t status = cublasCreate(&cnpHandle);
  
        float* LU_data;
        cudaMalloc(&LU_data, batch_size * mat_size * mat_size * sizeof(float));
        
        status = cublasScopy(cnpHandle,
                             batch_size * mat_size * mat_size,
                             bottom_data, 1,
                             LU_data, 1);

        
        float **Aarray;
        cudaMalloc((void**)&Aarray, batch_size * sizeof(float*));
        AssignPointer<<< (batch_size + MAX_T - 1) / MAX_T, MAX_T, 0, d.stream() >>>(batch_size, mat_size, LU_data, Aarray);
        
        int* PivotArray;
        cudaMalloc(&PivotArray, batch_size * mat_size * sizeof(int));
        
        int* InfoArray;
        cudaMalloc(&InfoArray, batch_size * sizeof(int));
        
        status = cublasSgetrfBatched(cnpHandle,
                                     mat_size,
                                     Aarray,
                                     mat_size,
                                     PivotArray,
                                     InfoArray,
                                     batch_size);
        
        ComputeLogdet<<< (batch_size + MAX_T - 1) / MAX_T, MAX_T, 0, d.stream() >>>(batch_size, mat_size, top_logdet, LU_data);
        
        float **Carray;
        cudaMalloc((void**)&Carray, batch_size * sizeof(float*));
        AssignPointer<<< (batch_size + MAX_T - 1) / MAX_T, MAX_T, 0, d.stream() >>>(batch_size, mat_size, top_data, Carray);
        
        status = cublasSgetriBatched(cnpHandle,
                                     mat_size,
                                     (const float**) Aarray,
                                     mat_size,
                                     PivotArray,
                                     Carray,
                                     mat_size,
                                     InfoArray,
                                     batch_size);
        
        cublasDestroy(cnpHandle);
        cudaFree(LU_data);
        cudaFree(Aarray);
        cudaFree(Carray);
        cudaFree(PivotArray);
        cudaFree(InfoArray);
    }
}  // namespace tensorflow


#endif // GOOGLE_CUDA