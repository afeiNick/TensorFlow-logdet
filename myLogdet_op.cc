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

//#if GOOGLE_CUDA

#include "myLogdet_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"

using namespace tensorflow;
    
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("MyLogdet").Input("bottom_data: float").Output("logdet: float").Output("top_data: float");

class MyLogdetOp : public OpKernel {
public:
    explicit MyLogdetOp(OpKernelConstruction* context) : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {
        
        const Tensor& bottom_tensor = context->input(0);
        
        // require input tensor to be 3 dimensional
        OP_REQUIRES(context, bottom_tensor.dims() == 3,
                    errors::InvalidArgument("tensor_in must be 3-dimensional"));
        
        OP_REQUIRES(context, bottom_tensor.dim_size(1) == bottom_tensor.dim_size(2),
                    errors::InvalidArgument("last 2 dimensions of tensor_in must be square"));
        
        TensorShape logdet_shape = bottom_tensor.shape();
        int dimension = logdet_shape.dims();
        logdet_shape.set_dim(dimension-2, 1);
        logdet_shape.set_dim(dimension-1, 1);
        
        Tensor* logdet_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, logdet_shape, &logdet_tensor));        
        
        Tensor* top_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, bottom_tensor.shape(), &top_tensor));

        int batch_size = bottom_tensor.dim_size(0);
        int mat_size = bottom_tensor.dim_size(1);
        const GPUDevice& d = context->eigen_gpu_device();

        MyLogdetKernelLauncher(batch_size,
                               mat_size,
                               bottom_tensor.flat<float>().data(),
                               logdet_tensor->flat<float>().data(),
                               top_tensor->flat<float>().data(),
                               d);
        
        
    }
};
    

// register op
REGISTER_KERNEL_BUILDER(Name("MyLogdet").Device(DEVICE_GPU), MyLogdetOp);

//#endif // GOOGLE_CUDA

