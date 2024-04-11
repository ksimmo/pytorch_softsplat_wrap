#include <torch/extension.h>
#include <vector>

#include <math.h>

//forward function
template <typename scalar_t>
__global__ void __launch_bounds__(512) splat_forward_kernel(const int n, 
                        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> source, 
                        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> flow,
                        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> target)
{ 
    for (int index = (blockIdx.x * blockDim.x) + threadIdx.x; index< n; index += blockDim.x * gridDim.x) 
    {
        const int B = (index/target.size(3)/target.size(2)/target.size(1))%target.size(0);
        const int C = (index/target.size(3)/target.size(2))%target.size(1);
        const int H = (index/target.size(3))%target.size(2);
        const int W = index%target.size(3);

        assert(flow.size(1) == 2); //flow only should have two channels

        //get target position
        scalar_t targetX = (scalar_t)W+flow[B][0][H][W];
        scalar_t targetY = (scalar_t)H+flow[B][1][H][W];

        if (isfinite(targetX) == false) { return; }
        if (isfinite(targetY) == false) { return; }

        scalar_t feature = source[B][C][H][W];

        int intNorthwestX = (int) (floor(targetX));
        int intNorthwestY = (int) (floor(targetY));
        int intNortheastX = intNorthwestX + 1;
        int intNortheastY = intNorthwestY;
        int intSouthwestX = intNorthwestX;
        int intSouthwestY = intNorthwestY + 1;
        int intSoutheastX = intNorthwestX + 1;
        int intSoutheastY = intNorthwestY + 1;

        scalar_t fltNorthwest = ((scalar_t) (intSoutheastX) - targetX) * ((scalar_t) (intSoutheastY) - targetY);
        scalar_t fltNortheast = (targetX - (scalar_t) (intSouthwestX)) * ((scalar_t) (intSouthwestY) - targetY);
        scalar_t fltSouthwest = ((scalar_t) (intNortheastX) - targetX) * (targetY - (scalar_t) (intNortheastY));
        scalar_t fltSoutheast = (targetX - (scalar_t) (intNorthwestX)) * (targetY - (scalar_t) (intNorthwestY));

        //atomic add because different locations can point to the same target
        if ((intNorthwestX >= 0) && (intNorthwestX < target.size(3)) && (intNorthwestY >= 0) && (intNorthwestY < target.size(2))) 
        {
            atomicAdd(&target[B][C][intNorthwestY][intNorthwestX], feature*fltNorthwest);
        }

        if ((intNortheastX >= 0) && (intNortheastX < target.size(3)) && (intNortheastY >= 0) && (intNortheastY < target.size(2))) 
        {
            atomicAdd(&target[B][C][intNortheastY][intNortheastX], feature * fltNortheast);
        }

        if ((intSouthwestX >= 0) && (intSouthwestX < target.size(3)) && (intSouthwestY >= 0) && (intSouthwestY < target.size(2))) 
        {
            atomicAdd(&target[B][C][intSouthwestY][intSouthwestX], feature * fltSouthwest);
        }

        if ((intSoutheastX >= 0) && (intSoutheastX < target.size(3)) && (intSoutheastY >= 0) && (intSoutheastY < target.size(2))) 
        {
            atomicAdd(&target[B][C][intSoutheastY][intSoutheastX], feature * fltSoutheast);
        }
    } 
}

torch::Tensor splat_forward_cuda(const torch::Tensor& source, const torch::Tensor& flow)
{
    //create output tensor
    auto target = torch::zeros_like(source);
    
    //define grids and blocks
    const dim3 grids((int)((source.numel()+512-1)/512),1,1);
    const dim3 blocks(512,1,1);

    //run kernel
    AT_DISPATCH_FLOATING_TYPES(source.type(), "splat_forward_cuda", ([&] 
        {
            splat_forward_kernel<scalar_t><<<grids, blocks>>>(
                source.numel(),
                source.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                flow.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                target.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>()
            );
        }));

    return target;
}


//backward function
template <typename scalar_t>
__global__ void __launch_bounds__(512) splat_backward_source_kernel(const int n,
                    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> source, 
                    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> flow,
                    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad,
                    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_source) 
{ 
    for (int index = (blockIdx.x * blockDim.x) + threadIdx.x; index < n; index += blockDim.x * gridDim.x) 
    {
        const int B = (index/grad_source.size(3)/grad_source.size(2)/grad_source.size(1))%grad_source.size(0);
        const int C = (index/grad_source.size(3)/grad_source.size(2))%grad_source.size(1);
        const int H = (index/grad_source.size(3))%grad_source.size(2);
        const int W = index%grad_source.size(3);

        assert(flow.size(1) == 2); //flow only should have two channels

        scalar_t targetX = (scalar_t)W+flow[B][0][H][W];
        scalar_t targetY = (scalar_t)H+flow[B][1][H][W];

        if (isfinite(targetX) == false) { return; }
        if (isfinite(targetY) == false) { return; }

        int intNorthwestX = (int) (floor(targetX));
        int intNorthwestY = (int) (floor(targetY));
        int intNortheastX = intNorthwestX + 1;
        int intNortheastY = intNorthwestY;
        int intSouthwestX = intNorthwestX;
        int intSouthwestY = intNorthwestY + 1;
        int intSoutheastX = intNorthwestX + 1;
        int intSoutheastY = intNorthwestY + 1;

        scalar_t fltNorthwest = ((scalar_t) (intSoutheastX) - targetX) * ((scalar_t) (intSoutheastY) - targetY);
        scalar_t fltNortheast = (targetX - (scalar_t) (intSouthwestX)) * ((scalar_t) (intSouthwestY) - targetY);
        scalar_t fltSouthwest = ((scalar_t) (intNortheastX) - targetX) * (targetY - (scalar_t) (intNortheastY));
        scalar_t fltSoutheast = (targetX - (scalar_t) (intNorthwestX)) * (targetY - (scalar_t) (intNorthwestY));

        if ((intNorthwestX >= 0) && (intNorthwestX < grad.size(3)) && (intNorthwestY >= 0) && (intNorthwestY < grad.size(2))) 
        {
            grad_source[B][C][H][W] += grad[B][C][intNorthwestY][intNorthwestX] * fltNorthwest;
        }

        if ((intNortheastX >= 0) && (intNortheastX < grad.size(3)) && (intNortheastY >= 0) && (intNortheastY < grad.size(2))) 
        {
            grad_source[B][C][H][W] += grad[B][C][intNortheastY][intNortheastX] * fltNortheast;
        }

        if ((intSouthwestX >= 0) && (intSouthwestX < grad.size(3)) && (intSouthwestY >= 0) && (intSouthwestY < grad.size(2))) 
        {
            grad_source[B][C][H][W] += grad[B][C][intSouthwestY][intSouthwestX] * fltSouthwest;
        }

        if ((intSoutheastX >= 0) && (intSoutheastX < grad.size(3)) && (intSoutheastY >= 0) && (intSoutheastY < grad.size(2))) 
        {
            grad_source[B][C][H][W] += grad[B][C][intSoutheastY][intSoutheastX] * fltSoutheast;
        }
    } 
}


torch::Tensor splat_backward_source_cuda(const torch::Tensor& source, const torch::Tensor& flow, const torch::Tensor& grad)
{
    //create output tensors
    auto grad_source = torch::zeros_like(source);

    //define grids and blocks
    const dim3 grids((int)((grad_source.numel()+512-1)/512),1,1);
    const dim3 blocks(512,1,1);

    //run kernel
    AT_DISPATCH_FLOATING_TYPES(source.type(), "splat_backward_source_cuda", ([&] 
        {
            splat_backward_source_kernel<scalar_t><<<grids, blocks>>>(
                grad_source.numel(),
                source.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                flow.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                grad_source.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>()
            );
        }));

    return grad_source;

}



template <typename scalar_t>
__global__ void __launch_bounds__(512) splat_backward_flow_kernel(const int n,
                    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> source, 
                    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> flow,
                    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad,
                    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_flow) 
{ 
    for (int index = (blockIdx.x * blockDim.x) + threadIdx.x; index < n; index += blockDim.x * gridDim.x) 
    {
        const int B = (index/grad_flow.size(3)/grad_flow.size(2)/grad_flow.size(1))%grad_flow.size(0);
        const int C = (index/grad_flow.size(3)/grad_flow.size(2))%grad_flow.size(1);
        const int H = (index/grad_flow.size(3))%grad_flow.size(2);
        const int W = index%grad_flow.size(3);

        assert(flow.size(1) == 2); //flow only should have two channels

        scalar_t fltIngrad = 0.0f;

        scalar_t targetX = (scalar_t)W+flow[B][0][H][W];
        scalar_t targetY = (scalar_t)H+flow[B][1][H][W];

        if (isfinite(targetX) == false) { return; }
        if (isfinite(targetY) == false) { return; }

        int intNorthwestX = (int) (floor(targetX));
        int intNorthwestY = (int) (floor(targetY));
        int intNortheastX = intNorthwestX + 1;
        int intNortheastY = intNorthwestY;
        int intSouthwestX = intNorthwestX;
        int intSouthwestY = intNorthwestY + 1;
        int intSoutheastX = intNorthwestX + 1;
        int intSoutheastY = intNorthwestY + 1;

        scalar_t fltNorthwest = 0.0f;
        scalar_t fltNortheast = 0.0f;
        scalar_t fltSouthwest = 0.0f;
        scalar_t fltSoutheast = 0.0f;

        if (C == 0) {
            fltNorthwest = ((scalar_t) (-1.0f)) * ((scalar_t) (intSoutheastY) - targetY);
            fltNortheast = ((scalar_t) (+1.0f)) * ((scalar_t) (intSouthwestY) - targetY);
            fltSouthwest = ((scalar_t) (-1.0f)) * (targetY - (scalar_t) (intNortheastY));
            fltSoutheast = ((scalar_t) (+1.0f)) * (targetY - (scalar_t) (intNorthwestY));

        } else if (C == 1) {
            fltNorthwest = ((scalar_t) (intSoutheastX) - targetX) * ((scalar_t) (-1.0f));
            fltNortheast = (targetX - (scalar_t) (intSouthwestX)) * ((scalar_t) (-1.0f));
            fltSouthwest = ((scalar_t) (intNortheastX) - targetX) * ((scalar_t) (+1.0f));
            fltSoutheast = (targetX - (scalar_t) (intNorthwestX)) * ((scalar_t) (+1.0f));

        }

        for (int channel = 0; channel < grad.size(1); channel += 1) 
        {
            scalar_t fltIn = source[B][channel][H][W];
            if ((intNorthwestX >= 0) && (intNorthwestX < grad.size(3)) && (intNorthwestY >= 0) && (intNorthwestY < grad.size(2))) 
            {
                grad_flow[B][C][H][W] += grad[B][channel][intNorthwestY][intNorthwestX] * fltNorthwest * fltIn;
            }

            if ((intNortheastX >= 0) && (intNortheastX < grad.size(3)) && (intNortheastY >= 0) && (intNortheastY < grad.size(2))) 
            {
                grad_flow[B][C][H][W] += grad[B][channel][intNortheastY][intNortheastX] * fltNortheast * fltIn;
            }

            if ((intSouthwestX >= 0) && (intSouthwestX < grad.size(3)) && (intSouthwestY >= 0) && (intSouthwestY < grad.size(2))) 
            {
                grad_flow[B][C][H][W] += grad[B][channel][intSouthwestY][intSouthwestX] * fltSouthwest * fltIn;
            }

            if ((intSoutheastX >= 0) && (intSoutheastX < grad.size(3)) && (intSoutheastY >= 0) && (intSoutheastY < grad.size(2))) 
            {
                grad_flow[B][C][H][W] += grad[B][channel][intSoutheastY][intSoutheastX] * fltSoutheast * fltIn;
            }
        }
    } 
}


torch::Tensor splat_backward_flow_cuda(const torch::Tensor& source, const torch::Tensor& flow, const torch::Tensor& grad)
{
    //create output tensors
    auto grad_flow = torch::zeros_like(flow);

    //define grids and blocks
    const dim3 grids((int)((grad_flow.numel()+512-1)/512),1,1);
    const dim3 blocks(512,1,1);

    //run kernel
    AT_DISPATCH_FLOATING_TYPES(source.type(), "splat_backward_flow_cuda", ([&] 
        {
            splat_backward_flow_kernel<scalar_t><<<grids, blocks>>>(
                grad_flow.numel(),
                source.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                flow.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                grad_flow.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>()
            );
        }));


    return grad_flow;
}
