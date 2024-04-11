#include <torch/extension.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//forward pass, return splatted output
torch::Tensor splat_forward_cuda(const torch::Tensor& source, const torch::Tensor& flow);
//torch::Tensor splat_forward_cpu(torch::Tensor source, torch::Tensor flow);

torch::Tensor splat_forward(torch::Tensor source, torch::Tensor flow)
{
    //if (source.is_cuda() || flow.is_cuda())
    //{
    //make sure everything is on GPU!
    CHECK_INPUT(source);
    CHECK_INPUT(flow);
    return splat_forward_cuda(source, flow);
    //}

    //return splat_forward_cpu(source, flow)
}

//backward pass, return gradients
torch::Tensor splat_backward_source_cuda(const torch::Tensor& source, const torch::Tensor& flow, const torch::Tensor& grad);
//std::vector<torch::Tensor> splat_backward_source_cpu(torch::Tensor source, torch::Tensor flow);

torch::Tensor splat_backward_source(torch::Tensor source, torch::Tensor flow, torch::Tensor grad)
{
    //if (source.is_cuda() || flow.is_cuda() || grad.is_cuda())
    //{
    //make sure everything is on GPU!
    CHECK_INPUT(source);
    CHECK_INPUT(flow);
    CHECK_INPUT(grad);
    return splat_backward_source_cuda(source, flow, grad);
    //}

    //return splat_backward_source_cpu(source, flow, grad)
}

torch::Tensor splat_backward_flow_cuda(const torch::Tensor& source, const torch::Tensor& flow, const torch::Tensor& grad);
//std::vector<torch::Tensor> splat_backward_flow_cpu(torch::Tensor source, torch::Tensor flow);

torch::Tensor splat_backward_flow(torch::Tensor source, torch::Tensor flow, torch::Tensor grad)
{
    //if (source.is_cuda() || flow.is_cuda() || grad.is_cuda())
    //{
    //make sure everything is on GPU!
    CHECK_INPUT(source);
    CHECK_INPUT(flow);
    CHECK_INPUT(grad);
    return splat_backward_flow_cuda(source, flow, grad);
    //}

    //return splat_backward_flow_cpu(source, flow, grad)
}


//python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("splat_forward", &splat_forward, "splat forward");
    m.def("splat_backward_source", &splat_backward_source, "splat backward_source");
    m.def("splat_backward_flow", &splat_backward_flow, "splat backward_flow");
}