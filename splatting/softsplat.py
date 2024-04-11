import os
import torch
from torch.utils.cpp_extension import load

cur_dir = os.path.dirname(__file__)
softsplat_ext = load(name='softsplat_ext', sources=[os.path.join(cur_dir,"softsplat.cpp"), os.path.join(cur_dir, "softsplat.cu")], verbose=True)

class softsplat_func(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, source, flow):
        ctx.save_for_backward(source, flow)

        #do calculations
        return softsplat_ext.splat_forward(source, flow)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        source, flow = ctx.saved_tensors
        #calculate gradients
        if ctx.needs_input_grad[0]:
            grad_source = softsplat_ext.splat_backward_source(source, flow, grad)
        else:
            grad_source = None
        if ctx.needs_input_grad[1]:
            grad_flow = softsplat_ext.splat_backward_flow(source, flow, grad)
        else:
            grad_flow = None
        return grad_source, grad_flow
    

def _softsplat(source, flow, weight, mode="soft", epsmode="add"):
    #do stuff
    if mode=="avg":
        source = torch.cat([source, torch.ones_like(source[:,:1])], dim=1)
    elif mode=="linear":
        source = torch.cat([source*weight, weight], dim=1)
    elif mode=="soft":
        source = torch.cat([source*weight.exp(), weight.exp()], dim=1)

    warped = softsplat_func.apply(source, flow)

    #do stuff
    if mode in ["avg", "linear", "soft"]: #only normalize for the modes we use weights
        norm = warped[:,-1:]
        if epsmode=="add":
            norm = norm +1e-7
        elif epsmode=="replace":
            norm[norm==0.0] = 1.0
        elif epsmode=="clip":
            norm = norm.clip(1e-7, None)

        warped = warped[:,:-1]/norm
    return warped

