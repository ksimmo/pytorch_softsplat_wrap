import torch

from motion_diffusion.modules.extensions.splatting.softsplat import _softsplat

def splat(source ,flow, weight=None, mode="soft", epsmode="add"):
    assert flow.size(1)==2
    if weight is None:
        weight = torch.sqrt(torch.sum(flow.pow(2), dim=1)).unsqueeze(1) #use flow magnitude if weight is not specified
        #normalize?
        weight = weight.clamp(0.0, 20.0) #so it does not explode
    
    H,W = source.size(-2),source.size(-1)

    #rescale flow and weights to match feature dimensions
    scaled_flow = torch.nn.functional.interpolate(flow, size=(H,W), mode="bilinear", align_corners=False)
    scaled_flow[:,0] = scaled_flow[:,0]/flow.size(-1)*W
    scaled_flow[:,1] = scaled_flow[:,1]/flow.size(-2)*H
    scaled_weight = torch.nn.functional.interpolate(weight, size=(H,W), mode="bilinear", align_corners=False)

    return _softsplat(source, scaled_flow, scaled_weight, mode, epsmode)