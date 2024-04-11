from PIL import Image

import numpy as np

import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image

from einops import rearrange

from splatting import splat

if __name__ == "__main__":
    #load images
    img1 = np.array(Image.open("start_frame.jpeg"))
    img2 = np.array(Image.open("target_frame.jpeg"))

    img1 = (img1.astype(float)/255.0-0.5)/0.5
    img2 = (img2.astype(float)/255.0-0.5)/0.5

    device = torch.device("cuda:5")

    img1 = torch.FloatTensor(img1).to(device)
    img1 = rearrange(img1.unsqueeze(0), "b h w c -> b c h w")

    img2 = torch.FloatTensor(img2).to(device)
    img2 = rearrange(img2.unsqueeze(0), "b h w c -> b c h w")

    raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    raft.eval()
    for param in raft.parameters():
        param.requires_grad = False

    with torch.no_grad():
        flow = raft(img1, img2)[-1]
    
        warped = splat(img1, flow)

    flow = (flow_to_image(flow).float()/255.0-0.5)/0.5

    img = torch.cat([img1, flow, warped, img2],dim=-1).cpu().numpy()
    img = (img[0]+1.0)*0.5*255.0
    img = rearrange(img, "c h w -> h w c")

    Image.fromarray(img.astype(np.uint8)).save("figures/warp_result.png")

