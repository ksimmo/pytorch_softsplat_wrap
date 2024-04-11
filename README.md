# pytorch_softsplat_wrap
This repository provides a wrap up for the softmax splatting kernel introduced by [1] avoiding additional libraries.
Most of the code is either completely copied or closely following the original implementation from https://github.com/sniklaus/softmax-splatting!
The main goal of this repository was to gain first experiences in how to develop and write custom cuda kernel additions for pytorch
and avoid messing around with older cupy versions as the official implementation does not run with newer cupy versions.
The current stage of this kernel is not verified/tested against the original implementation, so use at your own risk!

## Example
<img src=figures/warp_result.png>

## References
```
[1]  @inproceedings{Niklaus_CVPR_2020,
         author = {Simon Niklaus and Feng Liu},
         title = {Softmax Splatting for Video Frame Interpolation},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2020}
     }
```

