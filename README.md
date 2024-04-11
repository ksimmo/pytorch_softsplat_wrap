# pytorch_softsplat_wrap
This repository provides a wrap up for the softmax splatting kernel introduced by [[1]](#1) avoiding additional libraries.
Most of the code is either completely copied or closely following the original implementation from https://github.com/sniklaus/softmax-splatting!
The main goal of this repository was to gain first experiences in how to develop and write custom cuda kernel additions for pytorch
and avoid messing around with older cupy versions as the official implementation does not run with newer cupy versions.
The current stage of this kernel is not verified/tested against the original implementation, so use at your own risk!

## Example
Simple example demonstrating a forward warp using flow extracted with RAFT. Images are obtained from UCF101 dataset [[2]](#2).
<img src=figures/warp_result.png>

## License
This provided implementation is for academic purposes only. For commercial use please get in touch with authors of official code (https://github.com/sniklaus/softmax-splatting) or [[1]](#1).

## References
<a id="1">[1]</a> Nikolaus S., Liu F.: Softmax Splatting for Video Frame Interpolation. In IEEE Conference on Computer Vision and Pattern Recognition (2020)
<a id="2">[2]</a> Soomro, K., Zamir, A., Shah, M.: UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild. CoRR abs/1212.0402 (2012)

