# Third-party notices

Vulkanizer is released under the MIT License (see `LICENSE`). It includes the following third-party
material, under its own license.

## Sobol direction numbers (Joe and Kuo)

`src/curand/sobol_joe_kuo.inc`, compiled into `libcurand.so`: the Sobol direction numbers of
`new-joe-kuo-6.21201`, from S. Joe and F. Y. Kuo, "Constructing Sobol sequences with better
two-dimensional projections", SIAM J. Sci. Comput. 30, 2635-2654 (2008);
https://web.maths.unsw.edu.au/~fkuo/sobol/

```
Copyright (c) 2008, Frances Y. Kuo and Stephen Joe
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice, this list of conditions
    and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice, this list of
    conditions and the following disclaimer in the documentation and/or other materials provided
    with the distribution.
  * Neither the names of the copyright holders nor the names of the University of New South Wales
    and the University of Waikato and its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Not included

Vulkanizer contains no NVIDIA source code, headers or binaries. Its CUDA, cuBLAS, cuBLASLt, cuFFT,
cuRAND, cuSOLVER, cuSPARSE and NCCL headers are clean-room declarations written from NVIDIA's public
documentation; they share with NVIDIA's headers only the names, types and values that binary
compatibility requires. `tools/setup_cuda.sh` downloads NVIDIA's CUDA toolkit from NVIDIA's servers
for building tests; that download is governed by NVIDIA's own license and is not part of Vulkanizer.
At run time Vulkanizer loads the system's libzstd and Vulkan loader, which are not distributed with
it.

## Trademarks

NVIDIA, CUDA, cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE, NCCL and PTX are trademarks or registered
trademarks of NVIDIA Corporation. Vulkan and SPIR-V are registered trademarks of the Khronos Group
Inc. All other names are the property of their respective owners and are used only to identify the
software Vulkanizer is compatible with. Vulkanizer is an independent project, not affiliated with,
endorsed or sponsored by NVIDIA Corporation or the Khronos Group.
