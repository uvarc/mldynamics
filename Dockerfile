FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ America/New_York
RUN apt-get update && apt-get -y --no-install-recommends install tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ \
        cmake libarmadillo-dev \
        wget ca-certificates unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN wget -q https://download.pytorch.org/libtorch/cu110/libtorch-cxx11-abi-shared-with-deps-1.7.1%2Bcu110.zip -O libtorch.zip && \
    unzip libtorch.zip && rm libtorch.zip

WORKDIR /opt/mldynamics
COPY CMakeLists.txt analysis.h lattice_base.h main.cpp ./

RUN mkdir build && cd build && cmake .. && make

FROM ubuntu:20.04
COPY --from=build /opt/mldynamics/build/de_c_pure /opt/mldynamics/de_c_pure
COPY --from=build /lib/libarmadillo.so.9 /lib/libarmadillo.so.9

COPY --from=build \
    /lib/x86_64-linux-gnu/libarpack.so.2 \
    /lib/x86_64-linux-gnu/libblas.so.3 \
    /lib/x86_64-linux-gnu/libc.so.6 \
    /lib/x86_64-linux-gnu/libdl.so.2 \
    /lib/x86_64-linux-gnu/libgcc_s.so.1 \
    /lib/x86_64-linux-gnu/libgfortran.so.5 \
    /lib/x86_64-linux-gnu/liblapack.so.3 \
    /lib/x86_64-linux-gnu/libm.so.6 \
    /lib/x86_64-linux-gnu/libpthread.so.0 \
    /lib/x86_64-linux-gnu/libquadmath.so.0 \
    /lib/x86_64-linux-gnu/librt.so.1 \
    /lib/x86_64-linux-gnu/libstdc++.so.6 \
    /lib/x86_64-linux-gnu/libsuperlu.so.5 \
    /lib/x86_64-linux-gnu/

COPY --from=build /lib64/ld-linux-x86-64.so.2 /lib64/ld-linux-x86-64.so.2
COPY --from=build \
    /opt/libtorch/lib/libc10.so \
    /opt/libtorch/lib/libc10_cuda.so \
    /opt/libtorch/lib/libcudart-3f3c6934.so.11.0 \
    /opt/libtorch/lib/libgomp-75eea7e8.so.1 \
    /opt/libtorch/lib/libnvToolsExt-24de1d56.so.1 \
    /opt/libtorch/lib/libtorch.so \
    /opt/libtorch/lib/libtorch_cpu.so \
    /opt/libtorch/lib/libtorch_cuda.so \
    /opt/libtorch/lib/

COPY --from=build \
    /usr/local/cuda/lib64/libcudart.so.11.0 \
    /usr/local/cuda/lib64/libnvToolsExt.so.1 \
    /usr/local/cuda/lib64/

ENV PATH /opt/mldynamics:$PATH
ENV LD_LIBRARY_PATH /opt/libtorch/lib:$LD_LIBRARY_PATH

LABEL maintainer=rs7wz@virginia.edu
ENTRYPOINT ["de_c_pure"]
