FROM intelanalytics/ipex-llm-inference-cpp-xpu:2.3.0-SNAPSHOT
# Device to use, like Max, Flex, Arc, iGPU
ENV DEVICE=iGPU
ENV OLLAMA_NUM_GPU=999
ENV no_proxy=localhost,127.0.0.1
ENV ZES_ENABLE_SYSMAN=1
ENV SYCL_CACHE_PERSISTENT=1
ENV ZES_ENABLE_SYSMAN=1
# [optional] under most circumstances, the following environment variable may improve performance, but sometimes this may also cause performance degradation
ENV SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
# [optional] if you want to run on single GPU, use below command to limit GPU may improve performance
ENV ONEAPI_DEVICE_SELECTOR=level_zero:0
RUN mkdir -p /llm/ollama
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/llm/ollama"
RUN cd /llm/ollama && init-ollama

WORKDIR /llm/ollama
