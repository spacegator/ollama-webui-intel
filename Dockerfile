FROM intelanalytics/ipex-llm-inference-cpp-xpu:latest

RUN mkdir -p /llm/ollama \
    && cd /llm/ollama \
    && init-ollama

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/llm/ollama"

WORKDIR /llm/ollama
