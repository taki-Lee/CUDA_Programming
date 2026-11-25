#!/usr/bin/env bash
set -euo pipefail

CONTAINER="lsl_cudadocker"
IMAGE="pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel"
WORK_DIR="$(pwd)"
GPU_SPEC="${GPU_SPEC:-all}"

# 检查 Docker 服务及访问权限
docker info >/dev/null

if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    # 已有容器：保留原来的环境和挂载配置
    if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]; then
        docker start "$CONTAINER" >/dev/null
    fi
else
    # 首次创建：使用本地镜像，挂载当前目录
    docker run -d \
        --name "$CONTAINER" \
        --pull=never \
        --gpus "$GPU_SPEC" \
        --init \
        --shm-size=8g \
        --mount "type=bind,source=${WORK_DIR},target=/workspace" \
        --workdir /workspace \
        "$IMAGE" \
        sleep infinity >/dev/null
fi

docker exec -it --workdir /workspace "$CONTAINER" bash