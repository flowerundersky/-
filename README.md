# Qwen2.5 Code Generation Web

这是一个基于 Qwen2.5 的本地代码生成 Web 应用。项目默认通过 Python + Gradio 启动，也提供了一个可选的 Go 网关，用于把 Python 工作流封装成 HTTP 服务。

## 功能

- 本地代码生成与补全
- 支持基于 Hugging Face / 本地模型目录加载模型
- 支持自定义生成参数
- 可选 Go 服务接入，便于作为统一 API 对外提供能力

## 本地部署

### 1. 准备环境

建议使用 Python 3.10+。先进入项目根目录：

```bash
cd /home/user/Qwen_codegen_web
```

创建并激活虚拟环境后安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果你希望使用 GPU 版本的 `torch`，请按照你的 CUDA / 显卡环境单独安装对应版本。

### 2. 准备模型

项目默认会从本地模型目录加载模型，而不是在启动时在线下载。默认路径来自代码配置：

`models/Qwen2.5-3B/models--Qwen--Qwen2.5-3B-Instruct-AWQ/snapshots/3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd`

如果你的模型实际放在别的位置，可以通过环境变量覆盖：

- `MODEL_PATH`：本地模型路径
- `MODEL_NAME`：模型名称，默认 `Qwen/Qwen2.5-3B-Instruct`
- `BACKEND`：后端类型，默认 `qwen_hf`
- `DEVICE`：运行设备，默认 `auto`

### 3. 启动 Web 应用

最简单的方式是直接启动 Python 入口：

```bash
python app.py
```

或者使用仓库里的开发脚本：

```bash
bash scripts/run_dev.sh
```

启动后默认访问地址为：

```bash
http://127.0.0.1:7860
```

### 4. 调整生成参数

如需修改默认生成行为，可以通过环境变量配置：

- `MAX_NEW_TOKENS`
- `MIN_NEW_TOKENS`
- `TEMPERATURE`
- `TOP_P`
- `REPETITION_PENALTY`

例如：

```bash
export MODEL_PATH=/data/models/qwen
export DEVICE=cuda
export TEMPERATURE=0.2
python app.py
```

## 可选：Go 网关部署

如果你想把 Python 工作流包装成 HTTP 接口，可以使用 `go-public-service` 目录下的 Go 网关。

### 启动方式

```bash
cd go-public-service
go run ./cmd/server
```

默认监听 `:8088`，并通过仓库根目录下的 `python_bridge.py` 调用 Python 工作流。

### 常用环境变量

- `ADDR`：HTTP 监听地址，默认 `:8088`
- `PYTHON_BIN`：Python 可执行文件，默认 `python3`
- `BRIDGE_SCRIPT`：桥接脚本路径，默认仓库根目录下的 `python_bridge.py`
- `REQUEST_TIMEOUT_SECONDS`：单次请求超时，默认 300 秒

## 目录说明

- `app.py`：Gradio Web 应用入口
- `src/`：核心逻辑，包括模型、提示词、解析、图工作流和 UI
- `scripts/`：启动与环境检查脚本
- `go-public-service/`：可选的 Go HTTP 网关
- `docs/`：架构和提示词设计文档

## 说明

项目当前以本地部署为主，启动前最重要的是确认模型目录存在且与 `MODEL_PATH` 对应。
