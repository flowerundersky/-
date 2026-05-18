# Go 公网承接层

这是一个最小系统闭环的公网承接层实现，保持原有 Python 代码不变，通过 Go 对外暴露 HTTP 接口，再调用仓库根目录下的 Python 工作流。

当前 UI 层只作为前端展示与交互使用，按钮点击后会先请求 Go 网关，再由 Go 转发到 Python agent。

## 现有闭环

- `POST /v1/preview`：解析自然语言，返回结构化字段和请求预览。
- `POST /v1/generate`：先解析，再执行图工作流，返回最终输出。
- `GET /healthz`：进程健康检查。
- `GET /readyz`：桥接器就绪检查。

如果请求体里带有 `revision_note` 和 `workflow_state`，`/v1/generate` 会进入继续修正流程，仍然不需要额外的公网路由。

公网部署拓扑见 [docs/deployment-topology.md](/home/user/Qwen_codegen_web/docs/deployment-topology.md)。

## 运行

在 `go-public-service` 目录下执行：

```bash
go run ./cmd/server
```

默认监听 `:8088`。

## 请求示例

```bash
curl -X POST http://127.0.0.1:8088/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"message":"帮我把 parser 改成支持更多字段，输出 patch", "self_check": false}'
```
