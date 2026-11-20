# 三种 8 字节反馈编码器

用于投机解码的三种反馈方法共用 8 字节消息和 128 维控制向量。实现包括训练集拟合的 PCA＋INT8、小 MLP＋INT8，以及八组离散码本。

54 次微型 Qwen3 本机 TCP 正确性测试中，三种方法的输出均逐 token 匹配独立教师。生产输入维度的时延数据来自随机输入和单线程 CPU。正式模型接受率及双机 GPU 时延尚未测量。

| 内容 | 位置 |
| --- | --- |
| 实验结果、方法与限制 | [完整报告](METHODS_REPORT.md) |
| 编码器、训练头与推理 | [demo](demo/) |
| 测试源码与原始结果 | [tests](tests/) · [release-tests](release-tests/) |
| 时延原始样本与汇总 | [release-bench](release-bench/) |
| 实现规格与基线差异 | [program.md](program.md) · [methods.patch](methods.patch) |
| 已复核的完整交付包 | [codec-methods-local-20261120.zip](artifacts/codec-methods-local-20261120.zip) |

## 本机复现

在仓库根目录用 PowerShell 运行。原测试环境为 CPU PyTorch 2.9.1、Transformers 4.57.3。可通过脚本参数 -Python 和 -Deps 指定本机 Python 和依赖目录。

```powershell
.\run_tests.ps1 -Out .\repro-tests
.\run_bench.ps1 -TestRoot (Resolve-Path .\repro-tests) -Out .\repro-bench
```

完整交付包 SHA-256 为 `7b1a9870d413621b067e1a69f476f6788caf6dc9b769ed4b75a7d22dd09815ba`。

正式 A100 权重复用、完整 3986 题补训和原 503 题验证的命令见 [formal/README.md](formal/README.md)。
