"""Build the Chinese report and portable source/evidence archive from passing receipts."""
import difflib
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "release-tests"
BENCH = ROOT / "release-bench"
METHODS = ("pca_int8", "mlp_int8", "grouped_vq8")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_diff(original):
    files = [p for name in ("demo", "tests") for p in (ROOT/name).rglob("*")
             if p.is_file() and p.suffix in {".py", ".json"} and p.name != "baseline_vq.py"]
    files += [ROOT / n for n in ("run_tests.ps1", "run_bench.ps1")]
    diff = []
    for p in sorted(files):
        name = p.relative_to(ROOT).as_posix()
        old = original / name
        before = old.read_text(encoding="utf-8-sig").splitlines(True) if old.exists() else []
        after = p.read_text(encoding="utf-8-sig").splitlines(True)
        diff.extend(difflib.unified_diff(before, after, fromfile="a/"+name if old.exists() else "/dev/null",
                                         tofile="b/"+name))
    (ROOT/"methods.patch").write_text("".join(diff), encoding="utf-8")


def report():
    bench = read(BENCH/"summary.json")
    training = read(TEST/"training.json")
    audit = read(TEST/"audit.json")
    precision = read(TEST/"precision.json")
    config = read(ROOT/"demo/variant_manifest.json")
    assert len(audit["methods"]) == 3 and audit["legacy_exact_tcp_trajectories"] == 9
    rows = ["# 三种八字节反馈方法的本机实测", "",
        "独立副本以 e99bc471 为基线。三种方法各完成 24 步真实微型 Qwen3 下一 token 交叉熵训练，以及本机两端 TCP 验证。正式匹配训练、生产 GPU 推理与双机时延未运行。", "",
        "## 正确性与训练", "",
        "- 31 个训练请求、7 个验证请求来自分离的随机微型模型提示组。使用相同的 Draft、Target、非零 repair、种子和预算。教师给出纠正后的下一 token，冻结接收器仍参加反向计算。每种方法的全部预期参数都有有限非零梯度，冻结权重、PCA 与尺度 buffer 保持逐位不变。",
        "- PCA 对训练输入做中心化 SVD，拒绝验证标签、少样本和秩不足。八个尺度按各分量训练绝对最大值除以 127 并设置 1e-8 下限。小 MLP 的输出经 tanh，尺度为 1/127。两法共用对称 INT8、最近偶数舍入、饱和到 ±127、二补码逐坐标线序，拒绝 0x80。全收时拒绝词为空，缺失消息解为 128 维零。分组法保留原有分类 argmax 与八组 256×16 码本。",
        "- 训练 STE 的前向严格采用硬量化，直通梯度用于舍入、VQ softmax。七个验证请求上，每法各自 FP32、BF16 的训练前向均逐位等于编码解码。跨精度结果见下表。尺度持久化为 FP32，模型参数精度随 checkpoint 绑定。",
        "- 每法 3 个请求 × 3 臂 × 2 条推理路径，共 54 次真实本机 TCP。每端独立加载参数，握手核对 contract。最终输出逐 token 等于独立自回归 Target；同法同权重的完整前缀与 KV 路径，候选、每轮接受、载荷字节和输出逐项一致；RAW 与 SHADOW 候选和接受一致。",
        "- 三法分别强制覆盖 0/1/7/8/3 个接受位置、全收 bonus、EOS、截断和跨请求重置。每法额外 9 个终止用例。原 12 项安全回归、32 组全收截断和原 FP32/BF16 对账通过。旧码本 9 个协议输入在原实现与新实现的字节及 control 相同，旧版 9 条 TCP 轨迹复现。",
        "- 编码、解码、control 构建、repair 均有方法钩子；Draft decoder 的 forward pre-hook 核实实际调用及输入 token 数。KV 下每个有反馈的非终止轮各编码、解码和构建 control 一次。终止轮与 RAW 均无多余构建。",
        "- 正常同法三臂可汇总。篡改方法、状态、64 位预算、实现、contract 或输出均已实际测试拒绝。", "",
        "| 方法 | 跨精度字节完全相同的请求 | 56 个坐标中字节相同 | 跨精度 control 最大差 | 验证集饱和超界率 |",
        "|---|---:|---:|---:|---:|"]
    for m in METHODS:
        v = precision[m]
        saturation = training[m]["heldout_clipped_fraction"]
        rows.append(f"| {m} | {sum(x['identical_coordinates']==8 for x in v)}/7 | {sum(x['identical_coordinates'] for x in v)} | {max(x['max_cross_dtype_control_difference'] for x in v):.6g} | {saturation if saturation is not None else '不适用'} |")
    rows += ["", "| 方法 | 训练步数 | 首步 / 末步 token CE | 最大参数梯度范数 | 24 步耗时 ms | 离线拟合 ms |", "|---|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        r = training[m]
        fit = f"{r['fit_ms']:.1f}" if r["fit_ms"] is not None else "不适用"
        rows.append(f"| {m} | {r['training_steps']} | {r['losses'][0]:.4f} / {r['losses'][-1]:.4f} | {max(r['max_gradient_norm'].values()):.6f} | {r['optimizer_smoke_ms']:.1f} | {fit} |")
    rows += ["", "首末步使用不同请求。这里检验梯度和权重更新，训练收益需要另做匹配试验。"]
    rows += ["", "## CPU 时延", "",
        "Intel Core Ultra 9 185H，CPU FP32，PyTorch 单线程。每法每个形状预热 50 次，随后轮换方法顺序测 3×500 次。下表为微秒，生产形状只测随机输入的 codec；tiny 的 control 已包含解码。完整 codec 从合法协议输入开始计到 128 维输出，阶段耗时存在嵌套。",
        "", "| H/D | 方法 | 编码 p50 | 量化打包 p50 | 解码 p50 | 完整 p50 / p95 / 均值 | 接收端共享字节 |", "|---|---|---:|---:|---:|---:|---:|"]
    for s in bench["shapes"].values():
        for m in METHODS:
            t = s["timings"][m]
            c = t["complete_codec"]
            rows.append(f"| {s['target_hidden']}/{s['draft_hidden']} | {m} | {t['encoder']['p50_us']:.1f} | {t['quantize_pack']['p50_us']:.1f} | {t['decoder']['p50_us']:.1f} | {c['p50_us']:.1f} / {c['p95_us']:.1f} / {c['mean_us']:.1f} | {s['codecs'][m]['receiver_bytes']:,} |")
    rows += ["", "| 方法 | tiny control p50 µs | tiny repair p50 µs | 本机 TCP RAW / SHADOW / ACTIVE 请求 p50 ms | ACTIVE 反馈 / 完整下行帧字节（九请求合计） |",
             "|---|---:|---:|---:|---:|"]
    for m in METHODS:
        t = bench["shapes"]["tiny"]["timings"][m]
        tcp = bench["localhost_tcp"][m]
        vals = " / ".join(f"{tcp[arm]['p50_us']/1e3:.1f}" for arm in ("raw","shadow","active"))
        rows.append(f"| {m} | {t['control_including_decode']['p50_us']:.1f} | {t['repair_projection']['p50_us']:.1f} | {vals} | {tcp['active']['feedback_bytes']} / {tcp['active']['framed_down_bytes']} |")
    rows += ["", "TCP 另做 3 轮交错方法顺序，各连接先预热三臂，再测三个请求；每法每臂共 9 次正式请求。帧字节为生成期间应用层帧，含协议元数据，排除 IP/TCP 包头。微基准 cold、模型配置来源和摘要、分项 p50/p95/均值、参数量与离线拟合耗时见 release-bench/summary.json。GPU 事件为 null。原始样本包含在同目录。", "",
             "## 复现及证据", "", "```powershell", ".\\run_tests.ps1 -Out .\\repro-tests",
             ".\\run_bench.ps1 -TestRoot (Resolve-Path .\\repro-tests) -Out .\\repro-bench", "```", "",
             "上述命令在解压目录运行。环境为随包脚本指定的 CPU Python；可通过 -Python 与 -Deps 替换。两份已核对 SHA256 的模型配置随包在 tests/model_configs。替换配置可通过 tests/bench_codecs.py 的 --target-config / --draft-config 指定。无需下载预训练权重。",
             "", "- demo/message_codecs.py 第 79 行拟合 PCA，第 129 行起绑定和复核完整状态。第 61 行实现 INT8 的训练直通。demo/h_vq64_head.py 接入训练；demo/latency_smoke.py 接入推理及汇总。测试用 checkpoint 保存 codec 全部参数、buffer、schema、状态哈希和共同的非零 repair U。",
             "- release-tests/training.json 逐步记录损失、最大梯度范数、训练/验证请求与拟合耗时；precision.json 记录每个验证样本的两种精度。各 TCP 子目录保留 edge/cloud 行，包括反馈十六进制、分帧字节、实际调用及 decoder 输入长度。",
             "- demo/variant_config.json 固定既有生产 VQ checkpoint 与源文件。另两法的已测合同来自各自的 tiny checkpoint；生产上需先获得匹配权重。历史包自带的 evidence/ 属于 e99bc471 基线。",
             "- 本地实现与基线的代码差异见 methods.patch。所有方法固定 8 字节，完整参数状态通过 schema 和 state 双摘要绑定。修订 contract 为 " + config["variant_contract_id"] + "。", "",
             "## 结论边界", "", "最值得质疑的假设是同一模型改变精度后仍会发送相同字节。上表给出真实反例，PCA 和小 MLP 均出现变化。在约定的同一精度和同一状态下，训练前向与实际 wire 解码逐位一致。",
             "", "本轮范围内未见阻断正确性的问题。当前证据限于小模型和一台 CPU；尚无相同生产训练预算下的接受率、完整 8B GPU 读回、端到端双机速度或真实请求负载。无法据此选择线上接受率最优的方法。"]
    (ROOT/"METHODS_REPORT.md").write_text("\n".join(rows)+"\n", encoding="utf-8")


if __name__ == "__main__":
    original_diff(Path(sys.argv[1]))
    report()
    destination = ROOT.parent / "codec-methods-local-20261120.zip"
    selected = ["demo","tests","release-tests","release-bench","evidence/tcp-kv","methods.patch",
                "METHODS_REPORT.md","program.md","run_tests.ps1","run_bench.ps1","run_cpu.ps1"]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in selected:
            path = ROOT / name
            for p in sorted(path.rglob("*")) if path.is_dir() else [path]:
                if p.is_file() and "__pycache__" not in p.parts:
                    z.write(p, p.relative_to(ROOT))
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
    print(destination, destination.stat().st_size, digest(destination))
