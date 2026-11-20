# 三种8字节反馈 codec：本机实现与测试计划

## 用户要求与交付目标

来源：@ohhandme，2026-11-20，dm:@ohhandme 消息 f03451ef-af50-4460-8cce-60acae58b6a2。
用户明确要求将下述三种方法写成 program.md，交 CodeReviewer 在本机写代码并实际测试：
1. PCA/固定线性投影＋INT8。
2. 学习型小 MLP＋INT8。
3. 当前分组离散码本。

本轮目标是交付真实可运行的统一实现、正确性/训练链路测试及本机 CPU 时延证据。不能把微型模型、合成输入或少量优化步骤当作正式模型接受收益。无需启动/停止远端训练、下载模型、部署或进行双机测速。
执行负责人 CodeReviewer；PaperReader 提供接口及对照口径。结果向用户自己的 DM 报告并回传 PaperReader，由后者按需交原执行席。

## 基于哪个版本

复制以下完整包到你自己的新工作目录，再开发，保留原件和前后差异：
C:/Users/asus/.raft/agents/b6f4ae66-6dae-4817-8014-779ad45d8b68/draft-kv-reviewed-20261120
其中 demo/ 为源码，tests/ 为已有测试，run_cpu.ps1 为现成本机环境入口。
已审版本 contract：e99bc4710666ea02b4400bdcb3434d931085418c15b850fd71e6c8593664409b。
既有安全回归：逐层KV完整性、拒绝/全收/终止边界、control构建失败不留半成品、汇总拒绝跨实现/跨contract/不等输出。修改后必须保留。

关键入口：
- demo/h_vq64_codec.py：现 ConditionedVQ64。
- demo/h_vq64_head.py：HTaskHead 使用 message 模块构建128维反馈control。
- demo/cached_draft.py：prepare/build_control 中每轮解码一次，候选复用。
- demo/latency_smoke.py：实际推理、计时、分组汇总。
- demo/cpu_test.py：真实微型Qwen3与本机TCP，不是GPU速度证据。

不要把旧 draft-kv-fix-20261120 当最新源。更换文件后刷新 source_sha256、codec配置/状态绑定和 contract，不沿用旧哈希运行改动文件。

## 统一数据与接口

第一版固定8字节额外反馈，不扩展4/16字节，不做RL或补传。只比较 codec 方法。
沿用当前合法输入：当前已确认边界 Target hidden H、已知接受数、拒绝词embedding（全收为零）和纠正/bonus词embedding。拼接/归一化/缺失表示与原 _inputs 一致，三方法获得相同信息。不得读取未来label、下一轮接受决定或测试集统计。
对合法 batch=1 输入，统一接口至少具备：
- encode_payload(H, accepted_count, rejected_embedding, correction_embedding) -> bytes，恰好8字节。
- decode_payload(payload: bytes | None) -> Tensor[1,128]。
- train_control(..., temperature等可选) -> Tensor[1,128]，量化前向遵循实际wire规则，反向为明确记录的直通近似。
- 配置/完整参数和buffer状态可保存加载、可校验版本与哈希。
None 表示无额外H消息，返回零message control；既有本地control和必要协议照旧。
保持 public token 预测接口一致，训练/验证代码可按 codec_type 工厂选择实现。不能伪造 .codebook 属性适配耦合，应明确清理原 VQ 专属断言，仅在 VQ 分支检查码本形状。

## 三种实现的明确规格

### A. pca_int8

使用仅训练集拟合的中心化 PCA，默认8个分量；均值/投影均冻结，PCA不能访问验证/测试数据。固定线性投影是可选消融，不必再添加第四个主方法。
先得到8个实数，再逐坐标对称量化到 [-127,127]，按二补码序列化为8个字节。解码时恢复符号和尺度，再用可训练的 Linear(8,128) 生成统一control。这是“PCA瓶颈＋任务解码器”，不是声称8个PCA值能无损恢复H。
量化尺度是训练集拟合、预先共享的8个buffer，不逐轮发送；可固定为各训练分量绝对值最大值/127，零分量用有界epsilon，必须记录规则、饱和率和异常输入处理。若选其他规则，写入配置后固定，不能在测试集选百分位。
CPU夹具至少保证训练样本数和有效秩足够；不足8维时明确拒绝或显式有记录地补零，不静默改变载荷语义。

### B. mlp_int8

同一输入 x -> Linear(input_dim,128) -> SiLU -> Linear(128,8) -> tanh。
8个坐标取固定共享尺度1/127，对称量化并按与A相同的8字节格式打包。边端反量化后 Linear(8,128)。
训练使用round的直通近似，前向实际量化，不能训练时绕开量化而只在推理时量化。scale/边界/舍入规则必须显式且可复现。

### C. grouped_vq8（现有基线）

保留现编码器 input_dim ->128->8*256 logits，每组argmax选0..255；8编号为8rawbytes。
共享codebook形状8*256*16，查表得到128维control；保留原hard/soft直通训练方式。
这是当前学习型类别索引/分组码本实现，不要改写为最近邻VQ后仍冒称旧方法等价。单独验证选此分支时，与原 reviewed 代码同参数/输入/payload 的输出和轨迹不变。

## 接收端与训练公平性

三者统一输出128维message control，接入相同本地control和位置相关repair。反馈每个验证轮只解码、构建一次，逐候选只算位置相关部分。编码也每个反馈轮一次。不可为选择方法额外跑三次Draft；本轮method在请求前配置固定。
首轮本机训练冒烟可冻结共同Draft/repair，只更新各codec允许的参数（A投影冻结，A解码器可训练，B/C编码和解码/码本可训练）；确保梯度经过冻结的接收计算，不用no_grad切断新模块的训练梯度。
当前旧VQ checkpoint可能与repair共同适配，不能拿它已训练的表现和随机初始化A/B宣布方法优劣。正式效果对照应从匹配的共同初始化/相同任务数据和损失/同更新及token预算出发，报告参数量与训练计算差别。如果本机没有正式数据，只交工程和冒烟证据，正式匹配训练标未运行。
沿用共同实际上下文的Target token CE（或另行明确统一的蒸馏设置），不能只为某一法加H重建损失。用训练split拟合PCA及scale；按请求/题族隔离验证，禁止窗口跨split泄漏。
不要自行延伸为一轮完整LoRA或大模型训练。本机优化器冒烟可设16～32步的小预算，训练/验证夹具分离，目的仅验证wire训练链路及冻结规则。

## 必须实际执行的测试

1. Wire：每法恰好8rawbytes；8个坐标/编号覆盖边界；INT8负数/零/正数、饱和、符号还原、端序与固定舍入；None为零；长度错误/类型错误/NaN/Inf/协议输入不合法明确拒绝。wire长度不含已有协议及包头，整包另报。
2. 编解码与checkpoint：相同输入确定性、保存加载后payload/control相同；PCA均值/投影/scale完整持久化；codebook/scale版本不匹配拒绝混用。train_control的硬前向与encode再decode对应，检查FP32和CPU支持的BF16，逐项报告精度误差与字节一致性。
3. 训练链路：实际token CE经冻结接收侧传播到A decoder、B encoder/decoder、C encoder/codebook的预期参数；冻结模块无梯度更新；随机adapter不能无意冻死。至少做有非零repair的优化器冒烟，报告loss/梯度/有限值，不把必然单调降loss作脆弱门槛。
4. 集成：在实际微型Qwen3、本机TCP上，三种codec各跑RAW/SHADOW/ACTIVE；验证最终输出等于独立原生Target，RAW与SHADOW候选/接受相同。不同codec之间的候选和接受允许不同，绝不要求它们相等。
5. 同一codec、同checkpoint的新缓存入口与完整前缀参考路径对账候选、反馈bytes、接受轨迹/最终输出；覆盖首拒/中拒/全收bonus/EOS/截断/跨请求reset和已有12项安全回归。
6. 汇总按codec_type、bit预算、codec参数/状态哈希、draft_implementation及contract绑定。正常同法三臂可汇总；混codec、混checkpoint、混实现或三臂正确输出不一致拒绝。不要把所有method名合成同一个kv桶。
7. 调用钩子验证每轮反馈decode/control一次，终止无多余prepare；独立计encoder、payload打包、decoder、control、repair/decoder调用和实际输入token，不只测封装函数次数。

## 时延测量与结果解释

提供一个可复现CPU benchmark脚本：三法使用相同输入样本、设备、dtype和线程数，分开预热与正式计时，交错/轮换方法顺序，重复多轮。默认可用50次预热、500次计时、3轮；若太慢可调整并记录，不能只跑一次计时。
分列encoder数值计算、量化/argmax与打包、decode、完整codec、control/repair、TCP请求总墙钟。注明嵌套区间，不相加重复成本。真实encode包含GPU->CPU同步，但本机CPU测量不能推算它；GPU事件无观测写null。
同时测tiny输入及从实际模型配置核得的生产形状随机输入；后者只表示形状下CPU codec成本，不是生产模型质量。报告payload=8、共享参数/表/scale字节、host p50/p95/均值、样本数、线程数、库版本与冷启动另项。PCA拟合/训练时间另报，不计入在线每轮codec。
实际TCP是本机socket，不是两机链路、4090/A100速度或8B通信收益；请在结果JSON中明确scope。沿用原Target验证的正确输出要求。未测GPU部分不得填0冒充免费。
无需另写一个真实网络模拟器；若附载荷估算，使用8*实际字节/R，并明确未含排队/包头/RTT和重传，不能把估算称实测。

## 已知本机环境及限制

原 run_cpu.ps1 已绑定可用CPU PyTorch2.9.1、Transformers4.57.3，路径：
Python: C:/Users/asus/.raft/agents/6946fc4f-34e5-42f8-abf5-cd3fd9376a5e/review-v71-next-heads-20261108/cpu-env/Scripts/python.exe
PYTHONPATH: C:/Users/asus/.raft/agents/6946fc4f-34e5-42f8-abf5-cd3fd9376a5e/task58-draft-opd/cpu-deps
设置HF_HUB_OFFLINE=1；优先复用此环境，无需重装Torch。
之前资源核查：Draft文件7/8、缺configuration.json，反馈头匹配；8B Target分片未找到，CPU Torch/Intel Arc，不应把源码Linux设备入口当成本机GPU可直接执行。若资源现状改变，重新核实并记录，不复述旧结论作为新测量。

## 交付与完成条件

交独立工作目录、源码diff、三法配置/状态schema及哈希、新contract、运行入口(run_tests.ps1/run_bench.ps1或等价)、测试原件和benchmark逐样本JSON/汇总CSV或JSON。
报告必须明确：已实现哪些方法，已跑哪些正确性/训练/TCP测试，哪个shape和CPU条件下测了多少，失败/未运行/资源阻塞有哪些；CPU最快不能直接宣布最终方法最优。
按严重程度列剩余缺陷，提供可复现命令。请直接实施、测试、修复直到本轮本地目标完成；若正式数据缺失，不阻塞可完成的工程测试，也不编造正式接受结果。
用户已授权本地实现测试，本计划无需再向用户确认一般实现细节。远端生产实验和排程继续由原执行席负责。
