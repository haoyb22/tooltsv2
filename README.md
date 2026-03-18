# TASK

## 1.功能 添加中继; 引申code snippet

3/18 已人工添加中继功能。方案简介：添加注册表，允许部分工具通过参数source和register_as调用和注册工具结果。实现：基本ok。可用：待评估。注意：人工编写，与claude code方案和支持的工具有一定不同。

## 2.评估框架
1. 训练框架
2. 非训练框架
3. 消融实验

## 3.SOP与领域知识注入

## 4.工具箱修复与迭代

## 5.数据集角度
1. 扩展基模
2. 扩展来源

## 6.功能 完成reporter, 补充verifier

## claude code建议
```
ToolTS v2 任务报告
任务块一：评估框架构建
1.1 大纲
构建一套不依赖模型微调的、多维度综合评估体系，用于衡量 Plan-Act-Reflect 数据合成流程的质量，并与 TimeART 的 ReAct 基线进行对比。

1.2 详细任务
1.2.1 幻觉率（Hallucination Rate）
定义：LLM 在推理链中自行编造 Observation（工具输出）的比例
实现方式：检测 reasoning history 中是否存在模型自行生成的 Observation: 文本（TimeART 基线已有此检测：hallucinated = 'Observation:' in res_his）
ToolTS v2 优势：由于 Actor 严格控制每轮只输出一个 action 然后由系统执行真实工具，理论上幻觉率应为 0。此指标主要用于与 TimeART 对比
扩展：还可检测 LLM 在 Thought 中引用了不存在的工具结果、或歪曲工具实际返回值的情况（语义级幻觉）
1.2.2 工具调用有效性（Tool Call Effectiveness）
三个子指标：

子指标	定义	计算方式
成功率	工具调用执行成功的比例	成功调用数 / 总调用数
冗余率	重复或无意义调用的比例	检测相同工具+相同参数的重复调用，以及结果未被后续步骤引用的调用
利用率	工具返回结果被实际用于推理的比例	检测每个 Observation 是否在后续 Thought / Final Answer 中被引用或转化
1.2.3 结构性指标
平均推理步数：每条数据的 Plan→Act→Reflect 循环次数
工具调用多样性：每条数据使用的不同工具种类数
计划执行一致性：Planner 输出的 tool_intent 列表与 Actor 实际调用工具的匹配程度
收敛性：Reflector 判定为 continue 的轮次占比（越低说明效率越高）
1.2.4 证据链完整性（Evidence Grounding Rate, EGR）
定义：最终答案中每个事实性声明（claim）是否都有对应的工具输出证据支撑
实现：
从 Final Answer / Report 中提取所有事实性 claims
回溯推理链，检查每个 claim 是否能追溯到某个 Observation
EGR = 有证据支撑的 claims 数 / 总 claims 数
工具：可借助 LLM 辅助完成 claim 提取与证据匹配（LLM-as-Extractor）
1.2.5 Observation 一致性验证
目的：验证 TimeART 基线数据中记录的 Observation 是否与真实工具执行结果一致
实现：从 TimeART 的推理链中解析出工具名 + 参数，用同样的工具代码重新执行，比对输出
意义：量化 TimeART ReAct 方式中 Observation 的不可靠程度
注意：已有 verify.py 中 graph.verify(mode='image') 的初步实现，需扩展为系统化的一致性检查
1.2.6 LLM-as-Judge 评估
方式：使用独立的强 LLM（如 GPT-4o / Claude）对完整推理链进行质量评分
评分维度：
推理逻辑连贯性（1-5分）
工具使用合理性（1-5分）
最终答案正确性（1-5分）
证据充分性（1-5分）
实现：设计标准化的评分 prompt，对每条推理链进行打分，取平均
1.2.7 消融实验（Ablation Study）
三组对照实验：

变体	描述	对比目标
No Plan	去掉 Planner，Actor 直接根据问题行动	衡量计划步骤的价值
No Reflect	去掉 Reflector，Actor 执行固定轮次后直接 Report	衡量反思步骤的价值
No Both	退化为类似 TimeART 的单 Agent ReAct	衡量多 Agent 架构整体增益
1.2.8 实现优先级

P0（立即实现）：幻觉率、工具调用成功率、Observation 一致性验证
P1（核心指标）：EGR、LLM-as-Judge、工具冗余率/利用率
P2（深度分析）：消融实验、结构性指标
任务块二：工具机制扩展
2.1 大纲
在两个方向上扩展工具调用机制：(1) 在现有 JSON 工具调用基础上增加跨工具参数传递能力；(2) 开发代码式工具调用变体。两个方向作为同一项目的不同变体并行推进。

2.2 详细任务
2.2.1 JSON 工具调用变体 — 跨工具参数传递（derived_series 方案）
问题本质：当前所有工具都从 state['data_item']['timeseries'] 读取原始数据，无法将一个工具的输出（如移动平均序列）作为另一个工具的输入。

解决方案：在 state 中引入 derived_series 注册表


# state 结构扩展
state['data_item']['derived_series'] = {}
# 例如 rolling_stat 执行后自动注册
state['data_item']['derived_series']['rolling_mean_3'] = [4874.43, 4875.79, ...]
具体实现步骤：

扩展 state 结构：在 graph.py 的初始化中添加 derived_series 字典
修改工具函数签名：
所有工具增加可选参数 source，默认为 "original"
当 source 指定为某个 derived_series 的 key 时，从中读取数据
工具结果注册：
产生序列输出的工具（rolling_stat, volatility 等）执行后自动将结果注册到 derived_series
注册时使用语义化命名，如 "rolling_mean_w3", "volatility_w5"
Prompt 更新：
在 ACT_INSTRUCTION 中告知 Actor 可以通过 source 参数引用先前工具产生的衍生序列
提供使用示例
需要改造的工具列表：
产出序列的工具（需加注册逻辑）：rolling_stat, volatility, forecasting
消费序列的工具（需加 source 参数）：summary_stats, trend_classifier, seasonality_detector, change_point_detector, stationarity_test, spike_detector, autocorr 等
2.2.2 代码式工具调用变体（Code-based Tool Calling）
核心思想：Actor 每轮生成一段 Python 代码片段（code snippet），由系统的 Python 解释器执行。工具以函数形式暴露，代码中可以自然地用变量串联多个工具调用。

优势：

天然解决跨工具参数传递问题（通过变量赋值）
支持条件逻辑和循环
更灵活的数据处理
具体实现步骤：

工具函数接口重新封装：

将现有工具封装为纯函数形式，供代码调用
函数签名清晰化，例如：

def rolling_stat(stat: str, window: int, step: int, channel: str = None) -> dict
def trend_classifier(window: int = None, channel: str = None) -> dict
数据上下文通过闭包或全局变量传入（对代码片段透明）
安全执行环境：

使用受限的 exec() 环境，仅暴露工具函数 + 基础 Python 内置函数
禁止 import、文件 I/O、网络访问等
设置执行超时（防止死循环）
捕获异常并返回错误信息
执行结果捕获：

代码片段的最后一个表达式的值作为 Observation 返回
或约定使用 result = ... 变量作为输出
中间变量在同一轮内可复用
Prompt 改造：

重写 ACT_INSTRUCTION，指导 Actor 生成 code snippet 而非 JSON tool call
提供代码示例：

Action:
```python
ma = rolling_stat(stat="mean", window=3, step=1)
trend = trend_classifier(source=ma["rolling_results"])
result = {"moving_average": ma, "trend": trend}

明确变量命名规范和输出约定
Actor 控制逻辑修改：

将 agents.py 中 Actor 的正则匹配从 JSON tool call 格式改为代码块提取
提取 ```python ... ``` 中的代码
调用安全执行函数，获取结果
与 JSON 变体的统一：

在 graph.py 中通过配置项 config['tool_mode'] = 'json' | 'code' 切换
Planner / Reflector / Reporter / Verifier 保持不变，仅 Actor 行为差异化
2.2.3 实现优先级

P0：先完善 JSON 变体的 derived_series 方案（改动较小，风险可控）
P1：开发代码式工具调用变体（改动较大，但潜力更高）
P2：两种变体的对比实验设计
任务块三：SOP 体系构建
3.1 大纲
建立标准化分析流程（SOP），用于指导数据合成过程中 Planner 的规划和 Verifier 的评判，同时作为高质量推理链的参考模板。

3.2 详细任务
3.2.1 SOP 的定位与作用
对 Planner：作为 few-shot 示例或 instruction，引导其为特定任务类型生成合理的工具调用计划
对 Verifier：作为评判标准，检查推理链是否覆盖了该任务类型应有的分析步骤
对数据集：作为质量标注维度，标记每条数据对应的 SOP 类型
3.2.2 SOP 获取方式（四条路径并行）
路径 A：半自动任务类型归纳

对现有数据集（TimeSeriesExam, Time-MQA）中的问题进行分类
归纳出典型任务类型，如：
趋势分析类
异常检测类
周期性分析类
多通道相关性分析类
统计量计算类
预测推理类
为每类任务手动编写标准分析流程
路径 B：LLM 生成 + 人工审核

给 LLM 提供工具列表 + 任务类型描述
让 LLM 生成候选 SOP
人工审核、修正、合并
迭代优化
路径 C：从优质合成结果中反向提取

运行当前 Plan-Act-Reflect 流程合成一批数据
筛选高质量结果（工具调用成功、答案正确、推理连贯）
从中抽象出通用的分析步骤模式
形式化为 SOP
路径 D：领域文献提取

参考时序分析教材、实践指南中的标准分析流程
例如：Box-Jenkins 方法论（识别→估计→诊断→预测）
转化为工具调用序列
3.2.3 SOP 格式规范
建议采用结构化格式：


{
  "sop_id": "SOP-TREND-001",
  "task_type": "趋势分析",
  "description": "分析时间序列的整体趋势方向与强度",
  "steps": [
    {
      "step": 1,
      "action": "获取序列基本信息",
      "tool": "series_info",
      "purpose": "了解序列长度、通道数、缺失值情况"
    },
    {
      "step": 2,
      "action": "全局趋势分类",
      "tool": "trend_classifier",
      "args": {"window": null},
      "purpose": "判断整体趋势方向"
    },
    {
      "step": 3,
      "action": "分段趋势分析",
      "tool": "trend_classifier",
      "args": {"window": "T//4"},
      "purpose": "捕捉局部趋势变化"
    },
    {
      "step": 4,
      "action": "变点检测",
      "tool": "change_point_detector",
      "purpose": "定位趋势转折点"
    },
    {
      "step": 5,
      "action": "平稳性检验",
      "tool": "stationarity_test",
      "purpose": "确认序列是否具有单位根（趋势的统计证据）"
    }
  ],
  "expected_conclusion_template": "序列呈{direction}趋势，在索引{cp}处存在转折，ADF检验p值为{p}，{stationarity_conclusion}"
}
3.2.4 首批 SOP 清单（建议优先开发）
SOP ID	任务类型	核心工具链	优先级
SOP-TREND	趋势分析	series_info → trend_classifier → change_point_detector → stationarity_test	P0
SOP-SEASON	周期性分析	series_info → seasonality_detector → autocorr → rolling_stat	P0
SOP-ANOMALY	异常检测	series_info → spike_detector → summary_stats → noise_profile	P0
SOP-CORR	多通道相关性	series_info → channel_correlation → cross_correlation → granger_causality	P1
SOP-STAT	统计量计算	series_info → summary_stats → quantile_value → rolling_stat	P1
SOP-COMPARE	序列比较	series_info → dtw_distance → shape_similarity → channel_correlation	P2
3.2.5 SOP 集成到流程
Planner prompt 增强：根据问题类型自动匹配相关 SOP，注入到 Planner 的 prompt 中作为参考
Verifier 检查项：Verifier 可参照 SOP 检查推理链是否遗漏关键分析步骤
数据标注：每条合成数据标注其对应的 SOP 类型，便于后续按类分析
任务块四：工具集构建与迭代
4.1 大纲
修复现有工具的已知问题，补充缺失工具，并建立实证驱动的工具箱迭代机制。

4.2 详细任务
4.2.1 已知问题修复
问题	文件/位置	修复方案	状态
autocorr lag 参数硬编码	tools.py ~L182	col.autocorr(lag=5) → col.autocorr(lag=lag)	✅ 已修复
return_calc 接受标量而非时间索引	tools.py	重新设计：接受 (t1, t2) 时间索引，内部自动取值并计算	待实施
shape_similarity 与 channel_correlation 功能重叠	tools.py	方案一：删除 shape_similarity；方案二：让两者差异化（shape_similarity 做 z-score 归一化后计算）	待决策
graph.py L91-92 硬编码 hack	graph.py	清理 i==7 的特殊处理	待实施
4.2.2 工具补充（候选列表）
基于前期讨论中识别的工具缺口：

工具名	类别	功能	优先级
interpolate	Numerical	对缺失值进行插值（线性、样条等）	P0
differencing	Numerical	一阶/多阶差分，用于去趋势	P0
segment_stats	Numerical	按变点分段后分别计算统计量	P1
frequency_analysis	Pattern	FFT 频谱分析	P1
outlier_filter	Numerical	基于 IQR/Z-score 的异常值过滤	P2
resample	Numerical	时间序列重采样（上采样/下采样）	P2
4.2.3 工具元信息（TOOLCARD）完善
确保每个工具的 TOOLCARD 包含完整信息：
name：工具名
description：功能描述
parameters：参数列表（名称、类型、是否必需、默认值、取值范围）
returns：返回值结构描述
usage_example：调用示例
constraints：使用限制（如 noise_profile 的 window<10 限制）
TOOLCARD 作为 Prompt 生成的数据源，确保 Prompt 中的工具描述与实际实现一致
4.2.4 实证驱动的工具箱迭代机制
这是一个可作为贡献点讲述的 story：不同于以往论文直接选定工具集，本项目采用数据驱动的方式迭代优化工具箱。

迭代流程：


1. 初始工具箱 V0
       ↓
2. 运行合成流程，产出一批数据
       ↓
3. 分析工具调用日志
   - 哪些工具从未被调用？（可能不需要 / 描述不清晰）
   - 哪些工具调用失败率高？（可能有 bug / 参数设计不合理）
   - 哪些场景下 LLM 尝试调用不存在的工具？（说明缺少该工具）
   - 哪些工具结果从未被引用？（可能无用 / 输出格式不友好）
       ↓
4. 根据分析结果进行调整
   - 修复 bug
   - 优化参数设计
   - 补充缺失工具
   - 改善工具描述 / TOOLCARD
   - 删除冗余工具
       ↓
5. 更新工具箱 → V1，回到步骤 2
日志分析的具体维度：

维度	分析方法	产出
工具使用频率分布	统计每个工具被调用的次数	识别冷门工具
工具调用失败原因分类	对 error 信息聚类	定位系统性 bug
幻想工具名统计	提取 LLM 尝试调用但不存在的工具名	候选新增工具
参数错误模式	分析参数类型/范围不匹配的情况	参数设计改进
工具组合模式	挖掘高频的工具调用组合	SOP 发现 + 工具集协同性评估
4.2.5 实现优先级

P0：修复已知 bug（autocorr 已完成，return_calc, graph.py hack）
P0：补充 interpolate、differencing 工具
P1：完善 TOOLCARD，确保与 prompt 一致
P1：建立日志收集与分析脚本
P2：完整迭代一轮（V0→V1），形成可写入论文的实证数据
总体时间线建议

第一阶段（1-2周）：基础建设
  ├─ 修复已知工具 bug（P0）
  ├─ 实现评估框架核心指标（幻觉率、成功率、Observation一致性）
  └─ 编写首批 3 个 P0 级 SOP

第二阶段（2-3周）：机制扩展
  ├─ 实现 derived_series 跨工具传参（JSON变体）
  ├─ 开发代码式工具调用变体原型
  └─ 补充 interpolate / differencing 工具

第三阶段（2-3周）：迭代验证
  ├─ 运行合成流程，收集工具调用日志
  ├─ 完成首轮实证驱动的工具箱迭代
  ├─ 实现 EGR、LLM-as-Judge 评估
  └─ JSON vs Code 变体对比实验

第四阶段（1-2周）：消融与总结
  ├─ 消融实验（No Plan / No Reflect / No Both）
  ├─ 全部评估指标汇总
  └─ 数据与结论整理
以上就是基于我们前面讨论内容的完整任务报告。四个任务块之间存在依赖关系：工具集构建（块4） 是基础设施，直接影响合成质量；工具机制（块2） 和 SOP（块3） 是合成流程的核心增强；评估框架（块1） 贯穿始终，用于量化每次迭代的效果。建议按照时间线中的阶段顺序推进，但各块内部的 P0 任务可以并行启动。
```