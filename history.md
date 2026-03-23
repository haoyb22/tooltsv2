# ToolTS v2 修改历史

## 2026-03-23 P0 阶段修改

### 1. 删除 `graph.py` 硬编码 hack

**文件**: `graph.py`
**位置**: 原 L108-109（现已删除）

**问题**:
`run()` 方法中存在一段针对 Time-MQA 数据集第7条数据的临时补丁：
```python
if i==7 and self.config.get('data_path', ...).endswith('.csv'):
    state['data_item']['answer'] = "The mean of the data points is 30.67. ..."
```
这段代码会在处理 Time-MQA 数据集时，强制将第7条数据的 answer 替换为写死的字符串，导致该条数据的 ground truth 被污染，影响批量运行结果的可信度。

**修复**:
直接删除这两行。数据集原始的 answer 字段应当作为可信来源。

### 2. 修复 `return_calc` 工具 bug

**文件**: `tools.py`
**位置**: `return_calc()` 函数 + `return_calc_tool_card`

**问题 1（致命）——返回类型错误**:
原实现直接返回裸 float：
```python
def return_calc(state, arguments):
    t1, t2, kind = arguments['t1'], arguments['t2'], arguments['kind']
    if kind == "diff":
        return t2 - t1  # 返回 float
    ...
```
但 `execute_tool()` 的处理逻辑要求返回值必须是：
- `str` → 包装为 `{"result": str}`
- `tuple (data, str)` → 拆包处理

返回 float 时，`execute_tool` 执行 `return_value[1]`（对 float 取下标）会抛出 `TypeError`，被 `except TypeError` 捕获后输出误导性的 `"Argument mismatch"` 错误。因此**每次调用 `return_calc` 都静默失败**，且错误信息指向了错误的原因。

**问题 2（设计）——不使用 state，无法处理多通道**:
原函数完全忽略 `state` 参数，将 `t1`/`t2` 当作标量值直接计算，与其他所有工具（从 `state` 读取时序数据、对每个通道独立计算）的设计模式不一致。

**修复**:
- `t1`/`t2` 改为时间索引（integer），函数自己从 `state['data_item']['timeseries']` 中取对应位置的值
- 对每个通道独立计算，返回 `{channel: result}` 格式
- 新增可选参数 `channel` 用于指定单通道计算
- 返回值改为格式化字符串，与其他工具保持一致
- `v1 == 0` 时 pct 模式返回 `None`（原来是 raise，会导致工具完全失败而非返回可用结果）

**同步更新 `return_calc_tool_card`**:
- `t1`/`t2` 的 `type` 从 `"number"` 改为 `"integer"`
- description 从"The initial value"改为"The starting time index"
- 新增 `channel` 参数的说明

**修改后的函数签名行为**:
```
输入: t1=0, t2=5, kind="pct", channel="ts"（可选）
输出: "The pct return between index 0 and 5 are: {'ts': 0.032}"
```

### 3. 完成 `Reporter` 实现

**文件**: `prompts.py`、`agents.py`

**问题**:
`Reporter.run()` 是整条 Plan-Act-Reflect 流水线的终点节点，负责综合所有工具调用结果生成最终答案。原实现是占位 stub：
```python
def run(self, state):
    return 'debug'
```
这意味着整条链跑完没有任何有意义的产出，生成的合成数据缺少最关键的"最终答案"字段。

**修复——`prompts.py` 新增**:

新增 `REPORT_INSTRUCTION` 和 `create_report_prompt(state)` 函数。

设计要点：
- Reporter 被明确要求引用具体 observation 作为证据（对应 EGR 目标）
- 不能引入 history 之外的信息（防止幻觉）
- 选择题必须输出精确匹配选项的文字
- prompt 中包含可视化图（`<visualization>` 占位符），Reporter 也能看到时序图

同时将 `create_report_prompt` 添加到 `agents.py` 的 import 列表。

**修复——`agents.py` 更新**:
```python
def run(self, state):
    print("reporter run")
    prompt = create_report_prompt(state)
    response = self.llm.invoke([...])
    report = response.content
    print(report)
    return report
```
实现结构与 Planner、Reflector 完全对称。

### 4. 工具调用成功率统计

**文件**: `graph.py`

**需求**:
P0 评估指标之一。需要记录每次工具调用的成功/失败，并在批量运行结束后汇总统计，用于工具箱迭代分析。

**改动 1——state 初始化（`preprocess`）**:
在 state 字典中新增字段：
```python
'tool_call_records': [],
```

**改动 2——`_tool_node` 记录每次调用**:
每次工具执行后，根据 `execute_tool` 返回值中是否含有 `"error"` key 判断成败，追加记录：
```python
state['tool_call_records'].append({
    'tool': tool_name,
    'success': success,       # bool
    'error': result.get('error', None),  # 失败时的错误信息
})
```

**改动 3——新增 `_compute_tool_stats` 方法**:
`run()` 结束后调用，汇总统计：
```python
{
    "total_calls": 47,
    "success_rate": 0.87,
    "per_tool": {
        "trend_classifier": {
            "total": 12, "success": 12,
            "success_rate": 1.0, "errors": []
        },
        "return_calc": {
            "total": 3, "success": 0,
            "success_rate": 0.0,
            "errors": ["Execution failed in 'return_calc': ..."]
        },
        ...
    }
}
```
结果通过 `run()` 返回值中的 `tool_stats` 字段暴露，与 `all_results` 并列。

`per_tool.errors` 保留所有错误消息列表，便于后续"实证驱动的工具箱迭代"时直接定位问题工具和错误原因，无需翻日志。

### 5. Observation 一致性验证（verify.py 重写）

**文件**: `verify.py`

**背景**:
TimeART / TimeTBench 用 ReAct 方式让 LLM 自由推理：LLM 输出 Action 后，系统执行真实工具并注入 Observation；但如果 LLM 自己在推理文本里写了 Observation（幻觉），就会被直接当成工具结果使用。原来只有 `hallucinated = 'Observation:' in res_his` 这一个粗粒度的幻觉标志。

这个任务的目标是：**从历史记录中解析每一次工具调用和对应的 Observation，重新用真实代码执行一遍，比对两者是否一致**——从而量化 TimeART 中 Observation 的可靠程度。

**原 verify.py**:
只是一个简单的运行脚本，调用 `graph.verify(mode='image')` 做 LLM 问答验证。

**新 verify.py 的结构**:

保留了原有 LLM verifier 功能（封装为 `run_llm_verifier()`），并新增三个核心函数：

**① `parse_timeart_history(history)`**
解析 TimeART/TimeTBench 格式的历史字符串，提取所有 `(tool_name, args, recorded_output)` 三元组。
实现方式：用 `re.split(r'Action:\n', history)` 将历史分段，每段用正则分别提取 tool_input JSON 和 output JSON。
output 行是单行格式（`output: {"result": "..."}`），用 `.+` 贪婪匹配整行后 `json.loads()`。
验证：在真实数据上正确解析出 2 对 action/observation。

**② `extract_ts_from_question(question)`**
从 timetoolbench 结果的 `question` 字段中还原时序数据。
timetoolbench 在构建 question 时将原始 ts 数据用 Python `f-string` 拼入，形如 `ts1: [-0.256, 6.297, ...] ts2: [...]`。
用 `re.search(r'\bts1:\s*(\[[^\[\]]*\])', question)` 提取（`[^\[\]]` 确保只匹配平坦列表），再用 `ast.literal_eval()` 解析（不用 `json.loads`，因为 Python list repr 用单引号/没有引号的浮点数，不是合法 JSON）。
验证：正确还原双通道数据，ts1 前4个值精确匹配原始数据。

**③ `check_observation_consistency(result_file)`**
主函数，遍历 result JSON 中每条记录：
- 提取 timeseries → 构建 minimal state
- 解析 history 得到所有 (tool, args, recorded) 对
- 用 `execute_tool(state, ...)` 重新执行
- 对比结果（精确匹配 / 去空格匹配 / 不一致）
- 返回结构化报告：全局一致性率 + per-tool 分析 + 不一致样本

**返回报告结构**:
```
{
  total_calls: int,
  consistent_calls: int,
  consistency_rate: float,
  per_tool: {
    "stationarity_test": {total, consistent, rate, mismatches: [...]},
    ...
  },
  items: [{question_snippet, calls_parsed, calls_consistent, checks: [...]}, ...]
}
```

**新增入口**:
`__main__` 通过 `--mode obs_consistency` 或 `--mode llm_verify` 切换两种模式，默认为 `obs_consistency`，结果保存到 `result/consistency_report.json`。

**运行方式**:
```bash
python verify.py --mode obs_consistency --result_file result/timetoolbench_result_tse_1.json
```

## P0 完成情况汇总

| 任务 | 状态 | 文件 |
|------|------|------|
| 删除 graph.py 硬编码 hack | ✅ 完成 | `graph.py` |
| 修复 return_calc 返回类型 bug | ✅ 完成 | `tools.py` |
| 修复 return_calc 设计问题（多通道、时间索引） | ✅ 完成 | `tools.py` |
| 更新 return_calc TOOLCARD | ✅ 完成 | `tools.py` |
| 完成 Reporter prompt | ✅ 完成 | `prompts.py` |
| 完成 Reporter 实现 | ✅ 完成 | `agents.py` |
| 工具调用成功率记录（per call） | ✅ 完成 | `graph.py` |
| 工具调用成功率汇总（per run） | ✅ 完成 | `graph.py` |
| interpolate 工具 | ✅ 已有实现，无需补充 | `tools.py` |
| differencing 工具 | ✅ 已有实现，无需补充 | `tools.py` |
| Observation 一致性验证 | ✅ 完成 | `verify.py` |
