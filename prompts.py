from typing import List, Dict, Optional, Tuple
import numpy as np
from tools import TOOLCARD

def prompt_to_mmcontent(prompt, visualizations):
    p_spilt = prompt.split('<visualization>')
    content = []
    for i in range(len(visualizations)):
        content.append({"type": "text", "text": p_spilt[i]})
        content.append({'type': 'image_url', 'image_url': {"url": f"data:image/png;base64,{visualizations[i]}"}})
    content.append({"type": "text", "text": p_spilt[-1]})
    return content


def prepare_qa(data_item, with_answer=True):
    if data_item['qa_info']:
        given_qa = f"""### QUESTION ###
{data_item['question']}

### QA INFO ###
{data_item['qa_info']}"""
    else:
        given_qa = f"""### QUESTION ###
{data_item['question']}"""

    if with_answer:
        given_qa += f"""

### ANSWER ###
{data_item['answer']}"""
    return given_qa


PLAN_INSTRUCTION = """You are given time series data, a time series question and its final answer. Your task is to generate a plausible intermediate reasoning plan that explains *how* one could arrive at the given answer through a tool-aware analytical process.

The plan should:
- Be written as a single coherent paragraph in natural language. Brief and clear.
- Describe a step-by-step analytical process that logically connects the question to the provided answer.
- When external analysis with tools is needed, indicate it using a single JSON string representing a generic tool intent in the exact format including the label <tool_intent_start> and <tool_intent_end>:  
  `<tool_intent_start>{"index": int, "tool": "tool name", "intent": "tool intent"}<tool_intent_end>`  
  - "index": The index of the tool intent. It should be an integer starting from 0.
  - "tool": The name of the tool intented to be used.
  - "intent": A brief description of the task and the strategy of the tool.
- Use 1 to 5 such tool intents, placed naturally within the reasoning process. Don't place them seperately.
- Choose the most direct tool for the task. For example, if the task is 'forecasting', just use 'forecaster' or 'forecasting_tool' to forecast directly rather than using stats tools and reasoning by yourself.

"""

def create_plan_prompt(state):
    data_item = state['data_item']

    given_qa = prepare_qa(data_item)
    
    context = f"""
TIME SERIES DATA (as a Python dict, together with every column/metric/channel):
{data_item['data']}

VISUALIZATION:
<visualization>

HINTS:
{data_item['hints']}

GIVEN QA:
{given_qa}

"""
    return PLAN_INSTRUCTION + context + create_tools_info(mode='plan')


ACT_INSTRUCTION = """You are an analytical agent executing a time series task. You are given:
- **DATA**: A Python dictionary representing time series data. Maybe with visualization and captions.
- **QA**: A time series question and its final answer. The answer is for reference only.
- **Plan**: The high-level reasoning plan generated initially by the Planner and fixed by the Reflector.
- **History**: A sequence of past steps in the format:  
  action1 - observation1 - reflection1 - action2 - observation2 - reflection2 - ...

Each **action** consists of natural-language reasoning followed immediately by a single JSON string representing a tool call in the format: 
 `<tool_start>{"name": "tool_name", "arguments": {"arg1": value1, "arg2": value2, ...}}<tool_end>`

Each **observation** is the result of that tool call, wrapped as:  
 `<tool_response_start>result content<tool_response_end>`

Each **reflection** is a natural language report that explains the last observation, evaluates or adjusts the plan, and suggests what to do next.

Your current goal is to generate the **next action** based on information you have.

To generate your response:
- First, write a short paragraph of natural-language reasoning in first person. In this reasoning you can:
  - Consider the plan, the latest reflection, and the goal "generate a step of the reasoning process of QA".
  - Explain what to do next and why.
  - Choose and explain the exact tool and arguments to use.
- Then, output exactly one tool call in the following strict format including the label <tool_start> and <tool_end>:  
  `<tool_start>{"name": "tool_name", "arguments": {"arg1": value1, "arg2": value2, ...}}<tool_end>`
  - Additional hints: **Cross-tool parameter passing**: Some tools produce derived time series (e.g., rolling_stat, volatility, differencing). You can register their output by adding "register_as": "key_name" in the tool arguments. Then, subsequent tools can use this derived series as input by adding "source": "key_name" in their arguments. Example workflow: 1. Call rolling_stat with register_as="ma_5" to compute and register a 5-point moving average. 2. Call trend_classifier with source="ma_5" to classify the trend of the smoothed series.

Do not output anything else. Do not repeat past useless actions. The entire response must be: [reasoning text][tool_call].

"""

def create_act_prompt(state):
    plan = state['plan']
    history = state['history']
    
    data_item = state['data_item']
    given_qa = prepare_qa(data_item)
    
    context = f"""
TIME SERIES DATA (as a Python dict, together with every column/metric/channel):
{data_item['data']}

VISUALIZATION:
<visualization>

HINTS:
{data_item['hints']}

GIVEN QA:
{given_qa}

Here is your original Plan:
{plan}
Here are current tool intents fixed by the Reflector:
{state['tool_intents']}

Here is your History:
{history}
"""
    return ACT_INSTRUCTION + context + create_tools_info(mode='act')


REFLECT_INSTRUCTION = """You are a Reflector in a tool-augmented reasoning system for time series analysis. You are given:
- **DATA**: A Python dictionary representing time series data. Maybe with visualization and captions.
- **QA**: A time series question and its final answer. The answer is the ground truth.
- **Plan**: The high-level reasoning plan generated initially by the Planner and fixed by the Reflector.
- **History**: A sequence of past steps in the format:  
  action1 - observation1 - reflection1 - action2 - observation2 - reflection2 - ...

Each **action** consists of natural-language reasoning followed immediately by a single tool call in the format: 
 `<tool_start>{"name": "tool_name", "arguments": {"arg1": value1, "arg2": value2, ...}}<tool_end>`

Each **observation** is the result of that tool call, wrapped as:  
 `<tool_response_start>result content<tool_response_end>`

Your task is to generate the next **reflection**, which should be a report in natural language that include all the following parts:
1. Recent tool use:
   - Interprets the most recent observation in the context of the actor's stated intent (from the reasoning in the last action).
   - Evaluates whether the action produced a useful or valid result.
2. Plan adjustment:
   - Assesses whether the Plan is on track or needs adjustment. But if a tool get wrong answer just because of wrong parameters, it should not be considered as a reason to adjust the Plan.
   - If you need adjust the Plan, you should explain why followed by a single JSON string representing a generic tool intent in the exact format including the label <tool_intent_start> and <tool_intent_end>:  
        `<tool_intent_start>{"index": int, "tool": "tool name", "intent": "tool intent"}<tool_intent_end>`  
        - "index": The index of the tool intent. It should be an integer starting from 0 and following the previous index.
        - "tool": The name of the tool intented to be used.
        - "intent": A brief description of the task and the strategy of the tool.
    - If you have new tool intent, make sure to show its index in this part.
3. Suggest next step:
    - Suggests what kind of step the Actor should take next (e.g., continue as planned, retry with different parameters, switch strategy, or follow thw new tool intent).
4. Finish:
    - You should in this part generate a single JSON string including the label <finish_start> and <finish_end>: 
        `<finish_start>{"answer": "answer content", "option": "option"}<finish_end>`  
    - First try answer with the given history. If no answer is showed, the "answer" field should be an empty string "" and ignored.
    - Then you must follow the options below:
        - If you find that the history doesn't show a corresbonding answer, choose 'continue'.
        - If you find that the history shows a corresbonding answer: If it match the given ground truth answer (even if evidences are not so sufficient but at least 1 complete chain), choose 'true'. If you are not sure if it matches the ground truth answer (for example, the forcasting result is not so accurate), choose 'uncertain'. If it doesn't match the ground truth answer, choose 'false'.

Make sure generate a complete report of all parts. The format should be:
1. Recent tool use: ...
2. Plan adjustment: ...
3. Suggest next step: ...
4. Finish: ...

"""

def create_reflect_prompt(state):
    plan = state['plan']
    history = state['history']
    
    data_item = state['data_item']
    given_qa = prepare_qa(data_item)
    
    context = f"""
TIME SERIES DATA (as a Python dict, together with every column/metric/channel):
{data_item['data']}

VISUALIZATION:
<visualization>

HINTS:
{data_item['hints']}

GIVEN QA:
{given_qa}

Here is your original Plan:
{plan}
Here are current tool intents fixed by the Reflector:
{state['tool_intents']}

Here is your History:
{history}
"""
    return REFLECT_INSTRUCTION + context + create_tools_info(mode='reflect')


def create_tools_info(mode):
    tool_card_used = {}
    if mode=='plan':
        for tool in TOOLCARD:
            obj = TOOLCARD[tool]
            tool_card_used[tool] = obj['description']
    elif mode=='act' or mode=='reflect':
        tool_card_used = TOOLCARD
    tool_context = f"\n\nTOOL INFO:\n{tool_card_used}"
    return tool_context


VERIFY_INSTRUCTION = """You are an intelligent Time Series Reasoner capable
of performing time series analysis. 
Please answer the given question based on the provided data or visualizations.
If options are provided, choose the most appropriate one.

"""

def create_verify_prompt(state, mode='text'):
    data_item = state['data_item']
    given_qa = prepare_qa(data_item, with_answer=False)
    
    if mode == 'text':
        context = f"""
TIME SERIES DATA (as a Python dict, together with every column/metric/channel):
{data_item['data']}"""
    elif mode == 'image':
        context = f"""
VISUALIZATION:
<visualization>"""
    else:
        context = f"""
TIME SERIES DATA (as a Python dict, together with every column/metric/channel):
{data_item['data']}

VISUALIZATION:
<visualization>"""
        
    context += f"""

HINTS:
{data_item['hints']}

GIVEN QUESTION:
{given_qa}
"""
    return VERIFY_INSTRUCTION + context