from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
import re
import json
import argparse
from tools import execute_tool
from utils import DATAPATH_BASE, replace_with_indexed_placeholders


BASE_URL = 'https://jeniya.top/v1'
open_ai_key = 'sk-KEuOm1klq0oqthfAqkpojFADsOJpXrmNC1M5laHeAFrICozy'

llm = ChatOpenAI(
            model='gpt-4o',
            temperature=1.0,
            max_tokens=4096,
            api_key=open_ai_key,
            base_url=BASE_URL
        )

def create_prompt_template(a,b,c,d):
    return """
You are an intelligent Time Series Reasoner capable
of performing time series analysis by invoking
appropriate tools step by step.
1. You should first understand the question and analyze
whether it is necessary to invoke the tool. If not, you
can directly give the final answer.
2. If tools are needed, you can utilize them to help
answer the question. Multiple tool calls are encouraged.
3. You should think step by step in ReAct-style, output
a structured reasoning trajectory that leads to the
final answer.
You have access to the following tools: """ +a+ """ The
only tools you may use are: """ +b+ """.

Here is an example:
The question is: Calculate and interpret the 3-point moving average of the following sequence [4875.91, 4871.64, 4875.75, 4879.98,
4878.27, 4880.68]

Thought:
I need to compute the 3-point moving average for the given sequence. This can be done using the [rolling_stat] tool
with a window size of 3 and the "mean" statistic.
Action:
tool: [rolling_stat], tool_input: {"stat": "mean", "window": 3, "step": 1}

Observation:
tool: [rolling_stat],
output: {"statistic": "mean", "window_size": 3, "step_size": 1, "rolling_results": {"channel_0": [{"window_start":
0, "window_end": 3, "mean": 4874.433333333333}, {"window_start": 1, "window_end": 4, "mean": 4875.79},
{"window_start": 2, "window_end": 5, "mean": 4878.0}, {"window_start": 3, "window_end": 6, "mean":
4879.643333333333 }]}}"}

Final Answer:
The 3-point moving averages are approximately [4874.43, 4875.79, 4878.00, 4879.64], indicating an upward trend in
the sequence.


You should follow these rules:
1. Thought, Action, Observation, Thought, ...(repeat), Final Answer is a step sequence.
2. Make sure you only output one step at a time. You can't output Observation step.
3. In Action, you should follow the format "tool: [tool_name], tool_input: [tool_input]" (seeing the example) to call a tool. And the tool_input should be a JSON string.
4. Observation step is the output of the tool and provided for you. You should not output it yourself.
5. Each time series is called a channel, named as "ts_1", "ts_2", etc, by their order in default. If there is only one channel, it is named as "ts". But if arguments don't include channels, you should not provide their names or data.
6. Final Answer should be strictly one of options if options are provided, and don't explain in this step. If you think no option is correct, choose the closest one. 

"Begin!"
The question is: """ +c+ d

tools_prompt = """
Numerical Operators

- series_info():
  Retrieves basic metadata of the timeseries, including sequence length (T), number of channels(C), and missing value statistics.

- datapoint_value(index):
  Returns the specific values of all channels at a given time index.

- summary_stats(start, end, stat):
  Calculates a specific statistic (mean, sum, max, min, std) for all channels over a defined index range [start, end).

- return_calc(t1,t2, kind):
  Computes the percentage return (“pct”) or absolute difference (“diff”) between two specific time indices.

- autocorr(lag):
  Computes the autocorrelation coefficient for each channel at a specified time lag to measure self-similarity.

- rolling_stat(stat, window, step):
  Computes rolling statistics (mean, sum, max, min, std) using a sliding window across the time series.

- quantile_value(q):  
  Calculates the empirical value at a specific quantile level (between 0 and 1) for each channel (e.g., q=0.5 for median).

- volatility(window):  
  Computes rolling volatility (calculated as the standard deviation of first differences) over a specified window size.


Pattern Detector

- trend_classifier(window):  
  Classifies trends in time series as "up", "down", or "flat"; supports global analysis or window-based segment analysis.

- seasonality_detector(max_period):  
  Detects periodic patterns and returns estimated period with seasonality strength ("strong" or "weak").

- change_point_detector(penalty_or_n_cp):  
  Detects structural breaks (change points) in mean or variance and returns the indices of these changes.

- noise_profile(window):  
  Labels noise type (e.g., "white", "red") based on autocorrelation tests; performed globally or over a specific window. window<10 will be ignored because of too short length.

- stationarity_test(test):  
  Tests stationarity using Augmented Dickey-Fuller or KPSS methods; returns status ("stationary"/"nonstationary") and test statistics.

- spike_detector(threshold, min_sep):  
  Detects and locates spikes or dips in the series based on amplitude threshold and minimum separation.


Correlation Analyzer

- channel_correlation(channel_1, channel_2, lag, method):  
  Calculates correlation (“Pearson”/“Spearman”) between two channels with an optional time lag.

- cross_correlation(channel_1, channel_2, max_lag):  
  Computes cross-correlation across multiple lags to find the optimal time alignment between two channels.  

- dtw_distance(channel_1, channel_2, distance_metric):  
  Measures similarity between two channels using Dynamic Time Warping (DTW); returns a distance where lower values indicate greater similarity.  
  - distance_metric must be 'euclidean' or'sqeuclidean'

- shape_similarity(channel_1, channel_2, norm):  
  Measures shape similarity between two channels using normalized correlation, invariant to amplitude scaling.  
  - norm: Typically 'zscore' (recommended)

- granger_causality(cause_channel, effect_channel, max_lag):  
  Tests if one channel statistically predicts another (Granger causality) within a specified maximum lag.  

"""

def build_prompt(question, answer, history):
    input = f"{question}\n\n"
    tools = tools_prompt
    tool_names = [
        "series_info",
        "datapoint_value",
        "summary_stats",
        "return_calc",
        "autocorr",
        "rolling_stat",
        "quantile_value",
        "volatility",
        "trend_classifier",
        "seasonality_detector",
        "change_point_detector",
        "noise_profile",
        "stationarity_test",
        "spike_detector",
        "channel_correlation",
        "cross_correlation",
        "dtw_distance",
        "shape_similarity",
        "granger_causality"
    ]
    tool_names = f"{tool_names}"
    prompt = create_prompt_template(tools, tool_names, input, history)
    return prompt

TOOL_PATTERN = r'tool:\s*(.*?)\s*,\s*tool_input:\s*({[^}]*})'

def solve(context, mode):
    pattern = TOOL_PATTERN
    if mode == 'qa':
      original_q, original_a = context['question'], context['answer']
      question, timeseries, cols, masks = replace_with_indexed_placeholders(original_q)
    elif mode == 'obj':
      original_q, original_a = context['question'], context['answer']
      timeseries = context['timeseries']
      cols = context['cols']
      masks = None
    df = pd.DataFrame({
                col: ts for col, ts in zip(cols, timeseries)
            })
    data = df.to_dict(orient='dict')
    state = {
        'data_item':{
                    'timeseries': timeseries,
                    'cols': cols,
                    'masks': masks,
                    'df': df,
                    'data': data,
                }
    }

    res_his = ""
    history = ""
    it = 0
    final_answer = ""
    while(True):
        if it>8:
            break
        it += 1
        try:
          response = llm.invoke([
                      SystemMessage(content="You are a helpful assistant."),
                      HumanMessage(content=build_prompt(original_q, original_a, history))
          ])
          content = response.content.strip()
          
          if 'Final Answer:' in content:
              history += content
              res_his += content
              final_answer = content.split('Final Answer:')[-1].strip()
              break

          last_content = content.split('Action:')[-1]
          
          match = re.search(pattern, last_content)
          if match and ('Observation:' not in last_content):
            tool_name = match.group(1)[1:-1]
            tool_input_json_str = match.group(2)
            tool_input_json = json.loads(tool_input_json_str.strip())
            tool_call_json = {
                "name": tool_name,
                "arguments": tool_input_json,
            }
            result = execute_tool(state, tool_call_json)
            history += content + "\n"
            res_his += content + "\n"
            history += f"\nObservation:\ntool: [{tool_name}],\noutput: {json.dumps(result)}\n\n"
          else:
            history += content + "\n"
            res_his += content + "\n"
        except:
          continue
    hallucinated = 'Observation:' in res_his
    return {
        "history": history,
        "hallucinated": hallucinated,
        "F": final_answer
    }

argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', type=str, default=DATAPATH_BASE+'TimeSeriesExam\qa_dataset.json')
argparser.add_argument('--result_name', type=str, default='tse_20')
args = argparser.parse_args()
data_path = args.data_path
result_name = args.result_name

if data_path.endswith('.csv'):
  df = pd.read_csv(data_path)
  data = df.to_dict(orient='records')
elif data_path.endswith('.json'):
  with open(data_path, 'rb') as f:
    data = json.load(f)
all_results = []
for i, data_item in enumerate(data):
  if i>=20:
    break
  if data_path.endswith('.csv'):
    qa = json.loads('{' + data_item['QA_list'] + '}')
    result = solve(qa, mode='qa')
    result['question'] = qa['question']
    result['answer'] = qa['answer']
  elif data_path.endswith('.json'):
    if 'ts' in data_item:
      timeseries = [data_item['ts']]
      cols = ['ts']
      question = data_item['question']+f" ts: {data_item['ts']} Options: {data_item['options']}"
    else:
      timeseries = [data_item['ts1'], data_item['ts2']]
      cols = ['ts1', 'ts2']
      question = data_item['question']+f" ts1: {data_item['ts1']} ts2: {data_item['ts2']} Options: {data_item['options']}"
    result = solve({'question': question, 'answer': data_item['answer'], 'timeseries': timeseries, 'cols': cols}, mode='obj')
    result['question'] = question
    result['answer'] = data_item['answer']
  all_results.append(result)
  print(result)
hal = 0
for result in all_results:
  if result['hallucinated']:
    hal += 1
final = {
   'all_results': all_results,
   'hal_rate': hal/len(all_results)
}
with open(f'result\\timeart_result_{result_name}.json', 'w') as f:
   json.dump(final, f, indent=4)