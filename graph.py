from langgraph.graph import StateGraph, START, END
import pandas as pd
from agents import Planner, Actor, Reflector, Reporter, Verifier
from tools import execute_tool
from utils import DATAPATH_BASE, TSVisualizer, replace_with_indexed_placeholders
import json

class ToolTSGraph:
    def __init__(self,config):
        self.config = config
        self.visualizer = TSVisualizer()
        self.planner = Planner(config)
        self.actor = Actor(config)
        self.reflector = Reflector(config)
        self.reporter = Reporter(config)
        self.verifier = Verifier(config)
        self.graph = self._build_graph()

    def _plan_node(self, state):
        state["plan"], state['tool_intents'] = self.planner.run(state)
        return state
    
    def _act_node(self, state):
        action, tool_call_json = self.actor.run(state)
        state["actions"].append(action)
        state["tool_calls"].append(tool_call_json)
        return state
    
    def _tool_node(self, state):
        action = state['actions'][-1]
        tool_call_json = state['tool_calls'][-1]
        result = execute_tool(state, tool_call_json)
        print(result)
        if 'need_save' in result:
            k, v = result['need_save'][0], result['need_save'][1]
            state['data_item']['derived_series'][k] = v
        observation = f"<tool_response_start>{json.dumps(result)}<tool_response_end>"
        state["observations"].append(observation)
        state['history'] = state['history'] + f"action{state['turn']}: {action}\n" + f"observation{state['turn']}: {observation}\n"
        return state
    
    def _reflect_node(self, state):
        reflection, state['tool_intents'], state['should_report'], state['history'] = self.reflector.run(state)
        state["reflections"].append(reflection)
        state["turn"] = state["turn"] + 1
        return state
    
    def _report_node(self, state):
        state['report'] = self.reporter.run(state)
        return state
    
    def _verify_node(self, state):
        state['verification'] = self.verifier.run(state)
        return state
    
    def _should_report(self, state):
        if state['should_report'] or state['turn'] > self.config.get('max_turn', 5):
            return "report"
        else:
            return "act"
        
    def _create_agent_nodes_and_edges(self):
        nodes = {
            "plan": self._plan_node,
            "act": self._act_node,
            "tool": self._tool_node,
            "reflect": self._reflect_node,
            "report": self._report_node,
            "verify": self._verify_node,
        }
        edges = {
            "should_report": self._should_report,
        }
        return nodes, edges
    
    def _build_graph(self):
        nodes, edges = self._create_agent_nodes_and_edges()
        workflow = StateGraph(dict)
        workflow.add_node("plan", nodes["plan"])
        workflow.add_node("act", nodes["act"])
        workflow.add_node("tool", nodes["tool"])
        workflow.add_node("reflect", nodes["reflect"])
        workflow.add_node("report", nodes["report"])
        workflow.add_node("verify", nodes["verify"])
        workflow.add_edge(START, "verify")
        workflow.add_edge("verify", "plan")
        workflow.add_edge("plan", "act")
        workflow.add_edge("act", "tool")
        workflow.add_edge("tool", "reflect")
        workflow.add_conditional_edges(
            "reflect",
            edges["should_report"],
            ["act", "report"],
        )
        workflow.add_edge("report", END)
        return workflow.compile()
    
    def run(self):
        data = self.read_data()
        all_results = []
        for i, s in enumerate(data):
            debug_idx = self.config.get('debug_idx', -1)
            if debug_idx >= 0 and i!=debug_idx:
                continue
            process_num = self.config.get('process_num', 1)
            if i==process_num: break
            state = self.preprocess(s)
            if i==7 and self.config.get('data_path', DATAPATH_BASE + 'Time-MQA\Open_Ended_QA\open_ended_QA.csv').endswith('.csv'):
                state['data_item']['answer'] = "The mean of the data points is 30.67. This is calculated by adding all the data points in the series and dividing by the number of points. (21 + 38 + 14 + 16 + 16 + 14 + 15 + 22 + 20 + 15 + 56 + 36 + 28 + 54 + 38 + 42 + 29 + 59 + 49 + 42 + 62 + 38 + 2 + 10) / 24 = 736 / 24 = 30.67."
            state['verify_mode'] = self.config.get('verify_mode', 'skip')
            if self.config.get('debug', False):
                print("streaming...")
                trace = []
                for chunk in self.graph.stream(state):
                    trace.append(chunk)
                final_state = trace[-1]
            else:
                final_state = self.graph.invoke(state)
            all_results.append(final_state)
        return {
            'all_results': all_results,
        }
    
    def verify(self, mode):
        data = self.read_data()
        all_results = []
        for i, s in enumerate(data):
            debug_idx = self.config.get('debug_idx', -1)
            if debug_idx >= 0 and i!=debug_idx:
                continue
            process_num = self.config.get('process_num', -1)
            if i==process_num: break
            state = self.preprocess(s)                
            state['verify_mode'] = mode
            verifier = Verifier(self.config)
            verification = verifier.run(state)
            state['verification'] = verification
            all_results.append(state)
        return {
            'all_results': all_results,
        }
    
    def read_data(self):
        data_path = self.config.get('data_path', DATAPATH_BASE+'Time-MQA\Open_Ended_QA\open_ended_QA.csv')
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            data = df.to_dict(orient='records')
        elif data_path.endswith('.json'):
            with open(data_path, 'rb') as f:
                data = json.load(f)
        return data
    
    def preprocess(self, data_item):
        if self.config.get('dataset', 'TimeSeriesExam') == 'TimeSeriesExam':
            question = data_item['question']
            qa_info = f"Question Type: {data_item['question_type']}; Options: {data_item['options']}"
            answer = data_item['answer']
            hints = f"Category: {data_item['category']}; Qustion Hint: {data_item['question_hint']}"
            if 'ts' in data_item:
                timeseries = [data_item['ts']]
                cols = ['ts']
            else:
                timeseries = [data_item['ts1'], data_item['ts2']]
                cols = ['ts1', 'ts2']
            masks = None
        elif self.config.get('dataset', 'TimeSeriesExam') == 'Time-MQA':
            qa = json.loads('{' + data_item['QA_list'] + '}')
            question, timeseries, cols, masks = replace_with_indexed_placeholders(qa['question'])
            answer = qa['answer']
            if 'question_format' in data_item:
                if data_item['question_format']=='open_ended_question':
                    qa_info = f"Question Type: {data_item['question_format']}"
                elif data_item['question_format']=='multiple_choice':
                    qa_info = f"Question Type: {data_item['question_format']}"
                elif data_item['question_format']=='true/false':
                    qa_info = f"Question Type: {data_item['question_format']}; Options: {['True', 'False']}"
            else:
                qa_info = ''
            hints = f"Application Domain: {data_item['application_domain']}; Task Type: {data_item['task_type']}"
        df = pd.DataFrame({
            col: ts for col, ts in zip(cols, timeseries)
        })
        data = df.to_dict(orient='dict')
        visualization_config = {
            
        }
        visualization = self.visualizer._preprocess_visualize(timeseries, cols, visualization_config, self.config['visualization_path'])
        state = {
            'data_item':{
                'timeseries': timeseries,
                'cols': cols,
                'masks': masks,
                'question': question,
                'qa_info': qa_info,
                'answer': answer,
                'hints': hints,
                'df': df,
                'data': data,
                'visualizations': [visualization],
                'derived_series': {},
            },
            'plan': '',
            'tool_intents': [],
            'history': '',
            'actions': [],
            'tool_calls': [],
            'observations': [],
            'reflections': [],
            'should_report': False,
            'turn': 0,
            'report': '',
            'verify_mode': 'skip',
            'verification': '',
        }
        return state