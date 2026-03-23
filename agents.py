import os
import sys
import re
import numpy as np
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from prompts import prompt_to_mmcontent, create_plan_prompt, create_act_prompt, create_reflect_prompt, create_report_prompt, create_verify_prompt
from tools import execute_tool

class Agent:
    def __init__(self, config):
        self.llm = ChatOpenAI(
            model=config['model'],
            temperature=config['temperature'],
            max_tokens=config['max_tokens'],
            api_key=config['api_key'],
            base_url=config['base_url']
        )
        self.config = config

    def run(self, state):
        pass

class Planner(Agent):
    def __init__(self, config):
        super().__init__(config)

    def run(self, state):
        print("planner run")
        prompt = create_plan_prompt(state)
        #print(prompt)
        response = self.llm.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt_to_mmcontent(prompt, state['data_item']['visualizations']))
        ])
        plan = response.content
        print(plan)
        tool_intent_pattern = r'<tool_intent_start>\s*(.*?)\s*<tool_intent_end>'
        tool_intents_list = re.findall(tool_intent_pattern, plan, re.DOTALL)
        print(tool_intents_list)
        return plan, tool_intents_list
    
class Actor(Agent):
    def __init__(self, config):
        super().__init__(config)

    def run(self, state):
        print("actor run")
        prompt = create_act_prompt(state)
        while(True):
            response = self.llm.invoke([
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=prompt_to_mmcontent(prompt, state['data_item']['visualizations']))
            ])
            action = response.content
            print(action)
            tool_call_pattern = r'<tool_start>\s*(.*?)\s*<tool_end>'
            tool_calls = re.findall(tool_call_pattern, action, re.DOTALL)
            if len(tool_calls) != 1:
                continue
            else:
                tool_call_content = tool_calls[0]
                tool_call_json = json.loads(tool_call_content.strip())
                return action, tool_call_json
                
class Reflector(Agent):
    def __init__(self, config):
        super().__init__(config)

    def run(self, state):
        print("reflector run")
        prompt = create_reflect_prompt(state)
        #print(prompt)
        response = self.llm.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt_to_mmcontent(prompt, state['data_item']['visualizations']))
        ])
        reflection = response.content
        print(reflection)
        finish_pattern = r'<finish_start>\s*(.*?)\s*<finish_end>'
        finish_flags = re.findall(finish_pattern, reflection, re.DOTALL)
        should_report = state['should_report']
        if finish_flags: 
            print('checking finish flag')
            finish_content = finish_flags[0].strip()
            finish_json = json.loads(finish_content)
            if finish_json['option']=='continue' or finish_json['option']=='false': should_report = False
            else: should_report = True
            
        tool_intent_pattern = r'<tool_intent_start>\s*(.*?)\s*<tool_intent_end>'
        tool_intents_list = re.findall(tool_intent_pattern, reflection, re.DOTALL)
        state['tool_intents'] += tool_intents_list
        
        history = state['history'] + f"reflection{state['turn']}: {reflection}\n"
        return reflection, state['tool_intents'], should_report, history
    
class Reporter(Agent):
    def __init__(self, config):
        super().__init__(config)

    def run(self, state):
        print("reporter run")
        prompt = create_report_prompt(state)
        response = self.llm.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt_to_mmcontent(prompt, state['data_item']['visualizations']))
        ])
        report = response.content
        print(report)
        return report
    
class Verifier(Agent):
    def __init__(self, config):
        super().__init__(config)

    def run(self, state):
        print("verifier run")
        mode = state.get('verify_mode', 'skip')
        if mode == 'skip':
            return ''
        prompt = create_verify_prompt(state, mode)
        if mode == 'text':
            content = prompt
        else:
            content = prompt_to_mmcontent(prompt, state['data_item']['visualizations'])
        response = self.llm.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=content)
        ])
        verification = response.content
        print(verification)
        return verification