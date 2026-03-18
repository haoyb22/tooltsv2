import os
import sys
import json
from datetime import datetime
from pathlib import Path
from utils import DATAPATH_BASE

from graph import ToolTSGraph

if __name__ == "__main__":
    config = {}
    config['model']='gpt-4o'
    config['temperature']=1.0
    config['max_tokens']=4096
    config['api_key']='sk-KEuOm1klq0oqthfAqkpojFADsOJpXrmNC1M5laHeAFrICozy'
    config['base_url']='https://jeniya.top/v1'
    config['visualization_path']='vis\\tmp3'
    config['dataset']='Time-MQA'
    config['data_path']=DATAPATH_BASE+'Time-MQA\Open_Ended_QA\open_ended_QA.csv'
    config['process_num'] = 20
    graph = ToolTSGraph(config)
    results = graph.verify(mode='image')
    for i, result in enumerate(results['all_results']):
        result['data_item']['visualizations'] = [] # to show without large base64 images
        result['data_item']['df'] = None # to save
    with open('result\\verify_image_mqa_20.json', 'w') as f:
        json.dump(results, f, indent=4)