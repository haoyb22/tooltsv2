import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import base64
import re

DATAPATH_BASE = 'D:\wangz\科研\\tooltsdata\\'

def replace_with_indexed_placeholders(question):
    matches = re.findall(r'\[([^\[\]]*?)\]', question)
    timeseries_list, masks_list = [], []
    
    for match in matches:
        nums, masks = [], []
        for x in match.split(','):
            x = "".join(x.split())
            if x!="'X'":
                x = float(x)
            else:
                masks.append(len(nums))
            nums.append(x)
        timeseries_list.append(nums)
        masks_list.append(masks)
    
    # 从后往前替换（避免索引错乱）
    parts = re.split(r'(\[.*?\])', question)
    new_parts = []
    cols = []
    idx = 1
    for part in parts:
        if part.startswith('[') and part.endswith(']'):
            new_parts.append(f'ts{idx}')
            cols.append(f'ts{idx}')
            idx += 1
        else:
            new_parts.append(part)
    
    new_question = ''.join(new_parts)
    if idx==2: 
        new_question = new_question.replace('ts1', 'ts')
        cols = ['ts']

    return new_question, timeseries_list, cols, masks_list

class TSVisualizer:
    def __init__(self) -> None:
        self.colors = plt.cm.tab10.colors

    def _preprocess_visualize(self, timeseries: list, cols: List[str], config: Dict[str, Any], output_path) -> str:
        plt.figure(figsize=(12, 6), facecolor='white')
            
        for i in range(len(cols)):
            plt.plot(timeseries[i], label=cols[i], linewidth=2,color=self.colors[i])
            
        # plt.title(config.get('title', 'Time Series Plot'))
        plt.xlabel(config.get('xlabel', 'Time'))
        plt.ylabel(config.get('ylabel', 'Value'))
        plt.legend()
        # plt.grid(True)

        ax = plt.gca()
        ax.set_facecolor("white")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, color="black", alpha=0.3)  # 添加格子
        ax.grid(True, which="minor", linestyle=":", linewidth=0.3, color="black", alpha=0.2)   # 次级格子

        # 增大字体
        plt.title(config.get('title', 'Time Series Plot'), fontsize=16, fontweight='bold')
        plt.xlabel(config.get('xlabel', 'Time'), fontsize=14, fontweight='bold')
        plt.ylabel(config.get('ylabel', 'Value'), fontsize=14, fontweight='bold')
            
        # 增大刻度标签字体
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)

            
        # Add additional elements as specified by LLM
        additional_elements = config.get('additional_elements', [])
        if 'rotate_x_labels' in additional_elements:
            plt.xticks(rotation=45)
            
        if isinstance(output_path, str):
            output_path = Path(output_path)
        save_path = output_path / f"{config.get('name', 'time_series_plot')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        img_b64_str = base64.b64encode(open(save_path, 'rb').read()).decode('utf-8')
        return img_b64_str