import os
import sys
import re
import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path
from utils import DATAPATH_BASE
from tools import execute_tool


# ─── Observation Consistency Check ───────────────────────────────────────────

def parse_timeart_history(history):
    """
    Parse (tool_name, arguments, recorded_output) triplets from a
    TimeART / TimeTBench history string.

    History format:
        Action:
        tool: [tool_name], tool_input: {"arg": val}

        Observation:
        tool: [tool_name],
        output: {"result": "..."}

    Returns list of dicts:
        {'tool': str, 'args': dict, 'recorded': dict}
    """
    pairs = []
    # Split on Action: markers (each segment = one action + its observation)
    segments = re.split(r'(?:^|\n)Action:\n', history)
    for seg in segments[1:]:
        # Extract tool name and arguments (flat JSON, no nested braces)
        action_m = re.match(r'tool:\s*\[([^\]]+)\]\s*,\s*tool_input:\s*(\{[^}]*\})', seg)
        if not action_m:
            continue
        tool_name = action_m.group(1).strip()
        try:
            args = json.loads(action_m.group(2))
        except json.JSONDecodeError:
            continue

        # Extract the Observation block that follows in the same segment
        # output is always a single line: output: {"result": "..."}
        obs_m = re.search(r'\nObservation:\ntool:\s*\[[^\]]+\],\noutput:\s*(.+)', seg)
        if not obs_m:
            continue
        try:
            recorded = json.loads(obs_m.group(1))
        except json.JSONDecodeError:
            continue

        pairs.append({'tool': tool_name, 'args': args, 'recorded': recorded})
    return pairs


def extract_ts_from_question(question):
    """
    Reconstruct timeseries and column names from a timetoolbench question string.
    Handles both single-channel (ts: [...]) and multi-channel (ts1: [...] ts2: [...]).
    Returns (timeseries_list, cols_list) or (None, None) if parsing fails.
    """
    # Multi-channel: ts1: [...] ts2: [...]
    ts1_m = re.search(r'\bts1:\s*(\[[^\[\]]*\])', question)
    ts2_m = re.search(r'\bts2:\s*(\[[^\[\]]*\])', question)
    if ts1_m and ts2_m:
        try:
            ts1 = ast.literal_eval(ts1_m.group(1))
            ts2 = ast.literal_eval(ts2_m.group(1))
            return [ts1, ts2], ['ts1', 'ts2']
        except (ValueError, SyntaxError):
            pass

    # Single channel: ts: [...]
    ts_m = re.search(r'\bts:\s*(\[[^\[\]]*\])', question)
    if ts_m:
        try:
            ts = ast.literal_eval(ts_m.group(1))
            return [ts], ['ts']
        except (ValueError, SyntaxError):
            pass

    return None, None


def build_minimal_state(timeseries, cols):
    """Build the minimal state dict needed by execute_tool."""
    df = pd.DataFrame({col: ts for col, ts in zip(cols, timeseries)})
    return {
        'data_item': {
            'timeseries': timeseries,
            'cols': cols,
            'masks': [[] for _ in cols],
            'df': df,
            'data': df.to_dict(orient='dict'),
            'derived_series': {},
        }
    }


def results_match(recorded, actual):
    """
    Compare a recorded observation dict with the actual re-executed output.
    Returns (is_consistent: bool, reason: str).
    """
    if 'error' in actual:
        return False, f"re-execution error: {actual['error']}"

    rec = recorded.get('result', '')
    act = actual.get('result', '')

    if rec == act:
        return True, 'exact match'
    if rec.strip() == act.strip():
        return True, 'whitespace-normalized match'

    return False, 'content mismatch'


def check_observation_consistency(result_file):
    """
    Load a TimeART / TimeTBench result JSON, re-execute every tool call found
    in each history, and compare the output with the recorded observation.

    Returns a report dict with:
      - total_calls: total number of (tool, obs) pairs parsed
      - consistent_calls: number that matched on re-execution
      - consistency_rate: consistent_calls / total_calls
      - per_tool: per-tool breakdown {tool: {total, consistent, rate, mismatches[]}}
      - items: per-item detail list
    """
    with open(result_file) as f:
        data = json.load(f)

    all_results = data.get('all_results', [])

    report = {
        'source': str(result_file),
        'total_items': len(all_results),
        'total_calls': 0,
        'consistent_calls': 0,
        'consistency_rate': None,
        'per_tool': {},
        'items': [],
    }

    for item in all_results:
        history  = item.get('history', '')
        question = item.get('question', '')

        timeseries, cols = extract_ts_from_question(question)
        if timeseries is None:
            report['items'].append({
                'question_snippet': question[:80],
                'skipped': True,
                'reason': 'could not parse timeseries from question',
            })
            continue

        state = build_minimal_state(timeseries, cols)
        pairs = parse_timeart_history(history)

        item_checks = []
        for p in pairs:
            tool_call = {'name': p['tool'], 'arguments': p['args']}
            actual    = execute_tool(state, tool_call)
            consistent, reason = results_match(p['recorded'], actual)

            report['total_calls'] += 1
            if consistent:
                report['consistent_calls'] += 1

            # per-tool accumulation
            t = p['tool']
            if t not in report['per_tool']:
                report['per_tool'][t] = {'total': 0, 'consistent': 0, 'rate': None, 'mismatches': []}
            report['per_tool'][t]['total'] += 1
            if consistent:
                report['per_tool'][t]['consistent'] += 1
            else:
                report['per_tool'][t]['mismatches'].append({
                    'args': p['args'],
                    'recorded_snippet': str(p['recorded'])[:120],
                    'actual_snippet':   str(actual)[:120],
                    'reason': reason,
                })

            item_checks.append({'tool': p['tool'], 'consistent': consistent, 'reason': reason})

        item_consistent = sum(1 for c in item_checks if c['consistent'])
        report['items'].append({
            'question_snippet': question[:80],
            'calls_parsed': len(pairs),
            'calls_consistent': item_consistent,
            'checks': item_checks,
        })

    # Finalise rates
    total = report['total_calls']
    report['consistency_rate'] = report['consistent_calls'] / total if total > 0 else None
    for t in report['per_tool']:
        pt = report['per_tool'][t]
        pt['rate'] = pt['consistent'] / pt['total'] if pt['total'] > 0 else None

    return report


def print_consistency_report(report):
    print(f"\n=== Observation Consistency Report ===")
    print(f"Source : {report['source']}")
    print(f"Items  : {report['total_items']}  (skipped: {sum(1 for it in report['items'] if it.get('skipped'))})")
    print(f"Calls  : {report['total_calls']}  consistent: {report['consistent_calls']}  rate: {report['consistency_rate']:.2%}" if report['consistency_rate'] is not None else f"Calls  : 0")
    print(f"\nPer-tool breakdown:")
    for tool, stat in sorted(report['per_tool'].items(), key=lambda x: x[1]['total'], reverse=True):
        rate_str = f"{stat['rate']:.2%}" if stat['rate'] is not None else 'N/A'
        print(f"  {tool:<30s}  total={stat['total']:3d}  consistent={stat['consistent']:3d}  rate={rate_str}")
        for mm in stat['mismatches'][:2]:  # show at most 2 mismatch examples per tool
            print(f"    [mismatch] args={mm['args']}  reason={mm['reason']}")
            print(f"      recorded : {mm['recorded_snippet']}")
            print(f"      actual   : {mm['actual_snippet']}")


# ─── LLM-based Verifier Runner (original) ────────────────────────────────────

def run_llm_verifier(config, mode, output_file):
    from graph import ToolTSGraph
    graph = ToolTSGraph(config)
    results = graph.verify(mode=mode)
    for result in results['all_results']:
        result['data_item']['visualizations'] = []
        result['data_item']['df'] = None
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved to {output_file}")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['llm_verify', 'obs_consistency'], default='obs_consistency')
    parser.add_argument('--result_file', type=str, default='result/timetoolbench_result_tse_1.json')
    parser.add_argument('--output_file', type=str, default='result/consistency_report.json')
    args = parser.parse_args()

    if args.mode == 'obs_consistency':
        report = check_observation_consistency(args.result_file)
        print_consistency_report(report)
        # Save full report (drop mismatch details to keep it lean)
        with open(args.output_file, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"\nFull report saved to {args.output_file}")

    elif args.mode == 'llm_verify':
        config = {
            'model': 'gpt-4o',
            'temperature': 1.0,
            'max_tokens': 4096,
            'api_key': 'sk-KEuOm1klq0oqthfAqkpojFADsOJpXrmNC1M5laHeAFrICozy',
            'base_url': 'https://jeniya.top/v1',
            'visualization_path': 'vis/tmp3',
            'dataset': 'Time-MQA',
            'data_path': DATAPATH_BASE + 'Time-MQA/Open_Ended_QA/open_ended_QA.csv',
            'process_num': 20,
        }
        run_llm_verifier(config, mode='image', output_file=args.output_file)
