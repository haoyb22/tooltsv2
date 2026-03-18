import os
import re
import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, kpss, ccf, grangercausalitytests
from scipy.stats import linregress, pearsonr, spearmanr
from scipy.signal import find_peaks
import ruptures as rpt
from dtaidistance import dtw
from typing import *

from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.var import VAR
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.base import ForecastingHorizon
from sktime.detection.stray import STRAY

from sktime.detection.skchange_aseg import CAPA
from adtk.detector import AutoregressionAD, ThresholdAD, PcaAD

from sktime.detection.bs import BinarySegmentation

def col_idx(name: str, cols: List[str]) -> int:
    for i, col in enumerate(cols):
        if col == name:
            return i
    raise RuntimeError(f"Metric: {name} not found in cols={cols}")


def resolve_source(state, arguments):
    """Resolve data source: use derived_series if 'source' is specified, otherwise use original timeseries.

    Returns (timeseries, cols) where timeseries is a list of arrays and cols is a list of column names.
    If source is specified, it looks up state['data_item']['derived_series'][source_key].
    The derived_series entry should be a dict like: {'cols': [...], 'data': [[...], [...], ...]}
    """
    source_key = arguments.get('source', None)
    if source_key is None:
        return state['data_item']['timeseries'], state['data_item']['cols']

    derived = state['data_item'].get('derived_series', {})
    if source_key not in derived:
        raise ValueError(f"Derived series '{source_key}' not found. Available: {list(derived.keys())}")

    entry = derived[source_key]
    data = []
    cols = []
    for col in entry.keys():
        cols.append(col)
        data.append(entry[col])
    return data, cols



def forecasting_tool(state, arguments):
    timeseries, cols = state['data_item']['timeseries'], state['data_item']['cols']
    forecaster_name, metric_list, fh_list = arguments['forecaster_name'], arguments['metric_list'], arguments['fh_list']
    if forecaster_name == 'AutoARIMA':
        forecaster = AutoARIMA(sp=None, suppress_warnings=True, error_action="ignore")
    elif forecaster_name == 'VAR':
        forecaster = VAR()
    elif forecaster_name == 'Prophet':
        forecaster = Prophet(
            seasonality_mode="multiplicative",
        )
    else:
        forecaster = NaiveForecaster(strategy="last")
    mts_valid = True
    for fh in fh_list:
        if fh!=fh_list[0]:
            mts_valid = False
            break
    if mts_valid:
        forecasted_ts = pd.DataFrame({
                cols[i]: timeseries[col_idx(metric_list[i],cols)] for i in range(len(metric_list))
            })
        fh = ForecastingHorizon(list(range(1, fh_list[0] + 1)), is_relative=True)
        forecaster.fit(forecasted_ts, fh=fh)
        point_pred = forecaster.predict()
        return_value = point_pred.to_dict(orient='dict')
    else:
        results = {}
        for i in range(len(metric_list)):
            forecasted_ts = pd.Series(timeseries[col_idx(metric_list[i],cols)], dtype=np.float64)
            fh = ForecastingHorizon(list(range(1, fh_list[i] + 1)), is_relative=True)
            forecaster.fit(forecasted_ts, fh=fh)
            point_pred = forecaster.predict()
            results[metric_list[i]] = point_pred.tolist()
        return_value = results
    return return_value, f"The forecasting results are {return_value}"

def anomaly_detection_tool(state, arguments):
    timeseries, cols = state['data_item']['timeseries'], state['data_item']['cols']
    detector_name, detector_config, metric_list = arguments['detector_name'], arguments['detector_config'], arguments['metric_list']
    if detector_name == 'STRAY':
        detector = STRAY(**detector_config)
    elif detector_name == 'CAPA':
        detector = CAPA(**detector_config)
    elif detector_name == 'AutoregressionAD':
        detector = AutoregressionAD(**detector_config)
    elif detector_name == 'ThresholdAD':
        detector = ThresholdAD(**detector_config)
    else:
        detector = PcaAD(**detector_config)
    results = []
    for i, metric in enumerate(metric_list):
        detected_ts = pd.Series(timeseries[col_idx(metric,cols)], dtype=np.float64)
        if detector_name == 'STRAY':
            detector.fit(detected_ts)
            detected_ts_hat = detector.transform(detected_ts)
            anomalous_points = [int(i) for i, flag in enumerate(detected_ts_hat.to_numpy()) if bool(flag)]
            if anomalous_points:
                prompt = f"- For {metric}, the anomalous points are {anomalous_points}."
            else:
                prompt = f"- For {metric}, no anomaly is detected."
        elif detector_name == 'CAPA':
            detector.fit(detected_ts)
            anomaly_intervals = detector.predict(detected_ts)
            prompt = f"- For {metric}, the anomalous interval is left: {anomaly_intervals.ilocs[0].left} and right: {anomaly_intervals.ilocs[0].right}."
        else:
            anomalies = detector.fit_detect(detected_ts)
            anomalous_points = []
            for idx in range(1, len(anomalies)):
                if anomalies[idx] and not anomalies[idx-1]:
                    anomalous_points.append(idx)
            if anomalous_points:
                prompt = f"- For {metric}, the anomalous points are {anomalous_points}."
            else:
                prompt = f"- For {metric}, no anomaly is detected."
        results.append(prompt)
    return "The anomaly_detection results are: \n" + "\n".join(results)

def change_point_detection_tool(state, arguments):
    timeseries, cols = state['data_item']['timeseries'], state['data_item']['cols']
    metric_list, threshold_list = arguments['metric_list'], arguments['threshold_list']
    results = []
    for i, metric in enumerate(metric_list):
        detected_ts = pd.Series(timeseries[col_idx(metric,cols)], dtype=np.float64)
        bs = BinarySegmentation(threshold=threshold_list[i])
        change_points = bs.fit_predict(detected_ts)
        prompt = f"- For {metric}, the change points are {change_points}."
        results.append(prompt)
    return "The change_point_detection results are: \n" + "\n".join(results)


## Numerical Operators

def series_info(state, arguments):
    timeseries, cols, masks = state['data_item']['timeseries'], state['data_item']['cols'], state['data_item']['masks']
    results = {}
    for col in cols:
        results[col] = {
            "length": len(timeseries[col_idx(col,cols)]),
            "missing_value_indices": masks[col_idx(col,cols)],
        }
    return f"Number of channels: {len(cols)}. The series information is: {results}"

def datapoint_value(state, arguments):
    timeseries, cols = state['data_item']['timeseries'], state['data_item']['cols']
    index = arguments['index']
    results = {}
    for col in cols:
        results[col] = timeseries[col_idx(col,cols)][index]
    return f"The datapoint values of timeseries at index {index} are: {results}"

def summary_stats(state, arguments):
    timeseries, cols = resolve_source(state, arguments)
    start, end = arguments['start'], arguments['end']
    stat = arguments.get('stat', 'default')
    results = {}
    for col in cols:
        features = {}
        if stat == "mean" or stat == "default":
            features["mean"] = np.mean(timeseries[col_idx(col,cols)][start:end])
        if stat == "sum" or stat == "default":
            features["sum"] = np.sum(timeseries[col_idx(col,cols)][start:end])
        if stat == "min" or stat == "default":
            features["min"] = np.min(timeseries[col_idx(col,cols)][start:end])
        if stat == "max" or stat == "default":
            features["max"] = np.max(timeseries[col_idx(col,cols)][start:end])
        if stat == "std" or stat == "default":
            features["std"] = np.std(timeseries[col_idx(col,cols)][start:end])
        results[col] = features
    return f"The summary statistics of timeseries are: {results}"

def return_calc(state, arguments):
    t1, t2, kind = arguments['t1'], arguments['t2'], arguments['kind']
    if kind == "diff":
        return t2 - t1
    elif kind == "pct":
        if t1 == 0:
            raise ValueError("t1 cannot be zero when computing percentage return.")
        return (t2 - t1) / t1
    else:
        raise ValueError("kind must be 'pct' or 'diff'")
    
def autocorr(state, arguments):
    timeseries, cols = resolve_source(state, arguments)
    df = pd.DataFrame({
            col: ts for col, ts in zip(cols, timeseries)
        })
    lag = arguments['lag']
    corrs = df.apply(lambda col: col.autocorr(lag=lag)).values.tolist()
    results = {}
    for col, corr in zip(cols, corrs):
        results[col] = corr
    return results, f"The autocorrelation of timeseries at lag {lag} are: {results}"

def rolling_stat(state, arguments):
    timeseries, cols = resolve_source(state, arguments)
    stat, window, step = arguments['stat'], arguments['window'], arguments['step']
    
    stat_funcs = {'mean': np.mean, 'sum': np.sum, 'max': np.max, 'min': np.min, 'std': np.std}
    func = stat_funcs[stat]

    results = {}
    for col in cols:
        ts = np.array(timeseries[col_idx(col,cols)])
        T = len(ts)
        if window > T:
            results[col] = {'error': 'Window size too large for time series.'}
            continue
        windows = sliding_window_view(ts, window_shape=window)[::step]  # (n_win, window)
        stats = func(windows, axis=1)
        results[col] = stats.tolist()

    return results, f"The rolling {stat} of timeseries with window size {window} and step size {step} are: {results}"

def quantile_value(state, arguments):
    timeseries, cols = resolve_source(state, arguments)
    q = arguments['q']
    timeseries = np.asarray(timeseries)
    qs = np.quantile(timeseries, q, axis=1)

    results = {}
    if np.isscalar(q):
        for col, q_val in zip(cols, qs):
            results[col] = q_val
    else:
        for col, q_vals in zip(cols, qs.T):
            results[col] = q_vals.tolist()

    return f"The quantile values of timeseries at quantile {q} are: {results}"

def volatility(state, arguments):
    timeseries, cols = resolve_source(state, arguments)
    window = arguments['window']

    results = {}
    for col in cols:
        ts = np.array(timeseries[col_idx(col,cols)])
        T = len(ts)
        if window < 2 or window > T:
            results[col] = {'error': 'Invalid window size.'}
            continue
        diffs = np.diff(ts)  # (T-1,)
        # Window on diffs: each window has (window - 1) points
        win_size = window - 1
        if win_size > len(diffs):
            results[col] = {'error': 'Window size too large for time series.'}
            continue
        # Create sliding windows over diffs
        diff_windows = sliding_window_view(diffs, window_shape=win_size)
        vols = np.std(diff_windows, axis=1, ddof=1)  # (n_windows,)
        results[col] = vols.tolist()

    return results, f"The rolling volatility of timeseries with window size {window} are: {results}"
        

def interpolate(state, arguments):
    timeseries, cols, masks = state['data_item']['timeseries'], state['data_item']['cols'], state['data_item']['masks']
    method = arguments.get('method', 'linear')
    channel = arguments.get('channel', None)

    target_cols = [channel] if channel else cols

    results = {}
    for col in target_cols:
        ts = np.array(timeseries[col_idx(col, cols)], dtype=np.float64)
        missing_indices = masks[col_idx(col, cols)]

        if len(missing_indices) == 0:
            results[col] = {'interpolated_count': 0, 'message': 'No missing values found.'}
            continue

        ts_series = pd.Series(ts)
        ts_series.iloc[missing_indices] = np.nan

        if method == 'linear':
            filled = ts_series.interpolate(method='linear', limit_direction='both')
        elif method == 'spline':
            order = arguments.get('order', 3)
            try:
                filled = ts_series.interpolate(method='spline', order=order, limit_direction='both')
            except Exception:
                filled = ts_series.interpolate(method='linear', limit_direction='both')
        elif method == 'ffill':
            filled = ts_series.ffill().bfill()
        elif method == 'bfill':
            filled = ts_series.bfill().ffill()
        elif method == 'mean':
            mean_val = ts_series.mean()
            filled = ts_series.fillna(mean_val)
        else:
            raise ValueError(f"Unsupported interpolation method: {method}. Use 'linear', 'spline', 'ffill', 'bfill', or 'mean'.")

        filled_values = {int(idx): float(filled.iloc[idx]) for idx in missing_indices}
        results[col] = {
            'interpolated_count': len(missing_indices),
            'missing_indices': missing_indices,
            'filled_values': filled_values,
            'method': method,
        }

    return f"The interpolation results are: {results}"


def differencing(state, arguments):
    timeseries, cols = resolve_source(state, arguments)
    order = arguments.get('order', 1)
    channel = arguments.get('channel', None)

    if order < 1:
        raise ValueError("Differencing order must be at least 1.")

    target_cols = [channel] if channel else cols

    results = {}
    return_value = {}
    for col in target_cols:
        ts = np.array(timeseries[col_idx(col, cols)], dtype=np.float64)
        T = len(ts)

        if order >= T:
            results[col] = {'error': f'Differencing order {order} is too large for series length {T}.'}
            continue

        diff_ts = ts.copy()
        for _ in range(order):
            diff_ts = np.diff(diff_ts)

        results[col] = {
            'order': order,
            'original_length': T,
            'differenced_length': len(diff_ts),
            'values': diff_ts.tolist(),
        }
        return_value[col] = diff_ts.tolist()

    return return_value, f"The {order}-order differencing results are: {results}"


## Pattern Detector

def _classify_segment(ts, time_idx, slope_threshold, p_threshold):
    """Classify one segment using linear regression"""
    if len(ts) < 2:
        return "flat"
    
    # Handle constant series
    if np.all(ts == ts[0]):
        return "flat"

    try:
        slope, intercept, r_value, p_value, std_err = linregress(time_idx, ts)
    except Exception:
        return "flat"

    # Option 1: Use statistical significance (p-value)
    if len(ts) >= 3 and p_value < p_threshold:
        if slope > slope_threshold:
            return "up"
        elif slope < -slope_threshold:
            return "down"
        else:
            return "flat"
    else:
        # Option 2: Use slope magnitude only (fallback for small windows)
        if slope > slope_threshold:
            return "up"
        elif slope < -slope_threshold:
            return "down"
        else:
            return "flat"

def trend_classifier(state, arguments):
    timeseries, cols = resolve_source(state, arguments)
    window = arguments.get('window', None)
    p_threshold = arguments.get('p_threshold', 0.05)

    def classify_single(ts):
        T = len(ts)
        time_idx = np.arange(T)
        """Classify trend for a single time series (length T)"""
        slope_threshold = arguments.get('slope_threshold', (np.max(ts)-np.min(ts))/100)
        if window is None:
            # Global trend
            return _classify_segment(ts, time_idx, slope_threshold, p_threshold)
        else:
            # Rolling window
            if window > T:
                raise ValueError("Window larger than time series.")
            n_segments = T - window + 1
            labels = []
            for i in range(n_segments):
                seg_ts = ts[i:i + window]
                seg_time = time_idx[i:i + window]
                label = _classify_segment(seg_ts, seg_time, slope_threshold, p_threshold)
                labels.append(label)
            return labels

    results = {}
    for col in cols:
        results[col] = classify_single(np.array(timeseries[col_idx(col,cols)]))

    if window is None:
        return f"The global trend of timeseries are: {results}"
    
    return f"The trend of timeseries with window size {window} are: {results}"

def seasonality_detector(state, arguments):
    timeseries, cols = resolve_source(state, arguments)
    max_period = arguments.get('max_period', 100)
    strong_threshold = arguments.get('strong_threshold', 0.6)
    weak_threshold = arguments.get('weak_threshold', 0.2)

    results = {}
    for col in cols:
        ts = np.array(timeseries[col_idx(col,cols)])
        T = len(ts)
        if max_period >= T:
            max_period_num = T - 1
        elif max_period < 2:
            max_period_num = 2
        else:
            max_period_num = max_period

        period, strength = 'none', 'none'

        x = np.arange(len(ts))
        slope, intercept, _, _, _ = linregress(x, ts)
        detrended = ts - (slope * x + intercept)

        acf_vals = acf(detrended, nlags=max_period_num, fft=True)

        peaks, properties = find_peaks(acf_vals[1:], height=weak_threshold/2)
        peaks += 1  # adjust index because we skipped lag=0

        if len(peaks) == 0:
            results[col] = {'period': period, 'strength': strength}
            continue

        best_peak_idx = np.argmax(acf_vals[peaks])
        best_lag = peaks[best_peak_idx]
        best_acf = acf_vals[best_lag]

        if best_acf >= strong_threshold:
            strength = 'strong'
            period = best_lag
        elif best_acf >= weak_threshold:
            strength = 'weak'
            period = best_lag

        results[col] = {'period': period, 'strength': strength}

    return f"The seasonality of timeseries are: {results}"

def change_point_detector(state, arguments):
    timeseries, cols = resolve_source(state, arguments)
    penalty = arguments.get('penalty', None)
    n_cp = arguments.get('n_cp', None)
    cost_func = arguments.get('cost_func', 'l2')
    strategy = arguments.get('strategy', 'Pelt')

    results = {}
    for col in cols:
        ts = np.array(timeseries[col_idx(col,cols)])
        T = len(ts)
        if strategy == 'Pelt':
            algo = rpt.Pelt(model=cost_func).fit(ts)
            if penalty is None:
                sigma2 = np.var(ts)
                bkp_indices = algo.predict(pen=3 * sigma2 * np.log(T))
            else:
                bkp_indices = algo.predict(pen=penalty)
        else:
            algo = rpt.Binseg(model=cost_func).fit(ts)
            if n_cp:
                bkp_indices = algo.predict(n_bkps=n_cp)
            else:
                raise ValueError("n_cp must be specified for Binseg strategy.")

        change_points = [int(cp) for cp in bkp_indices[:-1]]
        results[col] = change_points
    
    return f"The change points of timeseries are: {results}"

def _classify_noise_global(ts, max_lag=10, alpha=0.05):
    """Classify noise type for a single time series."""
    n = len(ts)
    if n < 10:
        return "unknown"

    # Step 1: Remove linear trend (important!)
    x = np.arange(n)
    slope, intercept, _, _, _ = linregress(x, ts)
    detrended = ts - (slope * x + intercept)

    # Step 2: Compute ACF up to max_lag
    # Use unbiased=False, fft=True for speed
    acf_vals, confint = acf(detrended, nlags=min(max_lag, n-1), 
                            fft=True, alpha=alpha)
    
    # confint shape: (nlags+1, 2); confidence interval for each lag
    # For white noise, ACF(lag>=1) should be within [-z*se, z*se]

    # Step 3: Check lags >= 1
    lags_to_check = np.arange(1, len(acf_vals))
    if len(lags_to_check) == 0:
        return "white"

    # Get confidence bounds (approximate standard error for white noise: 1/sqrt(n))
    # But we use the returned confint from acf()
    lower = confint[lags_to_check, 0] - acf_vals[0]  # acf_vals[0] is ~1
    upper = confint[lags_to_check, 1] - acf_vals[0]
    # Actually, simpler: check if |acf| > 1.96 / sqrt(n)
    se = 1.0 / np.sqrt(n)
    z = 1.96  # for alpha=0.05

    significant = np.abs(acf_vals[lags_to_check]) > z * se
    n_significant = np.sum(significant)

    if n_significant == 0:
        return "white"

    # Step 4: Check sign and decay of first few ACFs
    first_acfs = acf_vals[1:min(4, len(acf_vals))]  # lags 1,2,3

    # Majority positive?
    pos_ratio = np.mean(first_acfs > 0)
    neg_ratio = np.mean(first_acfs < 0)

    if pos_ratio >= 0.7:
        # Positive autocorrelation → red or pink
        # Check decay: if ACF(1) is large and decays slowly → red
        if acf_vals[1] > 0.5:
            return "red"
        else:
            return "pink"
    elif neg_ratio >= 0.7:
        return "blue"
    else:
        return "colored"  # generic non-white

def noise_profile(state, arguments):
    timeseries, cols = resolve_source(state, arguments)
    window = arguments.get('window', None)
    max_lag = arguments.get('max_lag', 20)
    alpha = arguments.get('alpha', 0.05)

    def classify_single(ts):
        T = len(ts)
        if window is None:
            return _classify_noise_global(ts, max_lag, alpha)
        else:
            if window > T:
                raise ValueError("Window larger than series length.")
            labels = []
            for i in range(T - window + 1):
                seg = ts[i:i + window]
                label = _classify_noise_global(seg, max_lag, alpha)
                labels.append(label)
            return labels

    results = {}
    for col in cols:
        results[col] = classify_single(np.array(timeseries[col_idx(col,cols)]))

    if window is None:
        return f"The global noise of timeseries are: {results}"
    
    return f"The noise of timeseries with window size {window} are: {results}"

def stationarity_test(state, arguments):
    timeseries, cols = resolve_source(state, arguments)
    test = arguments.get('test', 'adf')
    alpha = arguments.get('alpha', 0.05)
    
    results = {}
    for col in cols:
        ts = np.array(timeseries[col_idx(col,cols)])
        if len(ts) < 4:
            results[col] = {'error': 'Time series too short for stationarity test tool.'}
            continue
        result = {}
        if test.lower() == 'adf':
            adf_result = adfuller(ts, regression='ct', autolag='AIC')
            statistic, p_value, lags, nobs, critical_values, icbest = adf_result
            status = 'stationary' if p_value < alpha else 'nonstationary'
            result['adf'] = {
                    'status': status,
                    'p_value': float(p_value),
                    'statistic': float(statistic),
                    'lags_used': int(lags),
                    'critical_values': {k: float(v) for k, v in critical_values.items()}
                }
        elif test.lower() == 'kpss':
            statistic, p_value, lags, critical_values = kpss(ts, regression='ct', nlags="auto")
            status = 'nonstationary' if p_value < alpha else 'stationary'
            result['kpss'] = {
                    'status': status,
                    'p_value': float(p_value),
                    'statistic': float(statistic),
                    'lags_used': int(lags),
                    'critical_values': {str(k): float(v) for k, v in critical_values.items()}
                }
        else:
            raise ValueError("test must be 'adf' or 'kpss'")
        results[col] = result

    return f"The stationarity of timeseries are: {results}"

def spike_detector(state, arguments):
    timeseries, cols = resolve_source(state, arguments)
    threshold = arguments.get('threshold', 2)
    min_sep = arguments.get('min_sep', 5)
    relative = arguments.get('relative', True)

    results = {}
    for col in cols:
        ts = np.array(timeseries[col_idx(col,cols)])
        ts_centered = ts - np.mean(ts)

        if relative:
            sigma = np.std(ts_centered)
            if sigma == 0:
                thresh = np.inf  # no variation → no spikes
            else:
                thresh = threshold * sigma
        else:
            thresh = abs(threshold)
        
        peak_idx, _ = find_peaks(ts_centered, height=thresh, distance=min_sep)
        spikes = peak_idx.tolist()

        dip_idx, _ = find_peaks(-ts_centered, height=thresh, distance=min_sep)
        dips = dip_idx.tolist()

        results[col] = {'spikes': spikes, 'dips': dips}

    return f"The spikes and dips of timeseries are: {results}"


## Correlation Analyzer

def _corr(x, y, lag, method):
    n1, n2 = len(x), len(y)

        # Determine valid overlapping region based on lag
    if lag >= 0:
        # x[t] vs y[t + lag] → t from 0 to n1-1, t+lag from lag to lag+n1-1
        # Require t+lag < n2 → t < n2 - lag
        max_t = min(n1, n2 - lag)
        if max_t <= 0:
            return {'correlation': np.nan, 'p_value': np.nan, 'n_used': 0, 'lag': lag}
        x_valid = x[:max_t]
        y_valid = y[lag:lag + max_t]
    else:
        # lag < 0: x[t] vs y[t + lag] = y[t - |lag|]
        # t must be >= |lag|, and t < n1
        offset = -lag  # positive
        max_t = min(n1 - offset, n2)
        if max_t <= 0:
            return {'correlation': np.nan, 'p_value': np.nan, 'n_used': 0, 'lag': lag}
        x_valid = x[offset:offset + max_t]
        y_valid = y[:max_t]

    n_used = len(x_valid)
    if n_used < 2:
        return {'correlation': np.nan, 'p_value': np.nan, 'n_used': n_used, 'lag': lag}

    # Compute correlation
    method = method.lower()
    if method == 'pearson':
        corr, pval = pearsonr(x_valid, y_valid)
    elif method == 'spearman':
        corr, pval = spearmanr(x_valid, y_valid)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

    results = {
        'correlation': float(corr),
        'p_value': float(pval),
        'n_used': n_used,
        'lag': lag
    }

    return results

def channel_correlation(state, arguments):
    timeseries, cols = state['data_item']['timeseries'], state['data_item']['cols']
    channel_1, channel_2 = arguments['channel_1'], arguments['channel_2']
    lag = arguments.get('lag', 0)
    method = arguments.get('method', 'pearson')

    x = np.array(timeseries[col_idx(channel_1,cols)])
    y = np.array(timeseries[col_idx(channel_2,cols)])
    results = _corr(x, y, lag, method)

    return f"The correlation between {channel_1} and {channel_2} with lag {lag} is {results['correlation']:.3f} (p-value={results['p_value']:.3f}, n={results['n_used']})."

def cross_correlation(state, arguments):
    timeseries, cols = state['data_item']['timeseries'], state['data_item']['cols']
    channel_1, channel_2 = arguments['channel_1'], arguments['channel_2']
    max_lag = arguments.get('max_lag', 50)

    x = np.array(timeseries[col_idx(channel_1,cols)])
    y = np.array(timeseries[col_idx(channel_2,cols)])
    
    n1, n2 = len(x), len(y)
    min_len = min(n1, n2)
    if min_len == 0:
        raise ValueError("Empty input.")
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")
    max_lag = min(max_lag, min_len - 1)
    
    x = x[:min_len]
    y = y[:min_len]
    
    ccf_pos = ccf(x, y, fft=True, nlags=max_lag+1)
    ccf_neg_full = ccf(y, x, fft=True, nlags=max_lag+1)
    ccf_neg = ccf_neg_full[1:max_lag + 1]

    correlations = np.concatenate([
        ccf_neg[::-1],   # reverse: now τ=-max_lag ... τ=-1
        ccf_pos          # τ=0 ... τ=max_lag
    ])

    lags = np.arange(-max_lag, max_lag + 1)

    # Find best lag (handle NaNs safely)
    if np.all(np.isnan(correlations)):
        best_lag = 0
        max_corr = np.nan
    else:
        # Use nanargmax to ignore NaNs
        best_idx = np.nanargmax(correlations)
        best_lag = lags[best_idx]
        max_corr = correlations[best_idx]

    results = {
        'best_lag': int(best_lag),
        'max_correlation': float(max_corr),
        'lags': lags,
        'correlations': correlations,
    }
    
    # best_lag k means x[t+k] <-> y[t]
    return f"The cross-correlation between {channel_1} and {channel_2} is {max_corr:.3f} at lag {best_lag} (n={min_len})."

def dtw_distance(state, arguments):
    timeseries, cols = state['data_item']['timeseries'], state['data_item']['cols']
    channel_1, channel_2 = arguments['channel_1'], arguments['channel_2']
    distance_metric = arguments.get('distance_metric', 'euclidean')

    x = np.array(timeseries[col_idx(channel_1,cols)])
    y = np.array(timeseries[col_idx(channel_2,cols)])

    n1, n2 = len(x), len(y)
    min_len = min(n1, n2)
    if min_len == 0:
        raise ValueError("Empty input.")
    
    if distance_metric == 'euclidean':
        dist = dtw.distance(x, y)
    elif distance_metric == 'sqeuclidean':
        dist = dtw.distance(x, y, use_squared=True)
    else:
        raise ValueError("distance_metric must be 'euclidean' or'sqeuclidean'")
    
    return f"The DTW distance between {channel_1} and {channel_2} is {dist:.3f}"

def shape_similarity(state, arguments):
    timeseries, cols = state['data_item']['timeseries'], state['data_item']['cols']
    channel_1, channel_2 = arguments['channel_1'], arguments['channel_2']

    x = np.array(timeseries[col_idx(channel_1,cols)])
    y = np.array(timeseries[col_idx(channel_2,cols)])
    results = _corr(x,y,lag=0,method='pearson')

    return f"The shape similarity between {channel_1} and {channel_2} is {results['correlation']:.3f} (p-value={results['p_value']:.3f}, n={results['n_used']})."

def granger_causality(state, arguments):
    timeseries, cols = state['data_item']['timeseries'], state['data_item']['cols']
    cause_channel, effect_channel = arguments['cause_channel'], arguments['effect_channel']
    max_lag = arguments.get('max_lag', 20)
    alpha = arguments.get('alpha', 0.05)

    cause = np.array(timeseries[col_idx(cause_channel,cols)])
    effect = np.array(timeseries[col_idx(effect_channel,cols)])

    try:
        cause = np.diff(cause)
        effect = np.diff(effect)
    except:
        raise ValueError("Input channels are too short to compute the difference.")

    if len(cause) != len(effect):
        raise ValueError("Both channels must have the same length.")
    
    T = len(cause)
    
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1.")
    
    if T <= 2 * max_lag:
        raise ValueError(f"Time series too short for max_lag={max_lag}. Need more than {2 * max_lag} samples.")

    data = np.column_stack([effect, cause])
    results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    f_test = results[max_lag][0]['ssr_ftest']  # (F-statistic, p-value, df1, df2)
    f_stat, p_val = f_test[0], f_test[1]

    results = {
        'p_value': float(p_val),
        'f_statistic': float(f_stat),
        'reject_null': bool(p_val < alpha),
        'max_lag': max_lag
    }

    return f"The Granger causality between cause: {cause_channel} and effect: {effect_channel} is {'rejected' if results['reject_null'] else 'not rejected'} at p-value {results['p_value']:.3f} (F-statistic={results['f_statistic']:.3f}, max_lag={results['max_lag']})."


## Forecasting and Anomaly Detection

#def anomaly_detection(state, arguments):

#def forecaster(state, arguments):



## TOOLBOX AND TOOLCARD

forecasting_tool_card = {
    "description": "Performs time series forecasting on specified metrics using a selected forecaster. Returns point forecasts as a dictionary mapping metric names to their predicted values.",
    "parameters": {
        "properties": {
            "forecaster_name": {
                "type": "string",
                "description": "The name of the forecasting model to use. Supported options include 'AutoARIMA', 'VAR', 'Prophet', or any other value (defaults to NaiveForecaster)."
            },
            "metric_list": {
                "type": "list",
                "description": "A list of metric names (strings) corresponding to columns in the input time series data that should be forecasted."
            },
            "fh_list": {
                "type": "list",
                "description": "A list of integers specifying the forecast horizon (number of future time steps to predict) for each metric in metric_list. If all values are identical, multivariate forecasting is attempted (if supported by the forecaster); otherwise, each metric is forecasted independently."
            },
            "register_as": {
                "type": "string",
                "description": "Optional. If provided, the output series will be registered as a derived series with this key, which can be used as 'source' in subsequent tool calls."
            },
        },
        "required": ["forecaster_name", "metric_list", "fh_list"]
    },
    "metadata": {}
}

anomaly_detection_tool_card = {
    "description": "Detects anomalies in specified time series metrics using a chosen anomaly detection algorithm. Returns a textual summary indicating anomalous points or intervals for each metric.",
    "parameters": {
        "properties": {
            "detector_name": {
                "type": "string",
                "description": "The name of the anomaly detection algorithm to use. Supported detectors and their purposes:\n"
                               "- 'STRAY': Detects general point anomalies in univariate time series. Config should be an empty dict: {}.\n"
                               "- 'CAPA': Detects anomalous subsequences (intervals) in univariate time series. Config should be an empty dict: {}.\n"
                               "- 'AutoregressionAD': Detects anomalies based on deviations from autoregressive behavior. Config should be an empty dict: {}.\n"
                               "- 'ThresholdAD': Detects anomalies that fall outside user-specified lower and upper bounds. Config must be a dict with keys 'low' and 'high', e.g., {'low': 10, 'high': 90}.\n"
                               "- Any other value: Defaults to 'PcaAD', which detects anomalies in multivariate time series using Principal Component Analysis. Config should be an empty dict: {}."
            },
            "detector_config": {
                "type": "dict",
                "description": "Configuration parameters for the selected detector. The required content depends on the detector_name:\n"
                               "- For 'STRAY', 'CAPA', 'AutoregressionAD', and 'PcaAD': use an empty dictionary.\n"
                               "- For 'ThresholdAD': must contain two keys: 'low' (float or int) for the lower threshold and 'high' (float or int) for the upper threshold."
            },
            "metric_list": {
                "type": "list",
                "description": "A list of metric names (strings) corresponding to columns in the input time series data on which anomaly detection should be performed."
            }
        },
        "required": ["detector_name", "detector_config", "metric_list"]
    },
    "metadata": {}
}

interpolate_tool_card = {
    "description": "Interpolates missing values in time series channels using a specified method. Returns the number of interpolated points, their indices, and the filled values for each channel.",
    "parameters": {
        "properties": {
            "method": {
                "type": "string",
                "description": "Interpolation method to use. Supported values: 'linear' (default), 'spline', 'ffill' (forward fill), 'bfill' (backward fill), 'mean' (fill with channel mean)."
            },
            "channel": {
                "type": "string",
                "description": "Optional. Name of a specific channel to interpolate. If omitted, all channels are processed."
            },
            "order": {
                "type": "integer",
                "description": "Spline order when method='spline'. Default is 3. Ignored for other methods."
            }
        },
        "required": []
    },
    "metadata": {}
}

differencing_tool_card = {
    "description": "Computes the n-th order difference of time series channels, useful for removing trends or achieving stationarity. Returns the differenced values, original and resulting lengths for each channel.",
    "parameters": {
        "properties": {
            "order": {
                "type": "integer",
                "description": "Order of differencing (number of times to apply the difference operator). Default is 1. Must be at least 1 and less than the series length."
            },
            "channel": {
                "type": "string",
                "description": "Optional. Name of a specific channel to difference. If omitted, all channels are processed."
            },
            "source": {
                "type": "string",
                "description": "Optional. Key of a derived series to use as input instead of the original timeseries. Available derived series keys are shown in the conversation history."
            },
            "register_as": {
                "type": "string",
                "description": "Optional. If provided, the output series will be registered as a derived series with this key, which can be used as 'source' in subsequent tool calls."
            },
        },
        "required": []
    },
    "metadata": {}
}

series_info_tool_card = {
    "description": "Retrieves basic metadata for each time series channel, including its length and the indices of missing values. Returns a summary containing the number of channels and detailed information per channel.",
    "parameters": {
        "properties": {},
        "required": []
    },
    "metadata": {}
}

datapoint_value_tool_card = {
    "description": "Retrieves the values of all time series channels at a specified time index. Returns a dictionary mapping each channel name to its value at that index.",
    "parameters": {
        "properties": {
            "index": {
                "type": "integer",
                "description": "The time index (zero-based) at which to extract the data point values from the time series."
            }
        },
        "required": ["index"]
    },
    "metadata": {}
}

summary_stats_tool_card = {
    "description": "Computes summary statistics for each time series channel over a specified index range. Returns a dictionary of statistical metrics per channel.",
    "parameters": {
        "properties": {
            "start": {
                "type": "integer",
                "description": "The starting index (inclusive) of the time series segment to analyze."
            },
            "end": {
                "type": "integer",
                "description": "The ending index (exclusive) of the time series segment to analyze."
            },
            "stat": {
                "type": "string",
                "description": "Specifies which statistic to compute. Options include 'mean', 'sum', 'min', 'max', 'std', or 'default' (which computes all of them). If omitted, defaults to 'default'."
            },
            "source": {
                "type": "string",
                "description": "Optional. Key of a derived series to use as input instead of the original timeseries. Available derived series keys are shown in the conversation history."
            },
        },
        "required": ["start", "end"]
    },
    "metadata": {}
}

return_calc_tool_card = {
    "description": "Calculates the return between two scalar values, either as an absolute difference or a percentage change. Returns a single numeric result.",
    "parameters": {
        "properties": {
            "t1": {
                "type": "number",
                "description": "The initial value (baseline) for return calculation."
            },
            "t2": {
                "type": "number",
                "description": "The final value used to compute the return relative to t1."
            },
            "kind": {
                "type": "string",
                "description": "The type of return to compute: 'diff' for absolute difference (t2 - t1), or 'pct' for percentage change ((t2 - t1) / t1)."
            }
        },
        "required": ["t1", "t2", "kind"]
    },
    "metadata": {}
}

autocorr_tool_card = {
    "description": "Computes the autocorrelation of each time series channel at a specified lag. Returns a dictionary mapping each channel name to its autocorrelation value.",
    "parameters": {
        "properties": {
            "lag": {
                "type": "integer",
                "description": "The lag at which to compute autocorrelation. Must be a positive integer."
            },
            "source": {
                "type": "string",
                "description": "Optional. Key of a derived series to use as input instead of the original timeseries. Available derived series keys are shown in the conversation history."
            },
            "register_as": {
                "type": "string",
                "description": "Optional. If provided, the output series will be registered as a derived series with this key, which can be used as 'source' in subsequent tool calls."
            },
        },
        "required": ["lag"]
    },
    "metadata": {}
}

rolling_stat_tool_card = {
    "description": "Computes rolling statistics (e.g., mean, sum, min, max, std) over a time series using a sliding window with configurable size and step. Returns the computed statistic values for each channel.",
    "parameters": {
        "properties": {
            "stat": {
                "type": "string",
                "description": "The type of rolling statistic to compute. Supported values: 'mean', 'sum', 'min', 'max', 'std'."
            },
            "window": {
                "type": "integer",
                "description": "The size of the rolling window (number of time steps). Must be a positive integer not exceeding the series length."
            },
            "step": {
                "type": "integer",
                "description": "The step size (stride) between consecutive windows. Must be a positive integer."
            },
            "source": {
                "type": "string",
                "description": "Optional. Key of a derived series to use as input instead of the original timeseries. Available derived series keys are shown in the conversation history."
            },
            "register_as": {
                "type": "string",
                "description": "Optional. If provided, the output series will be registered as a derived series with this key, which can be used as 'source' in subsequent tool calls."
            },
        },
        "required": ["stat", "window", "step"]
    },
    "metadata": {}
}

quantile_value_tool_card = {
    "description": "Computes quantile values for each time series channel across the time dimension. Returns a dictionary mapping each channel name to its quantile value(s).",
    "parameters": {
        "properties": {
            "q": {
                "type": "number",
                "description": "The quantile or list of quantiles to compute, between 0 and 1 (e.g., 0.5 for median, [0.25, 0.75] for quartiles)."
            },
            "source": {
                "type": "string",
                "description": "Optional. Key of a derived series to use as input instead of the original timeseries. Available derived series keys are shown in the conversation history."
            },
        },
        "required": ["q"]
    },
    "metadata": {}
}

volatility_tool_card = {
    "description": "Computes rolling volatility (standard deviation of first differences) for each time series channel using a sliding window. Returns a dictionary mapping each channel name to its sequence of volatility values.",
    "parameters": {
        "properties": {
            "window": {
                "type": "integer",
                "description": "The size of the rolling window used to compute volatility. Must be at least 2 and no larger than the length of the time series."
            },
            "source": {
                "type": "string",
                "description": "Optional. Key of a derived series to use as input instead of the original timeseries. Available derived series keys are shown in the conversation history."
            },
            "register_as": {
                "type": "string",
                "description": "Optional. If provided, the output series will be registered as a derived series with this key, which can be used as 'source' in subsequent tool calls."
            },
        },
        "required": ["window"]
    },
    "metadata": {}
}

trend_classifier_tool_card = {
    "description": "Classifies the trend (e.g., increasing, decreasing, or no trend) of each time series channel, either globally or using a rolling window. Returns a dictionary mapping each channel to its trend classification(s).",
    "parameters": {
        "properties": {
            "window": {
                "type": "integer",
                "description": "Optional. If provided, trend classification is performed in a rolling manner with this window size. If omitted or None, a global trend is computed for the entire series."
            },
            "p_threshold": {
                "type": "number",
                "description": "The p-value threshold for statistical significance in trend detection (default: 0.05)."
            },
            "slope_threshold": {
                "type": "number",
                "description": "The minimum absolute slope required to consider a trend as non-flat. If not provided, it defaults to 1% of the range (max - min) of the time series."
            },
            "source": {
                "type": "string",
                "description": "Optional. Key of a derived series to use as input instead of the original timeseries. Available derived series keys are shown in the conversation history."
            },
        },
        "required": []
    },
    "metadata": {}
}

seasonality_detector_tool_card = {
    "description": "Detects the presence and strength of seasonality in each time series channel by analyzing autocorrelation of detrended data. Returns a dictionary mapping each channel to its detected period (if any) and seasonality strength ('strong', 'weak', or 'none').",
    "parameters": {
        "properties": {
            "max_period": {
                "type": "integer",
                "description": "Maximum period (in time steps) to consider when searching for seasonality. Must be at least 2. If not provided, defaults to 100."
            },
            "strong_threshold": {
                "type": "number",
                "description": "Autocorrelation threshold above which seasonality is considered 'strong'. Default is 0.6."
            },
            "weak_threshold": {
                "type": "number",
                "description": "Autocorrelation threshold above which seasonality is considered 'weak' (but not strong). Default is 0.2."
            },
            "source": {
                "type": "string",
                "description": "Optional. Key of a derived series to use as input instead of the original timeseries. Available derived series keys are shown in the conversation history."
            },
        },
        "required": []
    },
    "metadata": {}
}

change_point_detector_tool_card = {
    "description": "Detects change points in each time series channel using segmentation algorithms. Returns a dictionary mapping each channel to a list of detected change point indices.",
    "parameters": {
        "properties": {
            "penalty": {
                "type": "number",
                "description": "Penalty value used in the Pelt algorithm to control the number of change points. If not provided, a default penalty based on variance and series length is used."
            },
            "n_cp": {
                "type": "integer",
                "description": "Number of change points to detect when using the Binseg strategy. Required if strategy is 'Binseg'."
            },
            "cost_func": {
                "type": "string",
                "description": "Cost function used for segmentation. Common options include 'l2' (default), 'l1', or 'rbf'."
            },
            "strategy": {
                "type": "string",
                "description": "Change point detection algorithm to use. Supported values: 'Pelt' (default) or 'Binseg'."
            },
            "source": {
                "type": "string",
                "description": "Optional. Key of a derived series to use as input instead of the original timeseries. Available derived series keys are shown in the conversation history."
            },
        },
        "required": []
    },
    "metadata": {}
}

noise_profile_tool_card = {
    "description": "Analyzes the noise characteristics of each time series channel, classifying it as white noise, colored noise, or non-noise either globally or in a rolling window. Returns a dictionary mapping each channel to its noise classification(s).",
    "parameters": {
        "properties": {
            "window": {
                "type": "integer",
                "description": "Optional. If provided, noise classification is performed in a rolling manner with this window size. If omitted or None, a global classification is computed for the entire series."
            },
            "max_lag": {
                "type": "integer",
                "description": "Maximum lag up to which autocorrelations are examined for noise testing. Default is 20."
            },
            "alpha": {
                "type": "number",
                "description": "Significance level for statistical tests (e.g., Ljung-Box) used in noise classification. Default is 0.05."
            },
            "source": {
                "type": "string",
                "description": "Optional. Key of a derived series to use as input instead of the original timeseries. Available derived series keys are shown in the conversation history."
            },
        },
        "required": []
    },
    "metadata": {}
}

stationarity_test_tool_card = {
    "description": "Performs a stationarity test on each time series channel using either the Augmented Dickey-Fuller (ADF) or KPSS test. Returns a dictionary with test results, including stationarity status, p-value, test statistic, and critical values for each channel.",
    "parameters": {
        "properties": {
            "test": {
                "type": "string",
                "description": "The stationarity test to perform. Supported values: 'adf' (Augmented Dickey-Fuller) or 'kpss'. Default is 'adf'."
            },
            "alpha": {
                "type": "number",
                "description": "Significance level for determining stationarity. Default is 0.05."
            },
            "source": {
                "type": "string",
                "description": "Optional. Key of a derived series to use as input instead of the original timeseries. Available derived series keys are shown in the conversation history."
            },
        },
        "required": []
    },
    "metadata": {}
}

spike_detector_tool_card = {
    "description": "Detects spikes (positive outliers) and dips (negative outliers) in each time series channel based on deviation from the mean. Returns a dictionary mapping each channel to lists of detected spike and dip indices.",
    "parameters": {
        "properties": {
            "threshold": {
                "type": "number",
                "description": "Threshold for detecting spikes/dips. If 'relative' is True, this is a multiplier of the standard deviation; otherwise, it is an absolute value. Default is 2."
            },
            "min_sep": {
                "type": "integer",
                "description": "Minimum separation (in time steps) between consecutive spikes or dips. Default is 5."
            },
            "relative": {
                "type": "boolean",
                "description": "If true, threshold is interpreted relative to the standard deviation of the series. If false, threshold is an absolute value. Default is true. Make sure use lower word true/false in json string response."
            },
            "source": {
                "type": "string",
                "description": "Optional. Key of a derived series to use as input instead of the original timeseries. Available derived series keys are shown in the conversation history."
            },
        },
        "required": []
    },
    "metadata": {}
}

channel_correlation_tool_card = {
    "description": "Computes the correlation (with optional lag) between two specified time series channels. Returns the correlation coefficient, p-value, and number of observations used.",
    "parameters": {
        "properties": {
            "channel_1": {
                "type": "string",
                "description": "Name of the first time series channel to correlate."
            },
            "channel_2": {
                "type": "string",
                "description": "Name of the second time series channel to correlate."
            },
            "lag": {
                "type": "integer",
                "description": "Lag applied to the second series (channel_2) before computing correlation. A positive lag means channel_2 is shifted forward in time. Default is 0."
            },
            "method": {
                "type": "string",
                "description": "Correlation method to use. Supported values: 'pearson' (default), 'spearman', or 'kendall'."
            }
        },
        "required": ["channel_1", "channel_2"]
    },
    "metadata": {}
}

cross_correlation_tool_card = {
    "description": "Computes the full cross-correlation function between two time series channels across a range of lags. Returns the lag with maximum correlation, the corresponding correlation value, and the full set of lags and correlation coefficients.",
    "parameters": {
        "properties": {
            "channel_1": {
                "type": "string",
                "description": "Name of the first time series channel."
            },
            "channel_2": {
                "type": "string",
                "description": "Name of the second time series channel."
            },
            "max_lag": {
                "type": "integer",
                "description": "Maximum absolute lag to consider (must be non-negative). The analysis will cover lags from -max_lag to +max_lag. Default is 50."
            }
        },
        "required": ["channel_1", "channel_2"]
    },
    "metadata": {}
}

dtw_distance_tool_card = {
    "description": "Computes the Dynamic Time Warping (DTW) distance between two time series channels, measuring their similarity under non-linear alignment. Returns the DTW distance as a numeric value.",
    "parameters": {
        "properties": {
            "channel_1": {
                "type": "string",
                "description": "Name of the first time series channel."
            },
            "channel_2": {
                "type": "string",
                "description": "Name of the second time series channel."
            },
            "distance_metric": {
                "type": "string",
                "description": "Distance metric used in DTW computation. Supported values: 'euclidean' (default) or 'sqeuclidean' (squared Euclidean)."
            }
        },
        "required": ["channel_1", "channel_2"]
    },
    "metadata": {}
}

shape_similarity_tool_card = {
    "description": "Measures the shape similarity between two time series channels using Pearson correlation (zero lag). Returns the correlation coefficient, p-value, and number of observations used.",
    "parameters": {
        "properties": {
            "channel_1": {
                "type": "string",
                "description": "Name of the first time series channel."
            },
            "channel_2": {
                "type": "string",
                "description": "Name of the second time series channel."
            }
        },
        "required": ["channel_1", "channel_2"]
    },
    "metadata": {}
}

granger_causality_tool_card = {
    "description": "Performs Granger causality test to determine whether one time series channel (cause) helps predict another (effect). Returns the test result including p-value, F-statistic, and whether the null hypothesis of no Granger causality is rejected.",
    "parameters": {
        "properties": {
            "cause_channel": {
                "type": "string",
                "description": "Name of the candidate causal (explanatory) time series channel."
            },
            "effect_channel": {
                "type": "string",
                "description": "Name of the affected (response) time series channel."
            },
            "max_lag": {
                "type": "integer",
                "description": "Maximum number of lags to include in the Granger causality test. Must be at least 1. Default is 20."
            },
            "alpha": {
                "type": "number",
                "description": "Significance level for hypothesis testing. Default is 0.05."
            }
        },
        "required": ["cause_channel", "effect_channel"]
    },
    "metadata": {}
}

TOOLBOX = {
    "forecasting_tool": forecasting_tool,
    "anomaly_detection_tool": anomaly_detection_tool,
    "series_info": series_info,
    "datapoint_value": datapoint_value,
    "summary_stats": summary_stats,
    "return_calc": return_calc,
    "autocorr": autocorr,
    "rolling_stat": rolling_stat,
    "quantile_value": quantile_value,
    "volatility": volatility,
    "interpolate": interpolate,
    "differencing": differencing,
    "trend_classifier": trend_classifier,
    "seasonality_detector": seasonality_detector,
    "change_point_detector": change_point_detector,
    "noise_profile": noise_profile,
    "stationarity_test": stationarity_test,
    "spike_detector": spike_detector,
    "channel_correlation": channel_correlation,
    "cross_correlation": cross_correlation,
    "dtw_distance": dtw_distance,
    "shape_similarity": shape_similarity,
    "granger_causality": granger_causality,
    #"anomaly_detection": anomaly_detection,
    #"forecaster": forecaster,
}

TOOLCARD = {
    "forecasting_tool": forecasting_tool_card,
    "anomaly_detection_tool": anomaly_detection_tool_card,
    "series_info": series_info_tool_card,
    "datapoint_value": datapoint_value_tool_card,
    "summary_stats": summary_stats_tool_card,
    "return_calc": return_calc_tool_card,
    "autocorr": autocorr_tool_card,
    "rolling_stat": rolling_stat_tool_card,
    "quantile_value": quantile_value_tool_card,
    "volatility": volatility_tool_card,
    "interpolate": interpolate_tool_card,
    "differencing": differencing_tool_card,
    "trend_classifier": trend_classifier_tool_card,
    "seasonality_detector": seasonality_detector_tool_card,
    "change_point_detector": change_point_detector_tool_card,
    "noise_profile": noise_profile_tool_card,
    "stationarity_test": stationarity_test_tool_card,
    "spike_detector": spike_detector_tool_card,
    "channel_correlation": channel_correlation_tool_card,
    "cross_correlation": cross_correlation_tool_card,
    "dtw_distance": dtw_distance_tool_card,
    "shape_similarity": shape_similarity_tool_card,
    "granger_causality": granger_causality_tool_card,
    #"anomaly_detection": anomaly_detection,
    #"forecaster": forecaster,
}

def execute_tool(state, tool_call_json):
    if "name" not in tool_call_json: return {"error": "Missing 'name' field"}
    if "arguments" not in tool_call_json: return {"error": "Missing 'arguments' field"}
    
    tool_name, arguments = tool_call_json["name"], tool_call_json["arguments"]
    if tool_name not in TOOLBOX:
        return {"error": f"Tool '{tool_name}' is not registered or does not exist."}
    try:
        tool_func = TOOLBOX[tool_name]
        return_value = tool_func(state, arguments)
        if isinstance(return_value, str):
            return {"result": return_value}
        else:
            if 'register_as' in arguments:
                return {"need_save": (arguments['register_as'],return_value[0]) , "result": return_value[1]}
            return {"result": return_value[1]}
    except TypeError as e:
        return {"error": f"Argument mismatch when calling '{tool_name}': {str(e)}"}
    except Exception as e:
        return {"error": f"Execution failed in '{tool_name}': {str(e)}"}