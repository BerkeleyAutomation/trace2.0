import sys
sys.path.insert(0, '..')

import numpy as np
import math
import matplotlib.pyplot as plt
import os
from collections import defaultdict

GROUP_SEPARATION = 2
DISTANCE_INCREMENT = 10
DISTANCE_THRESHOLD = 450 
DISPLAY_MODE = True
LENGTH_MISMATCH_PENALTY = 0.5
TRACES_DIR = '../data/traces/'

""" Calculation Methods """

def interpolate(p0, p1, d):
    """ Gets the point that is the fraction 0 <= d <= 1 of the way between p0 and p1 """
    return ((1-d)*p0[0] + d*p1[0], (1-d)*p0[1] + d*p1[1])


def get_l2_squared(p0, p1):
    """ Gets the squared Euclidean distance between p0 and p1 """
    return (p0[0] - p1[0])**2 + (p0[1] - p1[1])**2


def get_cumulative_cartesian_dist(trace):
    """ 
    Gets the cumulative distance between successive points.
    For example, if the points are 10, 5 and 7 mm apart, the array looks like
    [0, 10, 15, 22]
    """
    cumulative_dist = [0] # the nth entry here tells you how far it is to the nth point
    for i in range(len(trace) - 1):
        cumulative_dist.append(cumulative_dist[-1]+ math.sqrt((trace[i][0] - trace[i+1][0])**2 + (trace[i][1] - trace[i+1][1])**2))
    return cumulative_dist


def get_full_path_dist(trace):
    """ Gets the total length of the rope (roughly) """
    cumulative_dist = 0 
    for i in range(len(trace) - 1):
        cumulative_dist += math.sqrt((trace[i][0] - trace[i+1][0])**2 + (trace[i][1] - trace[i+1][1])**2)
    return cumulative_dist


def get_inter_pts(trace, increment):
    """
    Given a set of trace points and an increment (how far apart our points should be), interpolate
    as many points as you can at distance increment from each other, along the rope
    """
    inter_pts = []
    cumulative_dists = get_cumulative_cartesian_dist(trace)

    # if the length is 96.4, and the interval is 10, we can fit
    # floor(96.4/10) + 1 = 10 points spaced at 10 apart
    num_inter_pts = math.floor(cumulative_dists[-1] / increment) + 1

    # step through intervals between trace points, placing interpolated points where necessary
    current_trace_interval = 0

    for i in range(num_inter_pts):
        parameter_value = i * increment

        # travel the requisite amount along the path
        while parameter_value >= cumulative_dists[current_trace_interval]:
            current_trace_interval += 1

        current_trace_interval -= 1 # trust the process, we need to get the start of the bin

        dist_in_interval = cumulative_dists[current_trace_interval + 1] - cumulative_dists[current_trace_interval]
        scaled_proportion = (parameter_value - cumulative_dists[current_trace_interval]) / dist_in_interval

        new_pt = interpolate(trace[current_trace_interval], trace[current_trace_interval + 1], scaled_proportion)
        inter_pts.append(new_pt)

    return inter_pts


def get_inter_points_for_traces(trace_t, noisy_traces):
    """ Gets interpolated points for (original, noisy) traces """
    
    # get interpolated points
    inter_pts_t = get_inter_pts(trace_t, DISTANCE_INCREMENT)
    inter_pts_noisy = [get_inter_pts(trace_n, DISTANCE_INCREMENT) for trace_n in noisy_traces]
    return inter_pts_t, inter_pts_noisy


def get_penalties(inter_pts0, inter_pts1, trace0, trace1):
    """ Gets the variance, length, and total (variance + length) penalties """
    
    # take the variance of them
    total_var_penalty = 0
    for i in range(min(len(inter_pts0), len(inter_pts1))):
        total_var_penalty += get_l2_squared(inter_pts0[i], inter_pts1[i])
    
    # now, we have the penalty for one being longer than the other (which does indicate uncertainty)
    total_len_penalty = abs(get_full_path_dist(trace0) - get_full_path_dist(trace1))
    total_penalty = total_var_penalty + total_len_penalty
    return total_var_penalty, total_len_penalty, total_penalty


def group_divergences(divergences_by_idx, group_separation=GROUP_SEPARATION):
    """ 
    Clusters close divergences (maximally separated by group_separation) with one another 
    """
    grouped_divergences_by_idx = defaultdict(list)
    sorted_idx = sorted(divergences_by_idx.keys())
    for group in np.split(sorted_idx, np.where(np.diff(sorted_idx) > group_separation)[0] + 1):
        for group_idx in group:
            grouped_divergences_by_idx[group[0]].extend(divergences_by_idx[group_idx])
    return grouped_divergences_by_idx

""" Loader Methods """

def load_traces_for_trace_id(trace_id):
    """ Loads original, noisy traces for a given trace ID """

    # get the original (i.e. without noise) trace
    if not os.path.isfile(TRACES_DIR + f'{trace_id}/trace_0.npy'):
        raise Exception('Original trace file does not exist!')
    trace_t = np.load(TRACES_DIR + f'{trace_id}/trace_0.npy', allow_pickle=True)

    # get all perturbed traces
    ctr = 1
    noisy_traces = []
    while os.path.isfile(TRACES_DIR + f'{trace_id}/trace_{ctr}.npy'):
        noisy_traces.append(np.load(TRACES_DIR + f'{trace_id}/trace_{ctr}.npy', allow_pickle=True))
        ctr += 1

    return trace_t, noisy_traces

""" Methods For Importing """

def get_divergence_pts(trace_t, noisy_traces, group=True, prune=True):
    """ Gets all divergences for a given trace id """

    # get interpolated points
    inter_pts_t, inter_pts_noisy = get_inter_points_for_traces(trace_t, noisy_traces)

    # get (interpolated) point-wise var_penalties of true trace compared to all noise-added traces
    divergences_by_idx = defaultdict(list)
    for ctr, inter_pts_n in enumerate(inter_pts_noisy):
        for i in range(min(len(inter_pts_t), len(inter_pts_n))):
            if get_l2_squared(inter_pts_t[i], inter_pts_n[i]) > DISTANCE_THRESHOLD:
                divergences_by_idx[i].append(ctr + 1)
                break

    if group:
        divergences_by_idx = group_divergences(divergences_by_idx)

    if prune:
        for idx in list(divergences_by_idx.keys())[:]:
            if len(divergences_by_idx[idx]) == 1:
                del divergences_by_idx[idx]

    num_divergence_pts = sum([len(divergences_by_idx[idx]) for idx in divergences_by_idx])

    divergence_pts = []
    for idx in divergences_by_idx:
        x, y = inter_pts_t[idx][0], inter_pts_t[idx][1] 
        divergence_pts.append([x,y])
    return divergence_pts, num_divergence_pts
    

def get_first_divergence_penalty(trace_t, noisy_traces):
    """ Gets a sum of all lengths of diverged paths after a divergence > DISTANCE_THRESHOLD """

    # get interpolated points
    inter_pts_t, inter_pts_noisy = get_inter_points_for_traces(trace_t, noisy_traces)

    # get (interpolated) point-wise var_penalties of true trace compared to all noise-added traces
    total_penalty = 0
    
    divergences_by_idx = defaultdict(list)
    for ctr, inter_pts_n in enumerate(inter_pts_noisy):
        for i in range(min(len(inter_pts_t), len(inter_pts_n))):
            if get_l2_squared(inter_pts_t[i], inter_pts_n[i]) > DISTANCE_THRESHOLD:
                divergences_by_idx[i].append(ctr + 1)
                total_penalty += (len(inter_pts_t) - i) 
                break

    # control for us changing this increment - we're now dealing in pixel distance from divergence
    total_penalty *= DISTANCE_INCREMENT

    return total_penalty


def get_diverged_count_penalty(trace_t, noisy_traces):
    """ 
    Gets sum of how many times each trace differs from the original for a trace ID
    N.B. to avoid very short traces having a good score, we count the missing 
    points from a noisy trace as being divergent
    """

     # get interpolated points
    inter_pts_t, inter_pts_noisy = get_inter_points_for_traces(trace_t, noisy_traces)

    # get (interpolated) point-wise var_penalties and sum up diverged pts
    total_diverged_points = 0
    divergences_by_trace = [None for _ in range(len(inter_pts_noisy))]
    for ctr, inter_pts_n in enumerate(inter_pts_noisy):
        for i in range(min(len(inter_pts_t), len(inter_pts_n))):
            if get_l2_squared(inter_pts_t[i], inter_pts_n[i]) > DISTANCE_THRESHOLD:
                # note first divergence in divergences_by_trace
                if divergences_by_trace[ctr] is None:
                    divergences_by_trace[ctr] = i
                # increment counter
                total_diverged_points += 1
            total_diverged_points += max(0, len(inter_pts_t) - len(inter_pts_n))
    total_diverged_points /= len(inter_pts_noisy)

    return total_diverged_points

""" Methods For Testing """

def _get_divergence_pts_for_trace_id(trace_id, viz=True, group=True):
    """ Gets all divergences for a given trace ID """
    
    # get interpolated points
    trace_t, noisy_traces = load_traces_for_trace_id(trace_id)
    inter_pts_t, inter_pts_noisy = get_inter_points_for_traces(trace_t, noisy_traces)

    # get (interpolated) point-wise var_penalties of true trace compared to all noise-added traces
    divergences_by_idx = defaultdict(list)
    for ctr, inter_pts_n in enumerate(inter_pts_noisy):
        for i in range(min(len(inter_pts_t), len(inter_pts_n))):
            if get_l2_squared(inter_pts_t[i], inter_pts_n[i]) > DISTANCE_THRESHOLD:
                divergences_by_idx[i].append(ctr + 1)
                break

    if group:
        divergences_by_idx = group_divergences(divergences_by_idx)

    # print(f'Trace {trace_id}:', divergences_by_idx)
    if viz:
        if not os.path.isfile(TRACES_DIR + f'{trace_id}/trace_0.png'):
            raise Exception('Original trace image does not exist!')
        plt.imshow(plt.imread(TRACES_DIR + f'{trace_id}/trace_0.png'))
        scatter_x, scatter_y = [], []
        for idx in divergences_by_idx:
            # note: x, y reversed
            x, y = inter_pts_t[idx][1], inter_pts_t[idx][0] 
            scatter_x.append(x)
            scatter_y.append(y)
            plt.text(x, y, str(len(divergences_by_idx[idx])))
        plt.scatter(scatter_x, scatter_y)
        plt.savefig(TRACES_DIR + f'{trace_id}/divergences.png')
        plt.clf()

    divergence_pts = []
    for idx in divergences_by_idx:
        x, y = inter_pts_t[idx][0], inter_pts_t[idx][1] 
        divergence_pts.append([x,y])
    return divergence_pts


def _get_first_divergence_penalty_for_trace_id(trace_id, viz=True):
    """ Gets a sum of all lengths of diverged paths after a divergence > DISTANCE_THRESHOLD for a given trace ID """

    # get interpolated points
    trace_t, noisy_traces = load_traces_for_trace_id(trace_id)
    inter_pts_t, inter_pts_noisy = get_inter_points_for_traces(trace_t, noisy_traces)

    # get (interpolated) point-wise var_penalties of true trace compared to all noise-added traces
    total_penalty = 0
    
    divergences_by_idx = defaultdict(list)
    for ctr, inter_pts_n in enumerate(inter_pts_noisy):
        for i in range(min(len(inter_pts_t), len(inter_pts_n))):
            if get_l2_squared(inter_pts_t[i], inter_pts_n[i]) > DISTANCE_THRESHOLD:
                divergences_by_idx[i].append(ctr + 1)
                total_penalty += (len(inter_pts_t) - i) 
                break

    # control for us changing this increment - we're now dealing in pixel distance from divergence
    total_penalty *= DISTANCE_INCREMENT

    if viz:
        if not os.path.isfile(TRACES_DIR + f'{trace_id}/trace_0.png'):
            raise Exception('Original trace image does not exist!')
        plt.imshow(plt.imread(TRACES_DIR + f'{trace_id}/trace_0.png'))
        scatter_x, scatter_y = [], []
        for idx in divergences_by_idx:
            # note: x, y reversed
            x, y = inter_pts_t[idx][1], inter_pts_t[idx][0] 
            scatter_x.append(x)
            scatter_y.append(y)
        plt.scatter(scatter_x, scatter_y)
        plt.title(f'Total First Divergence Penalty: {total_penalty}')
        plt.savefig(TRACES_DIR + f'{trace_id}/global_first_penalty.png')
        plt.clf()

    return total_penalty


def _get_diverged_count_penalty_for_trace_id(trace_id, viz=True):
    """ 
    Gets sum of how many times each trace differs from the original for a given trace ID
    N.B. to avoid very short traces having a good score, we count the missing 
    points from a noisy trace as being divergent
    """

     # get interpolated points
    trace_t, noisy_traces = load_traces_for_trace_id(trace_id)
    inter_pts_t, inter_pts_noisy = get_inter_points_for_traces(trace_t, noisy_traces)

    # get (interpolated) point-wise var_penalties and sum up diverged pts
    total_diverged_points = 0
    divergences_by_trace = [None for _ in range(len(inter_pts_noisy))]
    for ctr, inter_pts_n in enumerate(inter_pts_noisy):
        for i in range(min(len(inter_pts_t), len(inter_pts_n))):
            if get_l2_squared(inter_pts_t[i], inter_pts_n[i]) > DISTANCE_THRESHOLD:
                # note first divergence in divergences_by_trace
                if divergences_by_trace[ctr] is None:
                    divergences_by_trace[ctr] = i
                # increment counter
                total_diverged_points += 1
            total_diverged_points += max(0, len(inter_pts_t) - len(inter_pts_n))
    total_diverged_points /= len(inter_pts_noisy)

    if viz:
        if not os.path.isfile(TRACES_DIR + f'{trace_id}/trace_0.png'):
            raise Exception('Original trace image does not exist!')
        plt.imshow(plt.imread(TRACES_DIR + f'{trace_id}/trace_0.png'))
        scatter_x, scatter_y = [], []
        for div_idx in divergences_by_trace:
            # note: x, y reversed
            if div_idx is not None:
                x, y = inter_pts_t[div_idx][1], inter_pts_t[div_idx][0] 
                scatter_x.append(x)
                scatter_y.append(y)
        plt.scatter(scatter_x, scatter_y)
        plt.title(f'Total Diverged Count Penalty: {total_diverged_points}')
        plt.savefig(TRACES_DIR + f'{trace_id}/global_count_penalty.png')
        plt.clf()
        
    return total_diverged_points


if __name__ == "__main__":
    if not os.path.exists(TRACES_DIR):
        raise Exception('Traces directory does not exist!')
    for trace_id in os.listdir(TRACES_DIR):
        if os.path.isdir(TRACES_DIR + f'/{trace_id}'):
            _get_divergence_pts_for_trace_id(trace_id)
            _get_first_divergence_penalty_for_trace_id(trace_id)
            _get_diverged_count_penalty_for_trace_id(trace_id)