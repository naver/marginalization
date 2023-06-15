# Marginalization
# Copyright 2023-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import numpy as np
import pandas as pd
from scipy.special import logsumexp
import string
import sys

standard_columns = ["token", "ll"]
sampled_columns = ["word", "categ", "num_toks", "tokens", "is_st", "q", "ll", "ll-log_q", "num_options"]

def parse_log(log_file):
    """
    A helper function to parse logs printed by functions in tokenizations.py
    and output them in a form of pandas DataFrames.
    
    Input:
    * log_file: a filename for a log file
    
    Output:
    * a list of lists of pd.DataFrames, 
      the external list corresponds to strings from the dataset,
      the inner list corresponds to different tokenizations.
      The first dataframe in the inner list corresponds to a default tokenization
      and all next dataframes correspond to sampled tokenizations.
    """
    with open(log_file) as fin:
        all_tabs = []
        tab = None
        for line in fin:
            if line.strip() == "" or "***Command***" in line:
                continue
            if "*****Default tokenization*****" in line:
                processing_sampled = False
                if tab is not None:
                    tab = pd.DataFrame(tab, 
                                   columns=sampled_columns)
                    seq_tabs.append(tab)
                    all_tabs.append(seq_tabs)
                processing_standard = True
                tab = []
                continue
            if "*****Sampled tokenization*****" in line:
                if processing_standard:
                    tab = pd.DataFrame(tab, columns=standard_columns)
                    seq_tabs = [tab]
                    processing_standard = False
                    processing_sampled = True
                else: # processing_sampled
                    tab = pd.DataFrame(tab, 
                                   columns=sampled_columns)
                    seq_tabs.append(tab)
                tab = []
                continue
            if processing_standard:
                vals = line.strip().split(";")
                vals[1] = np.float64(vals[1])
                tab.append(vals)
            if processing_sampled:
                vals = line.strip().split(";")[:9]
                for i in [5, 6, 7]:
                    vals[i] = np.float64(vals[i])
                tab.append(vals)
    tab = pd.DataFrame(tab, 
                      columns=sampled_columns)
    seq_tabs.append(tab)
    if len(seq_tabs) == len(all_tabs[-1]): all_tabs.append(seq_tabs)
    # skip last sequence if its importance samplig did not finish yet
    
    return all_tabs


ln2 = np.log(2)
punctset = set(string.punctuation).union({"\\n"})


def is_punctonly(s):
        return all([c in punctset for c in s])


def is_dummy_split(s):
        tokens = s.split(" ")
        mask = [is_punctonly(token) for token in tokens]
        return len(tokens) > 1 and sum(mask) >= len(tokens) - 1


def gather_stats(all_tabs):
    """
    Prints final results from all_tabs parsed by parse_log
    """
    stats = []
    for seq_tabs in all_tabs:
        tab_st = seq_tabs[0]
        tabs_sampl = seq_tabs[1:]
        nchars = tab_st["token"].apply(len).sum()
        ll_st = tab_st["ll"].sum()
        lls_sampl = []
        lls_logq = []
        st_num0, st_num1, st_num2, all_num = 0, 0, 0, 0
        c0, c1, c2 = 0, 0, 0
        for tab in tabs_sampl:
            lls_logq.append(tab["ll-log_q"].sum())
            lls_sampl.append(tab["ll"].sum())
            is_dummy_ = tab["tokens"].apply(is_dummy_split)
            st_num0 += (tab["is_st"]!="cst").sum()
            st_num1 += (tab[~is_dummy_]["is_st"]!="cst").sum()
            st_num2 += (tab[is_dummy_]["is_st"]!="cst").sum()
            all_num += tab.shape[0]
            c0 += (tab["categ"]=="T0").sum()
            c1 += (tab["categ"]=="T1").sum()
            c2 += (tab["categ"]=="T2").sum()
        impsampl = logsumexp(lls_logq, b=np.ones(len(lls_logq))/len(lls_logq))
        stats.append([-ll_st/ln2, -impsampl/ln2, nchars, \
                     st_num0, st_num1, st_num2, all_num, c0, c1, c2])
    return np.array(stats)

def print_stats(stats, all_tabs):
    print("Num processed sequences:", stats.shape[0])
    print("Num samples per sequence:", len(all_tabs[0])-1)
    print("Portion of sequences with BPC_df worse than BPC_is:", (stats[:, 0] > stats[:, 1]).mean())
    bpcs = stats[:, :2].sum(axis=0)/stats[:, 2].sum() # [BPC_df, BPC_is]
    print("[BPC_df, BPC_is]: ", bpcs)
    print("BPC_gap:", bpcs[0]-bpcs[1])
    print("Relative BPC gap (%):", (bpcs[0]-bpcs[1]) / bpcs[0] * 100)
    print("Portion of chosen non-default tokenizations: ", \
           stats[:, 3].sum(axis=0)/stats[:, 6].sum())
    print("Portion of [T0, T1, T2] type words:")
    print(stats[:, 7:10].sum(axis=0)/stats[:, 6].sum())
    
    
# confidence intervals with bootstrap
import string
from scipy.stats import bootstrap
def gather_confints(all_tabs):
    """
    Computes confidence intervals on BPC_df-BPC_is using bootstrapping,
    by using logs parsed by parse_logs
    
    Returns:
    * res: np.array with shape [num_seqs, 2] where each row is a confidence interval
    On plots in the paper we plot: plt.scatter(res[:, 0], res[:, 1]-res[:, 0])
    """
    def has_punkt(s):
        return any([c in s.replace("_", "") for c in string.punctuation])
    
    res = []
    for seq_tabs in all_tabs:
        tab_st = seq_tabs[0]
        tabs_sampl = seq_tabs[1:]
        nchars = tab_st["token"].apply(len).sum()
        ll_st = tab_st["ll"].sum()
        lls_sampl = []
        lls_logq = []
        for tab in tabs_sampl:
            lls_logq.append(tab["ll-log_q"].sum())
            lls_sampl.append(tab["ll"].sum())
        
        def ISestimate(lls_logq):
            lgs = logsumexp(lls_logq, b=np.ones(len(lls_logq))/len(lls_logq))
            return (lgs-ll_st) / nchars
        bsres = bootstrap((lls_logq,), 
                        ISestimate, confidence_level=0.9,
                        vectorized=False, n_resamples=999)
        lgs = logsumexp(lls_logq, b=np.ones(len(lls_logq))/len(lls_logq))
        est = (lgs-ll_st) / nchars
        
        
        res.append([bsres.confidence_interval.low, 
                    bsres.confidence_interval.high, est])
    return np.array(res)


if __name__ == "__main__":
    log_filename = sys.argv[1]
    all_tabs = parse_log(log_filename)
    stats = gather_stats(all_tabs)
    print_stats(stats, all_tabs)