import numpy as np
import pandas as pd
import visuals

def do(ds):
    # visuals.bivariate_scatter_plot(ds, ['sbytes', 'sttl', 'sloss', 'sload', 'spkts', 'swin', 'stcpb'])
    # visuals.bivariate_scatter_plot(ds, ['sbytes', 'sttl', 'sloss', 'spkts'])
    # visuals.bivariate_scatter_plot(ds, ['ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm'])
    visuals.histogram(ds, ['res_bdy_len'])