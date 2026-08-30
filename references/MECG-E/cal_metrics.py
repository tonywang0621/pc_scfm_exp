# -*- coding: utf-8 -*-

#============================================================
#
#  Deep Learning BLW Filtering
#  Main
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================

import _pickle as pickle
from datetime import datetime
import time
import numpy as np

from utils.metrics import MAD, SSD, PRD, COS_SIM
from utils import visualization as vs
import argparse
from os.path import isfile
import os
os.makedirs('score',exist_ok=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MECGE for ECG")
    parser.add_argument("--experiments", type=str, nargs='+', default=['MECGE_phase','MECGE_complex']) #
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--n_type', type=str, default='bw', help='noise type') # 'bw' or 'em' or 'ma' or 'all'
    args = parser.parse_args()

    Exp_names = args.experiments

    noise_versions = [1, 2]
    n_type = args.n_type

    SSD_all, MAD_all, PRD_all, COS_SIM_all = [], [], [], []

    for experiment in Exp_names:
    # Load Results SEMamba
        with open(f'results/test_results_{n_type}_' + experiment + '_nv1.pkl', 'rb') as input:
            test_SEMamba_nv1 = pickle.load(input)
        with open(f'results/test_results_{n_type}_' + experiment + '_nv2.pkl', 'rb') as input:
            test_SEMamba_nv2 = pickle.load(input)
    
        test_SEMamba = [np.concatenate((test_SEMamba_nv1[0], test_SEMamba_nv2[0])),
                     np.concatenate((test_SEMamba_nv1[1], test_SEMamba_nv2[1])),
                     np.concatenate((test_SEMamba_nv1[2], test_SEMamba_nv2[2]))]

        
        [X_test, y_test, y_pred] = test_SEMamba

        SSD_values_DL_SEMamba = SSD(y_test, y_pred)
        MAD_values_DL_SEMamba = MAD(y_test, y_pred)
        PRD_values_DL_SEMamba = PRD(y_test, y_pred)
        COS_SIM_values_DL_SEMamba = COS_SIM(y_test, y_pred)

        SSD_all.append(SSD_values_DL_SEMamba)
        MAD_all.append(MAD_values_DL_SEMamba)
        PRD_all.append(PRD_values_DL_SEMamba)
        COS_SIM_all.append(COS_SIM_values_DL_SEMamba)

    
    ####### Calculate Metrics #######

    print('Calculating metrics ...')
    
    metrics = ['SSD', 'MAD', 'PRD', 'COS_SIM']
    metric_values = [SSD_all, MAD_all, PRD_all, COS_SIM_all]

    # Metrics table
    experiments_dict = dict()
    for experiment in Exp_names:
        experiments_dict[experiment] = dict()
        for metric in metrics:
            experiments_dict[experiment][metric] = dict()
    experiments_dict = vs.generate_table(metrics, metric_values, Exp_names, experiments_dict)

    # Timing table
    # timing_var = ['training', 'test']
    # vs.generate_table_time(timing_var, timing, Exp_names, gpu=True)

    ################################################################################################################
    # Segmentation by noise amplitude
    rnd_test = np.load('rnd_test.npy')

    rnd_test = np.concatenate([rnd_test, rnd_test])

    segm = [0.2, 0.6, 1.0, 1.5, 2.0]  # real number of segmentations is len(segmentations) - 1
    SSD_seg_all = []
    MAD_seg_all = []
    PRD_seg_all = []
    COS_SIM_seg_all = []

    for idx_exp in range(len(Exp_names)):
        SSD_seg = [None] * (len(segm) - 1)
        MAD_seg = [None] * (len(segm) - 1)
        PRD_seg = [None] * (len(segm) - 1)
        COS_SIM_seg = [None] * (len(segm) - 1)
        for idx_seg in range(len(segm) - 1):
            SSD_seg[idx_seg] = []
            MAD_seg[idx_seg] = []
            PRD_seg[idx_seg] = []
            COS_SIM_seg[idx_seg] = []
            for idx in range(len(rnd_test)):
                # Object under analysis (oua)
                # SSD
                oua = SSD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    SSD_seg[idx_seg].append(oua)

                # MAD
                oua = MAD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    MAD_seg[idx_seg].append(oua)

                # PRD
                oua = PRD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    PRD_seg[idx_seg].append(oua)

                # COS SIM
                oua = COS_SIM_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    COS_SIM_seg[idx_seg].append(oua)

        # Processing the last index
        # SSD
        SSD_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = SSD_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                SSD_seg[-1].append(oua)

        SSD_seg_all.append(SSD_seg)  # [exp][seg][item]

        # MAD
        MAD_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = MAD_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                MAD_seg[-1].append(oua)

        MAD_seg_all.append(MAD_seg)  # [exp][seg][item]

        # PRD
        PRD_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = PRD_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                PRD_seg[-1].append(oua)

        PRD_seg_all.append(PRD_seg)  # [exp][seg][item]

        # COS SIM
        COS_SIM_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = COS_SIM_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                COS_SIM_seg[-1].append(oua)

        COS_SIM_seg_all.append(COS_SIM_seg)  # [exp][seg][item]

    # Printing Tables
    seg_table_column_name = []
    for idx_seg in range(len(segm) - 1):
        column_name = str(segm[idx_seg]) + ' < noise < ' + str(segm[idx_seg + 1])
        seg_table_column_name.append(column_name)

    # SSD Table
    SSD_seg_all = np.array(SSD_seg_all, dtype="object")
    SSD_seg_all = np.swapaxes(SSD_seg_all, 0, 1)

    print('\n')
    print('Printing Table for different noise values on the SSD metric')
    experiments_dict = vs.generate_table(seg_table_column_name, SSD_seg_all, Exp_names, experiments_dict, 'SSD')

    # MAD Table
    MAD_seg_all = np.array(MAD_seg_all, dtype="object")
    MAD_seg_all = np.swapaxes(MAD_seg_all, 0, 1)

    print('\n')
    print('Printing Table for different noise values on the MAD metric')
    experiments_dict = vs.generate_table(seg_table_column_name, MAD_seg_all, Exp_names, experiments_dict, 'MAD')

    # PRD Table
    PRD_seg_all = np.array(PRD_seg_all, dtype="object")
    PRD_seg_all = np.swapaxes(PRD_seg_all, 0, 1)

    print('\n')
    print('Printing Table for different noise values on the PRD metric')
    experiments_dict = vs.generate_table(seg_table_column_name, PRD_seg_all, Exp_names, experiments_dict, 'PRD')

    # COS SIM Table
    COS_SIM_seg_all = np.array(COS_SIM_seg_all, dtype="object")
    COS_SIM_seg_all = np.swapaxes(COS_SIM_seg_all, 0, 1)

    print('\n')
    print('Printing Table for different noise values on the COS SIM metric')
    experiments_dict = vs.generate_table(seg_table_column_name, COS_SIM_seg_all, Exp_names, experiments_dict, 'COS_SIM')

    for Exp_name in Exp_names:
        outname = f'score/{Exp_name}.pkl'
        pickle.dump(experiments_dict[Exp_name],open(outname,"wb"))

    ##############################################################################################################
    # Metrics graphs
    vs.generate_hboxplot(SSD_all, Exp_names, 'SSD (au)', log=False, set_x_axis_size=(0, 100.1))
    vs.generate_hboxplot(MAD_all, Exp_names, 'MAD (au)', log=False, set_x_axis_size=(0, 3.01))
    vs.generate_hboxplot(PRD_all, Exp_names, 'PRD (au)', log=False, set_x_axis_size=(0, 100.1))
    vs.generate_hboxplot(COS_SIM_all, Exp_names, 'Cosine Similarity (0-1)', log=False, set_x_axis_size=(0, 1))

    ################################################################################################################
    # Visualize signals

    signals_index = np.array([110, 210, 410, 810, 1610, 3210, 6410, 12810]) + 10

    ecg_signals2plot = []
    ecgbl_signals2plot = []
    dl_signals2plot = []
    fil_signals2plot = []

    signal_amount = 10

    [X_test, y_test, y_pred] = test_SEMamba
    for id in signals_index:
        ecgbl_signals2plot.append(X_test[id])
        ecg_signals2plot.append(y_test[id])
        dl_signals2plot.append(y_pred[id])

    # [X_test, y_test, y_filter] = test_IIR
    # for id in signals_index:
    #     fil_signals2plot.append(y_filter[id])

    # for i in range(len(signals_index)):
    #     vs.ecg_view(ecg=ecg_signals2plot[i],
    #                 ecg_blw=ecgbl_signals2plot[i],
    #                 ecg_dl=dl_signals2plot[i],
    #                 ecg_f=fil_signals2plot[i],
    #                 signal_name=None,
    #                 beat_no=None)

    #     vs.ecg_view_diff(ecg=ecg_signals2plot[i],
    #                      ecg_blw=ecgbl_signals2plot[i],
    #                      ecg_dl=dl_signals2plot[i],
    #                      ecg_f=fil_signals2plot[i],
    #                      signal_name=None,
    #                      beat_no=None)





