<<<<<<< HEAD
import os
import gc
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from pandarallel import pandarallel
from skmultilearn.model_selection import IterativeStratification

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR

from model import AKIPredictionModel, CustomBCELoss, PCGrad, EarlyStopping

tqdm.pandas()
pd.set_option('mode.chained_assignment',  None)
pandarallel.initialize(nb_workers=8,progress_bar=False)

def filter(icu, hosp, RRT_icu, RRT_hosp, KT_hosp):

    icu['RRT'] = 0

    def RRT_in_icu(target):

        target.reset_index(inplace=True,drop=True)

        RRT = RRT_icu[RRT_icu['stay_id'] == target['stay_id'].iloc[0]]

        if not RRT.empty:
            target['outtime'] = RRT['starttime'].iloc[0]
            target = target[target['charttime'] < target['outtime']]
            if not target.empty:
                target['RRT'].iloc[-1] = 1
        return target

    icu = icu.groupby('stay_id').parallel_apply(RRT_in_icu).reset_index(drop=True)

    icu['RRT_icu_history'] = icu.groupby('subject_id')['charttime'].transform(
    lambda x: (x > RRT_icu.loc[RRT_icu['subject_id'] == x.name, 'starttime'].min()).astype(int)
    )

    def RRT_in_hosp(target):

        target.reset_index(inplace=True,drop=True)

        from datetime import timedelta

        RRT = RRT_hosp[RRT_hosp['subject_id'] == target['subject_id'].iloc[0]]

        if not RRT.empty:
            for _, row in RRT.iterrows():
                end_criteria = row['chartdate'] + timedelta(days=1, hours=23, minutes=59)
                start_criteria = row['chartdate']
                target = target[~((target['charttime'] > start_criteria) & (target['charttime'] < end_criteria))]

        return target
    

    icu = icu.assign(RRT_hosp_history=icu.groupby('subject_id')['charttime'].transform(
    lambda x: (x > (RRT_hosp.loc[RRT_hosp['subject_id'] == x.name, 'chartdate'].min() + timedelta(days=1, hours=23, minutes=59))).astype(int)
    ))

    def KT_in_hosp(target):

        target.reset_index(inplace=True,drop=True)

        from datetime import timedelta

        KT = KT_hosp[KT_hosp['subject_id'] == target['subject_id'].iloc[0]]
    
        if not KT.empty:
            criteria = KT['chartdate'].min() + timedelta(days=0,hours=0,minutes=0)
            target = target[target['charttime'] < criteria]

        return target

    icu = icu.groupby('subject_id').parallel_apply(KT_in_hosp).reset_index(drop=True)

    if not hosp.empty:
        hosp = hosp.groupby('subject_id').parallel_apply(RRT_in_hosp).reset_index(drop=True)
        hosp = hosp.groupby('subject_id').parallel_apply(KT_in_hosp).reset_index(drop=True)

    return icu, hosp

def diff(icu, hosp):

    icu[['min', 'max', 'median', 'mean', 'diff']] = np.nan
    hosp[['min', 'max', 'median', 'mean', 'diff']] = np.nan
    SCr = pd.concat([icu, hosp]).dropna(subset=['SCr'])

    def operation(target):

        target.reset_index(inplace=True,drop=True)

        import numpy as np
        import pandas as pd
        from datetime import timedelta
        
        target_SCr = SCr[SCr['subject_id'] == target['subject_id'].iloc[0]]
        
        cri = target['charttime']
        cri_48 = cri - timedelta(hours=48)
        
        for i, charttime in enumerate(cri):
            value_SCr = target_SCr.loc[(target_SCr['charttime'] < charttime) & (target_SCr['charttime'] >= cri_48.iloc[i]), 'SCr']
            
            if not value_SCr.empty:
                
                if pd.isna(target.at[i, 'min']):
                    target.at[i, 'min'] = value_SCr.min()
                    if not pd.isna(target.at[i, 'SCr']):
                        target.at[i, 'diff'] = target.at[i, 'SCr'] - value_SCr.min()
            
                if pd.isna(target.at[i, 'max']):
                    target.at[i, 'max'] = value_SCr.max()

                if pd.isna(target.at[i, 'median']):
                    target.at[i, 'median'] = value_SCr.median()

                if pd.isna(target.at[i, 'mean']):
                    target.at[i, 'mean'] = value_SCr.mean()
            else:

                if pd.isna(target.at[i, 'min']):
                    if not pd.isna(target.at[i, 'SCr']):
                        target.at[i, 'min'] = target.at[i, 'SCr']
                        target.at[i, 'diff'] = 0

                if pd.isna(target.at[i, 'max']):
                    if not pd.isna(target.at[i, 'SCr']):
                        target.at[i, 'max'] = target.at[i, 'SCr']

                if pd.isna(target.at[i, 'median']):
                    if not pd.isna(target.at[i, 'SCr']):
                        target.at[i, 'median'] = target.at[i, 'SCr']

                if pd.isna(target.at[i, 'mean']):
                    if not pd.isna(target.at[i, 'SCr']):
                        target.at[i, 'mean'] = target.at[i, 'SCr']

        return target

    icu = icu.groupby('subject_id', group_keys=False).parallel_apply(operation).reset_index(drop=True)
    hosp = hosp.groupby('subject_id', group_keys=False).parallel_apply(operation).reset_index(drop=True)

    for column in ['min', 'max', 'median', 'mean', 'diff']:
        icu[column] = icu[column].round(1)
        hosp[column] = hosp[column].round(1)

    return icu, hosp

def SCr_gap(target):

    label = 'SCr'
    target[label] = target[label].round(1)
    target[label + '_diff'] = target[label].diff().round(1)

    label = 'charttime'
    target[label + '_diff'] = target[label].diff()
    target['SCr_' + label + '_diff'] = (target[label + '_diff'].dt.total_seconds() / 3600).round(1)
        
    return target

def Pre_admission(icu, hosp):

    common_columns = icu.columns.intersection(hosp.columns).tolist()
    SCr = pd.concat([icu[common_columns], hosp[common_columns]]).drop_duplicates().reset_index(drop=True)

    def MDRD(df):
            
        if df['gender'] == 'F' and df['race'] == 'BLACK': df['baseline'] = (75 / (0.742 * 1.21 * 186 * df['age'] ** (-0.203))) ** (-1 / 1.154)
        elif df['race'] == 'BLACK': df['baseline'] = (75 / (1 * 1.21 * 186 * df['age'] ** (-0.203))) ** (-1 / 1.154)
        elif df['gender'] == 'F': df['baseline'] = (75 / (0.742 * 1 * 186 * df['age'] ** (-0.203))) ** (-1 / 1.154)
        else: df['baseline'] = (75 / (1 * 1 * 186 * df['age'] ** (-0.203))) ** (-1 / 1.154)
    
        return round(df['baseline'], 1)

    def operation(target):

        target.reset_index(inplace=True,drop=True)

        from datetime import timedelta
        import numpy as np
        
        target_SCr = SCr[SCr['subject_id'] == target['subject_id'].iloc[0]]
        target_stay_id = target['stay_id'].drop_duplicates()

        if not target_SCr.empty :
            for i in target_stay_id:    
                intime = target.loc[target['stay_id'] == i, 'intime'].iloc[0]
                intime_7 = intime - timedelta(days=7)
                value_SCr = target_SCr.loc[(target_SCr['charttime'] < intime) & (target_SCr['charttime'] >= intime_7), 'SCr'] 

                if not value_SCr.empty : 
                    target.loc[target['stay_id'] == i, ['baseline', 'method']] = value_SCr.min(), 1
                else : 
                    intime_365 = intime - timedelta(days=365)
                    value_SCr = target_SCr.loc[(target_SCr['charttime'] <= intime_7) & (target_SCr['charttime'] >= intime_365), 'SCr']
                    if not value_SCr.empty : 
                        target.loc[target['stay_id'] == i, ['baseline', 'method']] = np.median(value_SCr), 2
                    else : 
                        target.loc[target['stay_id'] == i, ['baseline', 'method']] = MDRD(target.iloc[0]), 0
        else : 
            for i in target_stay_id:
                target.loc[target['stay_id'] == i, ['baseline', 'method']] = MDRD(target.iloc[0]), 0

        return target

    icu = icu.groupby('subject_id', group_keys=False).parallel_apply(operation).reset_index(drop=True)
    
    return icu

def SCr_AKI_stage(df):
    
    df['ratio'] = df['SCr'] / df['baseline']
    df['SCr_stage'] = 0

    condition_1 = ((df['ratio'] < 2) & (df['ratio'] >= 1.5)) | ((df['diff'] >= 0.3) & (df['diff'] < 4))
    condition_2 = (df['ratio'] < 3) & (df['ratio'] >= 2)
    condition_3 = (df['ratio'] >= 3) | (df['diff'] >= 4) | (df['RRT'] == 1)

    df.loc[condition_1, 'SCr_stage'] = 1
    df.loc[condition_2, 'SCr_stage'] = 2
    df.loc[condition_3, 'SCr_stage'] = 3

    df.loc[(df['SCr'].isnull()) & (df['RRT'] != 1), 'SCr_stage'] = 0

    return df

def SCr_resampling(df, label):

    def operation(group):
        
        import pandas as pd
        import numpy as np
        from datetime import timedelta

        start_frame = group.iloc[[0]].copy()
        end_frame = group.iloc[[0]].copy()

        start_frame['charttime'] = start_frame['intime']
        end_frame['charttime'] = end_frame['outtime']

        na_columns = [label, 'diff', 'min', 'max', 'median', 'mean', 'ratio', 'SCr_diff', 'SCr_stage', 'SCr_charttime_diff']
        start_frame[na_columns] = np.nan
        end_frame[na_columns] = np.nan

        group = pd.concat([start_frame, group, end_frame])
        group.set_index('charttime', inplace=True)

        resampled = group.resample('6h', origin=group['intime'].iloc[0], label='left').last()

        if pd.isna(resampled[label].iloc[-1]):
            last_contribute = group[(group.index < group['outtime']) & (group.index >= group['outtime'] - timedelta(hours=6))]
            if not last_contribute.empty:
                resampled.iloc[-1] = last_contribute.iloc[-1]

        if pd.isna(resampled[label].iloc[0]):
            first_contribute = group[(group.index > group['intime']) & (group.index <= group['intime'] + timedelta(hours=6))]
            if not first_contribute.empty:
                resampled.iloc[0] = first_contribute.iloc[-1]

        for col in ['subject_id', 'hadm_id', 'stay_id', 'first_careunit', 'intime', 'outtime', 'RRT_icu_history', 'RRT_hosp_history', 'los', 'race', 'age', 'gender', 'baseline', 'method']:
            resampled[col] = group[col].iat[0]

        resampled['charttime'] = resampled.index
        resampled['timedelta'] = resampled.index - group['intime'].iloc[0]

        resampled['RRT'] = 0
        if group['RRT'].sum() > 0:
            resampled['RRT'].iloc[-1] = 1
        else:
            resampled['RRT'] = resampled['RRT'].fillna(0)

        return resampled

    df = df.groupby('stay_id', group_keys=False).parallel_apply(operation).reset_index(drop=True)
    df['timedelta'] = pd.to_timedelta(df['timedelta'])
    df['SCr_presence'] = df['SCr'].notna().astype(int)

    return df

def SCr_copy_mask(df, df_icu, df_hosp, label):

    df_SCr = pd.concat([df_icu, df_hosp])
    columns_to_fill = ['min', 'max', 'mean', 'median', 'diff', 'ratio', 'SCr_stage', 'SCr_diff', 'SCr_charttime_diff']

    def SCr_copy(target):

        for col in [label] + columns_to_fill:
            target[col] = target[col].fillna(method='ffill', limit=4)
        return target

    def SCr_copy_hosp(target):

        target.reset_index(inplace=True,drop=True)

        from datetime import timedelta

        if target[label].isnull().iloc[0]:
            target_hosp = df_SCr.loc[df_SCr['subject_id'] == target['subject_id'].iloc[0]]
            forward = target_hosp[(target_hosp['charttime'] < target['charttime'].iloc[0]) & 
                                  (target_hosp['charttime'] > (target['charttime'].iloc[0] - timedelta(days=1)))]
            if not forward.empty:
                target_value = forward.iloc[-1]
                cri = target_value['charttime']
                target_2 = target[(target['charttime'] > cri) & (target['charttime'] < (cri + timedelta(days=1)))]
                if not target_2.empty:
                    for col in [label] + columns_to_fill:
                        target.loc[target_2.index, col] = target_value[col]
        return target

    def SCr_mask(target):

        target.reset_index(inplace=True,drop=True)

        columns = ['SCr', 'ratio', 'min', 'max', 'mean', 'median', 'diff', 'SCr_stage', 'SCr_diff', 'SCr_charttime_diff']
        for col in columns:
            target[f'{col}_mask'] = target[col].notna().astype(int)
            target[col] = target[col].fillna(0)
    
        return target

    df = df.groupby('stay_id', group_keys=False).parallel_apply(SCr_copy).reset_index(drop=True)
    df = df.groupby('stay_id', group_keys=False).parallel_apply(SCr_copy_hosp).reset_index(drop=True)
    df = SCr_mask(df)

    return df

def fetch_outputevents(itemids: str, engine):
    query = f"""
    SELECT subject_id, hadm_id, stay_id, charttime, itemid, value 
    FROM mimiciv_icu.outputevents 
    WHERE value IS NOT NULL
    AND itemid IN ({itemids})
    AND value > 0
    ORDER BY stay_id, charttime
    """
    return pd.read_sql_query(query, engine)


def preprocess_weight_data(df, chartevents, omr):
    
    Weight_icu_kg = chartevents[chartevents['itemid'].isin([224639, 226512])]
    Weight_icu_lbs = chartevents[chartevents['itemid'].isin([226531])]
    Weight_icu_lbs['valuenum'] = Weight_icu_lbs['valuenum'] / 2.20462  # lbs to kg
    Weight_icu = pd.concat([Weight_icu_kg, Weight_icu_lbs])
    Weight_icu = Weight_icu.query('27.2155 <= valuenum <= 317.515').copy()
    Weight_icu['valuenum'] = Weight_icu['valuenum'].round(1)
    Weight_icu = Weight_icu.rename(columns={'valuenum': 'Weight'})

    omr_weight_kg = omr[omr['result_name'] == 'Weight']
    omr_weight_lbs = omr[omr['result_name'] == 'Weight (Lbs)']

    omr_weight_kg['result_value'] = pd.to_numeric(omr_weight_kg['result_value'], errors='coerce')
    omr_weight_lbs['result_value'] = pd.to_numeric(omr_weight_lbs['result_value'], errors='coerce') / 2.20462  # lbs to kg

    omr_weight = pd.concat([omr_weight_kg, omr_weight_lbs])
    omr_weight = omr_weight.dropna(subset=['result_value'])
    omr_weight['chartdate'] = omr_weight['chartdate'].astype(str)
    omr_weight['charttime'] = pd.to_datetime(omr_weight['chartdate'] + ' 23:59:59')
    omr_weight['Weight'] = omr_weight['result_value'].round(1)
    omr_weight = omr_weight.query('27.2155 <= Weight <= 317.515').copy()
    omr_weight = omr_weight.sort_values(['subject_id', 'charttime']).drop_duplicates(subset=['subject_id', 'charttime'], keep='last')

    Weight_pool = pd.concat([omr_weight, Weight_icu]).sort_values(['charttime'])

    def Weight(target_icu):

        import pandas as pd

        target_Weight = Weight_pool[Weight_pool['subject_id'] == target_icu['subject_id'].iloc[0]]
        target_Weight = target_Weight.sort_values('charttime').reset_index(drop=True)
        target_icu = target_icu.sort_values('charttime').reset_index(drop=True)

        target_icu['Weight'] = pd.merge_asof(
            target_icu[['charttime']],
            target_Weight[['charttime', 'Weight']],
            on='charttime',
            direction='backward'
        )['Weight']

        return target_icu

    df = df.groupby('subject_id', group_keys=False).parallel_apply(Weight).reset_index(drop=True)
    
    return df

def Urine(df):

    df = df.sort_values(by=['stay_id', 'charttime'])
    df['charttime_diff'] = df.groupby('stay_id')['charttime'].diff().fillna(timedelta(seconds=0))
    df = df.assign(**{'6h': np.nan, '12h': np.nan, '24h': np.nan, 'Anuria_12h': np.nan, 'Urine_stage': np.nan, 'Urine_output_rate': np.nan})

    anuria_threshold = 50.0

    def operation(target):

        target = target.reset_index(drop=True)
        target['Urine_output_rate'] = target['Urine'] / (target['charttime_diff'].dt.total_seconds() / 3600.0) / target['Weight']
        target['cum_value'] = target['Urine'][::-1].cumsum()
        target['cum_time_diff'] = target['charttime_diff'][::-1].cumsum().dt.total_seconds() / 3600.0

        for i in range(1, len(target)):
            group = target.iloc[1:i+1]
            group['cum_value'] = group['cum_value'] - group['cum_value'].iloc[-1] + group['Urine'].iloc[-1]
            group['cum_time_diff'] = group['cum_time_diff'] - group['cum_time_diff'].iloc[-1] + (group['charttime_diff'].dt.total_seconds() / 3600.0).iloc[-1]

            for threshold_hours_min, threshold_hours_max, rate_threshold, column_name, stage in [
                (6, 12, 0.5, '6h', 1),
                (12, float('inf'), 0.5, '12h', 2),
                (24, float('inf'), 0.3, '24h', 3)]:

                condition = (group['cum_time_diff'] >= threshold_hours_min) & (group['cum_time_diff'] <= threshold_hours_max)
                filtered_group = group.loc[condition]

                if not filtered_group.empty:
                    urine_output_rate = filtered_group['cum_value'] / filtered_group['cum_time_diff'] / target['Weight'].iloc[i]

                    if urine_output_rate.iloc[-1] < rate_threshold:
                        target.at[i, column_name] = 1
                        target.at[i, 'Urine_stage'] = stage

                    if column_name == '12h' and filtered_group['cum_value'].iloc[-1] < anuria_threshold:
                        target.at[i, 'Anuria_12h'] = 1
                        target.at[i, 'Urine_stage'] = 3

                    if column_name in ['6h', '12h', '24h']:
                        target.at[i, f'Urine_volume_{column_name}'] = filtered_group['cum_value'].iloc[-1]
                        target.at[i, f'Urine_output_rate_{column_name}'] = urine_output_rate.iloc[-1]

        return target

    df = df.groupby('stay_id', group_keys=False).parallel_apply(operation).reset_index(drop=True)
    df = df[(~df['Urine_output_rate'].isna()) & (df['Urine_output_rate'] != float('inf'))]
    df['Urine_charttime_diff'] = (df['charttime_diff'].dt.total_seconds() / 3600).round(1)
    df.loc[df['Urine_charttime_diff'] >= 12, ['Urine_stage', 'Anuria_12h']] = 3, 1

    return df

def Urine_resampling(df, label):

    def operation(group):
        
        import pandas as pd
        import numpy as np
        from datetime import timedelta

        start_frame = group.iloc[[0]].copy()
        end_frame = group.iloc[[0]].copy()

        start_frame['charttime'] = start_frame['intime']
        end_frame['charttime'] = end_frame['outtime']

        na_columns = [label, 'Urine', 'Weight', '6h', '12h', '24h', 'Anuria_12h', 'Urine_stage', 'cum_value', 'cum_time_diff', 'Urine_charttime_diff', 'Urine_output_rate_6h', 'Urine_output_rate_12h', 'Urine_output_rate_24h', 'Urine_volume_6h', 'Urine_volume_12h', 'Urine_volume_24h']
        start_frame[na_columns] = np.nan
        end_frame[na_columns] = np.nan

        group = pd.concat([start_frame, group, end_frame])
        group.set_index('charttime', inplace=True)

        resampled = group.resample('6h', origin=group['intime'].iloc[0], label='left').last()

        if pd.isna(resampled[label].iloc[-1]):
            last_contribute = group[(group.index < group['outtime']) & (group.index >= group['outtime'] - timedelta(hours=6))]
            if not last_contribute.empty:
                resampled.iloc[-1] = last_contribute.iloc[-1]

        if pd.isna(resampled[label].iloc[0]):
            first_contribute = group[(group.index > group['intime']) & (group.index <= group['intime'] + timedelta(hours=6))]
            if not first_contribute.empty:
                resampled.iloc[0] = first_contribute.iloc[-1]

        for col in ['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'RRT_icu_history', 'RRT_hosp_history']:
            resampled[col] = group[col].iat[0]

        resampled['charttime'] = resampled.index
        resampled['timedelta'] = resampled.index - group['intime'].iloc[0]

        resampled['RRT'] = 0
        if group['RRT'].sum() > 0:
            resampled['RRT'].iloc[-1] = 1
        else:
            resampled['RRT'] = resampled['RRT'].fillna(0)

        return resampled

    df = df.groupby('stay_id', group_keys=False).parallel_apply(operation).reset_index(drop=True)
    df['timedelta'] = pd.to_timedelta(df['timedelta'])
    df['current_charttime'] = round((df['timedelta'] + timedelta(hours=6)).dt.total_seconds() / 3600, 1)
    df['Urine_presence'] = df['Urine'].notna().astype(int)

    return df

def Urine_copy_mask(df, label):

    def Urine_copy(target):
        columns_to_fill = [
            label, 'Urine_charttime_diff', 'Urine_stage', 'Urine', 
            '6h', '12h', '24h', 'Anuria_12h', 'cum_value', 'cum_time_diff', 
            'Urine_output_rate_6h','Urine_output_rate_12h', 'Urine_output_rate_24h',
            'Urine_volume_6h', 'Urine_volume_12h', 'Urine_volume_24h'
        ]
        
        for col in columns_to_fill:
            target[col] = target[col].fillna(method='ffill', limit=4)
        
        return target

    df = df.groupby('stay_id', group_keys=False).parallel_apply(Urine_copy).reset_index(drop=True)

    def Urine_mask(target):
        columns_to_mask = [
            label, 'Urine_charttime_diff', 'Urine_stage','Urine',
            '6h', '12h', '24h', 'Anuria_12h', 'Weight', 'cum_value', 'cum_time_diff', 
            'Urine_output_rate_6h', 'Urine_output_rate_12h', 'Urine_output_rate_24h', 
            'Urine_volume_6h', 'Urine_volume_12h', 'Urine_volume_24h'
        ]
        
        for col in columns_to_mask:
            mask_col = col + '_mask'
            target[mask_col] = target[col].notna().astype(int)
            target[col] = target[col].fillna(0)

        return target

    df = Urine_mask(df)
    return df

def GT(df, stage_pool_total, stage_pool_SCr_total, RRT_pool_total):
    presence_cols = [f'GT_presence_{i}' for i in range(6, 49, 6)]
    presence_cols_SCr = [f'GT_presence_{i}_SCr' for i in range(6, 49, 6)]

    stage_cols = ['GT_stage_1', 'GT_stage_2', 'GT_stage_3', 'GT_stage_3D']
    stage_cols_SCr = ['GT_stage_1_SCr', 'GT_stage_2_SCr', 'GT_stage_3_SCr', 'GT_stage_3D_SCr']

    for col in presence_cols + stage_cols + presence_cols_SCr + stage_cols_SCr:
        df[col] = 0

    df['charttime_end'] = df['charttime'] + timedelta(hours=6)

    def operation(target):

        from datetime import timedelta

        target = target.sort_values('charttime').reset_index(drop=True)
        stage_pool = stage_pool_total[stage_pool_total['stay_id'].isin(target['stay_id'].unique())]
        stage_pool_SCr = stage_pool_SCr_total[stage_pool_SCr_total['stay_id'].isin(target['stay_id'].unique())]
        RRT_pool = RRT_pool_total[RRT_pool_total['stay_id'].isin(target['stay_id'].unique())]

        for hours in range(6, 49, 6):
            target[f'charttime_{hours}'] = target['charttime_end'] + timedelta(hours=hours)

        for i, row in target.iterrows():
            for hours in range(6, 49, 6):
                area = stage_pool[(stage_pool['charttime'] > row['charttime_end']) & 
                                  (stage_pool['charttime'] <= row[f'charttime_{hours}'])]['stage']
                area_SCr = stage_pool_SCr[(stage_pool_SCr['charttime'] > row['charttime_end']) & 
                                  (stage_pool_SCr['charttime'] <= row[f'charttime_{hours}'])]['stage']
                RRT_count = len(RRT_pool[(RRT_pool['starttime'] >= row['charttime_end']) & 
                                         (RRT_pool['starttime'] <= row[f'charttime_{hours}'])]) + target['RRT'].iloc[i]

                if area.sum() > 0:
                    target.loc[i, f'GT_presence_{hours}'] = 1
                    if hours == 48:
                        max_stage = area.max()
                        if max_stage >= 1:
                            target.loc[i, 'GT_stage_1'] = 1
                        if max_stage >= 2:
                            target.loc[i, 'GT_stage_2'] = 1
                        if max_stage >= 3:
                            target.loc[i, 'GT_stage_3'] = 1

                if area_SCr.sum() > 0:
                    target.loc[i, f'GT_presence_{hours}_SCr'] = 1
                    if hours == 48:
                        max_stage = area_SCr.max()
                        if max_stage >= 1:
                            target.loc[i, 'GT_stage_1_SCr'] = 1
                        if max_stage >= 2:
                            target.loc[i, 'GT_stage_2_SCr'] = 1
                        if max_stage >= 3:
                            target.loc[i, 'GT_stage_3_SCr'] = 1
                            
                if RRT_count > 0:
                    target.loc[i, [f'GT_presence_{hours}'] + stage_cols] = 1
                    target.loc[i, [f'GT_presence_{hours}_SCr'] + stage_cols_SCr] = 1

        return target

    df = df.groupby('stay_id', group_keys=False).parallel_apply(operation)
    df = df.sort_values(['subject_id', 'stay_id', 'charttime']).reset_index(drop=True)

    return df

def MAX_AKI(df,SCr_icu,Urine_icu):

    def operation(target):

        SCr = SCr_icu[SCr_icu['stay_id'] == target['stay_id'].iloc[0]]
        Urine = Urine_icu[Urine_icu['stay_id'] == target['stay_id'].iloc[0]]
        
        max_SCr = max(SCr['SCr_stage'])
        max_Urine = max(Urine['Urine_stage'])
        
        target['max_stage'] = max(max_SCr,max_Urine)

        return target

    df = df.groupby('stay_id',group_keys=False).parallel_apply(operation).reset_index(drop=True)
    
    return df

def onehot(df):
      
    careunit_columns = ['Surgical', 'Medical', 'Medical/Surgical', 'Other']

    for col in careunit_columns:
        df.loc[df['first_careunit'] == col, col] = 1
        df[col] = df[col].fillna(0)
        
    race_columns = ['WHITE', 'UNKNOWN', 'BLACK', 'HISPANIC OR LATINO', 'OTHER', 'ASIAN']
    
    for col in race_columns:

        if col == 'BLACK':
            df.loc[df['race'] == col, col] = 1
            df[col] = df[col].fillna(0)

        else :
            df.loc[df['race'] == col, 'BLACK'] = 0
            df['BLACK'] = df['BLACK'].fillna(0)

    df.loc[df['gender'] == 'F','gender'] = 0
    df.loc[df['gender'] == 'M','gender'] = 1
    
    df['length'] = df.groupby('stay_id')['stay_id'].transform('size')

    return df

def ICD(df, icd_list):
    df_icd = df[df['icd_code'].str.startswith(tuple(icd_list))]
    return df_icd

def check_ICD(stage,df9,df10,label):

    check = pd.concat([df9,df10])
    check.drop_duplicates(subset='hadm_id',inplace=True)
    check[label] = 1
    check = check[['hadm_id',label]]
    check = pd.merge(stage,check,on='hadm_id',how='left')
    check[label] = check[label].fillna(0)

    return check 

def extract_icd_matches(df9, df10, codes_9, codes_10):
    matched_9 = ICD(df9, codes_9)
    matched_10 = ICD(df10, codes_10)
    return matched_9, matched_10

def add_comorbidity(stage_df, df9, df10, codes_9, codes_10, label):
    disease_9, disease_10 = extract_icd_matches(df9, df10, codes_9, codes_10)
    return check_ICD(stage_df, disease_9, disease_10, label)

def gap(target, label='valuenum', use_label_column=True):
    if use_label_column and label in target.columns:
        target[label] = target[label].round(1)
        target[label + '_diff'] = target[label].diff().round(1)
    else:
        target[label] = target['valuenum'].round(1)
        target[label + '_diff'] = target['valuenum'].diff().round(1)

    return target

def Mapping(df,df_icu,df_hosp,label,copy):
    
    if not df_hosp.empty :
        common_columns = df_icu.columns.intersection(df_hosp.columns).tolist()
        df_data = pd.concat([df_icu[common_columns], df_hosp[common_columns]]).drop_duplicates().sort_values(['subject_id', 'charttime']).reset_index(drop=True)
    
    else :
        df_data = df_icu.drop_duplicates().sort_values(['subject_id', 'charttime']).reset_index(drop=True)
    
    if not label in df.columns :
        df[label] = np.nan
        df[label + '_diff'] = np.nan

    if not label in df_data.columns:
        df_data.rename(columns = {'valuenum_diff': label + '_diff'}, inplace = True)    
        df_data.rename(columns = {'valuenum': label}, inplace = True)    

    def operation(target):

        import pandas as pd
        from datetime import timedelta

        target.reset_index(inplace=True,drop=True)
        target_data = df_data[df_data['subject_id'] == target['subject_id'].iloc[0]]

        for i in range(len(target)):

            target_lower = target['charttime'].iloc[i]
            target_upper = target_lower  + timedelta(hours=6)
            target_value = target_data[(target_data['charttime'] < target_upper) & (target_data['charttime'] >= target_lower)].sort_values(['charttime'])

            if not target_value.empty: 
                if pd.isna(target[label].iloc[i]):
                    target[label].iloc[i] = target_value[label].iloc[-1]
                    target[label + '_diff'].iloc[i] = target_value[label + '_diff'].iloc[-1]

        if copy:

            target[label] = target[label].fillna(method='ffill', limit=4)
            target[label + '_diff'] = target[label + '_diff'].fillna(method='ffill', limit=4)

            if target[label].isnull().iloc[0] == True:
                
                target_hosp = df_hosp.loc[df_hosp['subject_id'] == target['subject_id'].iloc[0]]
                
                forward = target_hosp[(target_hosp['charttime'] < target['charttime'].iloc[0]) & (target_hosp['charttime'] >= (target['charttime'].iloc[0] - timedelta(days=1)))]
                forward = forward.sort_values(['charttime'])

                if not forward.empty:
                
                    target_value = forward.iloc[-1]
                    cri = target_value['charttime']
                    target_2 = target[(target['charttime'] > cri) & (target['charttime'] < (cri + timedelta(days=1)))]
                    target_2 = target_2[label].isnull().sum()

                    if target_2 != 0:
                        target.loc[target.index[:target_2], label] = target_value[label]
                        target.loc[target.index[:target_2], label + '_diff'] = target_value[label + '_diff']

        return target
    
    df = df.groupby('stay_id',group_keys=False).parallel_apply(operation).reset_index(drop=True)
    print(df[label].isnull().sum())
    
    df[label + '_mask']  = df[label].notna().astype(int)
    df[label + '_diff_mask']  = df[label + '_diff'].notna().astype(int)

    return df

def Vital(df, stage, vitalsign):

    def convert_temperature(df):
        df['valuenum'] = (df['valuenum'] - 32) * 5 / 9
        df['itemid'] = 223762
        df['valueuom'] = '°C'
        return df

    item_config = {
        'temperature': {'itemids': [223762, 223761], 'range': (32, 43), 'unit': '°C'},
        'heartrate': {'itemids': [220045], 'range': (0, 300), 'unit': 'bpm'},
        'sbp': {'itemids': [220050, 220179], 'range': (0, 300), 'unit': 'mmHg'},
        'dbp': {'itemids': [220051, 220180], 'range': (10, 175), 'unit': 'mmHg'},
        'resprate': {'itemids': [220210], 'range': (0, 60), 'unit': 'insp/min'},
        'o2sat': {'itemids': [220227, 220277], 'range': (0, 100), 'unit': '%'},
    }

    subject_id = stage[['subject_id']].drop_duplicates()

    df_all = {}
    for label, info in item_config.items():
        dfs = []
        for idx, itemid in enumerate(info['itemids']):
            df_v = df[df['itemid'] == itemid].copy()
            if label == 'temperature' and itemid == 223761:
                df_v = convert_temperature(df_v)
            df_v = df_v[(df_v['valuenum'] >= info['range'][0]) & (df_v['valuenum'] <= info['range'][1])]
            df_v['valueuom'] = info['unit']
            df_v = pd.merge(df_v, subject_id, on='subject_id', how='inner')
            df_v = df_v.sort_values(['subject_id', 'charttime'])
            df_v[label] = df_v['valuenum'].round(1)
            df_v[label + '_diff'] = df_v.groupby('subject_id')[label].diff().round(1)
            if len(info['itemids']) > 1:
                df_v['Art'] = int(idx == 0)
            dfs.append(df_v)

        df_all[label] = pd.concat(dfs).drop_duplicates().reset_index(drop=True)

        if label in vitalsign.columns:
            vitalsign[label] = vitalsign[label].where(
                (vitalsign[label] >= info['range'][0]) & (vitalsign[label] <= info['range'][1])
            )

    vitalsign_results = {}
    for label in item_config.keys():
        if label in vitalsign.columns:
            vdf = vitalsign[vitalsign[label].notna()][['subject_id', 'charttime', label]]
            vdf = vdf.sort_values(['subject_id', 'charttime']).reset_index(drop=True)
            vdf = vdf.groupby('subject_id', group_keys=False).apply(lambda x: gap(x, label)).reset_index(drop=True)
            vitalsign_results[label] = vdf

    combined_results = {}
    for label in ['sbp', 'dbp', 'o2sat']:
        df_combined = df_all[label]
        df_combined = df_combined.sort_values(['subject_id', 'charttime']).reset_index(drop=True)
        df_combined = df_combined.groupby('subject_id', group_keys=False).apply(lambda x: gap(x, label)).reset_index(drop=True)
        combined_results[label] = df_combined

    mapping_info = [
        ('temperature', df_all['temperature'], vitalsign_results['temperature'], True),
        ('heartrate', df_all['heartrate'], vitalsign_results['heartrate'], True),
        ('sbp', df_all['sbp'][df_all['sbp']['Art'] == 1], vitalsign_results['sbp'], False),
        ('sbp', df_all['sbp'][df_all['sbp']['Art'] == 0], vitalsign_results['sbp'], True),
        ('dbp', df_all['dbp'][df_all['dbp']['Art'] == 1], vitalsign_results['dbp'], False),
        ('dbp', df_all['dbp'][df_all['dbp']['Art'] == 0], vitalsign_results['dbp'], True),
        ('resprate', df_all['resprate'], vitalsign_results['resprate'], True),
        ('o2sat', df_all['o2sat'][df_all['o2sat']['Art'] == 1], vitalsign_results['o2sat'], False),
        ('o2sat', df_all['o2sat'][df_all['o2sat']['Art'] == 0], vitalsign_results['o2sat'], True),
    ]

    del df

    for label, df_src, df_vital, copy_flag in mapping_info:
        stage = Mapping(stage, df_src, df_vital, label, copy=copy_flag)

    return stage

def process_lab_data(variables, subject_ids, engine):

    results = []
    
    for var_name, icu_itemid, hosp_itemids, left_bound, right_bound, left_inclusive, right_inclusive in variables:

        if hosp_itemids is not None:

            query = f"""
            select subject_id, stay_id, charttime, itemid, valuenum
            from mimiciv_icu.chartevents
            where valuenum is not null and valuenum != 999999 and stay_id is not null and 
            itemid in ({', '.join(map(str, icu_itemid))})
            order by subject_id, itemid, charttime
            """

            var_icu = pd.read_sql(query,engine)
            var_icu = var_icu[var_icu['subject_id'].isin(subject_ids)]

            query = f"""
            select subject_id, charttime, itemid, valuenum
            from mimiciv_hosp.labevents
            where valuenum is not null and valuenum != 999999 and
            itemid in ({', '.join(map(str, hosp_itemids))})
            order by subject_id, itemid, charttime
            """
    
            var_hosp = pd.read_sql(query,engine)
            var_hosp = var_hosp[var_hosp['subject_id'].isin(subject_ids)]

            var_icu['ICU'] = 1
            var_hosp['ICU'] = 0

            var_data = pd.concat([var_icu, var_hosp]).sort_values(by=['subject_id', 'charttime'])

            if left_bound is not None:
                if left_inclusive:
                    var_data = var_data[var_data.valuenum >= left_bound]
                else:
                    var_data = var_data[var_data.valuenum > left_bound]

            if right_bound is not None:
                if right_inclusive:
                    var_data = var_data[var_data.valuenum <= right_bound]
                else:
                    var_data = var_data[var_data.valuenum < right_bound]
        else :
            query = f"""
            select subject_id, stay_id, itemid, starttime, amount
            from mimiciv_icu.inputevents
            Where amount is not null and amount != 999999 and
            itemid in ({', '.join(map(str, icu_itemid))})
            order by subject_id, starttime
            """

            var_icu = pd.read_sql(query,engine)
            var_icu = var_icu[var_icu['subject_id'].isin(subject_ids)]
            var_icu['ICU'] = 1
            var_data = var_icu
            var_data.rename(columns={'amount': 'valuenum','starttime':'charttime'}, inplace=True)

        var_data = var_data.groupby('subject_id', group_keys=False).parallel_apply(gap).reset_index(drop=True)
        var_data.rename(columns={'valuenum': var_name, 'valuenum_diff': f'{var_name}_diff'}, inplace=True)
        results.append(var_data)
        results = pd.concat(results).reset_index(drop=True)

        icu = results[results['ICU']==1]
        hosp = results[results['ICU']==0]

    return icu, hosp

def Anti_Mapping(df,ICU,label):

    data = ICU[['subject_id','starttime']]

    df[f'{label}'] = 0
    
    def operation(target):

        from datetime import timedelta

        target = target.reset_index(drop=True)
        subject_id = target['subject_id'].iloc[0]
        target_data = data[data['subject_id'] == subject_id]

        for i in range(len(target)):
            
            target_lower = target['charttime'].iloc[i]
            target_upper = target_lower + timedelta(hours=6)

            case = target_data[(target_data['starttime'] >= target_lower) & (target_data['starttime'] <= target_upper)]
            if not case.empty :
                target[f'{label}'].iloc[i] = 1

        return target
    
    df = df.groupby('subject_id',group_keys=False).parallel_apply(operation).reset_index(drop=True)
    print(df[f'{label}'].value_counts())

    return df

def merge_anti(target):
    merge_dict = {
        'Others': [
            'Aztreonam', 'Doxycycline', 'Tigecycline', 'Bactrim (SMX/TMP)', 'Azithromycin',
            'Erythromycin', 'Colistin', 'Daptomycin', 'Linezolid', 'Clindamycin',
            'Acyclovir', 'Rifampin', 'Amikacin', 'Gentamicin', 'Tobramycin'
        ],
        'Fluoroquinolones': ['Ciprofloxacin', 'Levofloxacin', 'Moxifloxacin'],
        'Penicillins': ['Ampicillin', 'Nafcillin', 'Penicillin G potassium', 'Penicillin gen4'],
        'Betalactam': ['Ampicillin/Sulbactam (Unasyn)', 'Piperacillin/Tazobactam (Zosyn)'],
        'Cephalosporins': ['Cefazolin', 'Cefepime', 'Ceftazidime', 'Ceftriaxone', 'Keflex', 'Ceftaroline'],
        'Carbapenems': ['Imipenem/Cilastatin', 'Meropenem', 'Ertapenem sodium (Invanz)'],
        
    }

    for new_col, cols_to_merge in merge_dict.items():
        target[new_col] = target[cols_to_merge].max(axis=1)
        target.drop(columns=cols_to_merge, inplace=True)

    return target

def MV(df, engine):

    query = """
        select subject_id, hadm_id, stay_id, itemid, value, starttime, endtime
        from mimiciv_icu.procedureevents
        where value is not null
        and itemid in (225792, 225794)
        order by subject_id, itemid, starttime
        """

    MV_icu = pd.read_sql_query(query,engine)

    def operation(target):

        target_value = MV_icu[MV_icu['stay_id'] == target['stay_id'].iloc[0]].reset_index(drop=True)

        if ~target_value.empty :
            for i in range(len(target_value)):
                target.loc[(target['charttime'] >= target_value['starttime'].iloc[i]) & (target['charttime'] < target_value['endtime'].iloc[i]),'MV'] = 1
        
        return target

    df['MV'] = 0
    df = df.groupby('stay_id',group_keys=False).parallel_apply(operation).reset_index(drop=True)

    return df

def Fluid_Mapping(df,ICU,pre_adm,value):

    data = ICU[['subject_id','starttime','endtime','tev','rate',value]]

    df['input_total'] = np.nan   #total fluid given
    df['input_6hr'] = np.nan  #fluid given at this step
    
    def operation(target):

        from datetime import timedelta
        import numpy as np

        target = target.reset_index(drop=True)
        subject_id = target['subject_id'].iloc[0]
        target_data = data[data['subject_id'] == subject_id]
        pread = pre_adm[pre_adm['stay_id'] == target['stay_id'].iloc[0]]

        if len(pread) > 0:           
            totvol = np.nansum(pread['inputpreadm'])
        else:
            totvol = np.nan

        for i in range(len(target)):
            
            target_lower = target['charttime'].iloc[i]
            target_upper = target_lower + timedelta(hours=6)

            case_1_value, case_2_value, case_3_value, case_4_value = 0,0,0,0

            case_1 = target_data[(target_data['starttime'] >= target_lower) & (target_data['endtime'] <= target_upper)]
            if not case_1.empty:  case_1_value = np.nansum((case_1[value] * (target_data['endtime']-target_data['starttime'])).dt.total_seconds() / 3600)

            case_2 = target_data[(target_data['starttime'] <= target_lower) & (target_data['endtime'] >= target_lower) & (target_data['endtime'] <= target_upper)]
            if not case_2.empty:  case_2_value = np.nansum((case_2[value] * (target_data['endtime']-target_lower)).dt.total_seconds() / 3600)

            case_3 = target_data[(target_data['starttime'] >= target_lower) & (target_data['starttime'] <= target_upper) & (target_data['endtime'] >= target_upper)]
            if not case_3.empty:  case_3_value = np.nansum((case_3[value] * (target_upper-target_data['starttime']).dt.total_seconds() / 3600))

            case_4 = target_data[(target_data['starttime'] <= target_lower) & (target_data['endtime'] >= target_upper)]
            if not case_4.empty:  case_4_value = np.nansum((case_4[value] * (target_upper-target_lower)).dt.total_seconds() / 3600)

            infu = np.nansum([case_1_value,case_2_value,case_3_value,case_4_value])
            bolus = np.nansum(target_data[(np.isnan(target_data['rate'])) & (target_data['starttime'] >= target_lower) & (target_data['starttime'] <= target_upper)]['tev'])

            totvol = np.nansum([totvol, infu, bolus])
            target.loc[i,'input_total'] = totvol    #total fluid given
            target.loc[i,'input_6hr'] = np.nansum([infu, bolus])  #fluid given at this step
            target.loc[i,'input_6hr_bolus'] = np.nansum([bolus])  #fluid given at this step only bolus

        return target
    
    df = df.groupby('subject_id',group_keys=False).parallel_apply(operation).reset_index(drop=True)

    df.loc[df['input_total'] < 0, 'input_total'] = 0
    df.loc[df['input_total'].isna(),'input_total'] = 0
    
    df.loc[df['input_6hr'] < 0, 'input_6hr'] = 0
    df.loc[df['input_6hr'].isna(),'input_6hr'] = 0

    df.loc[df['input_6hr_bolus'] < 0, 'input_6hr_bolus'] = 0
    df.loc[df['input_6hr_bolus'].isna(),'input_6hr_bolus'] = 0

    del data

    return df

def Vaso_Mapping(df,ICU,label):

    data = ICU[['subject_id','itemid','starttime','endtime','rate_std']]

    def operation(target):

        from datetime import timedelta
        import numpy as np

        target = target.reset_index(drop=True)
        subject_id = target['subject_id'].iloc[0]
        target_data = data[data['subject_id'] == subject_id]
    
        target[f'max_{label}'] = 0
        target[f'median_{label}'] = 0

        for i in range(len(target)):
            
            target_lower = target['charttime'].iloc[i]
            target_upper = target_lower + timedelta(hours=6)

            #v = ((endv >= t0) & (endv <= t1)) | ((startv >= t0) & (endv<=t1)) | ((startv >= t0) & (startv <= t1))| ((startv <= t0) & (endv>=t1))

            # VASOPRESSORS
            # for MV: 4 possibles cases, each one needing a different way to compute the dose of VP actually administered:
            #----t0---start----end-----t1----
            #----start---t0----end----t1----
            #-----t0---start---t1---end
            #----start---t0----t1---end----

            target_value = target_data[((target_data['endtime'] <= target_upper) & (target_data['endtime'] >= target_lower)) | 
                                       ((target_data['endtime'] <= target_upper) & (target_data['starttime'] >= target_lower)) |
                                       ((target_data['starttime'] <= target_upper) & (target_data['starttime'] >= target_lower)) |
                                       ((target_data['endtime'] >= target_upper) & (target_data['starttime'] <= target_lower))].sort_values(['starttime'])
            
            if not target_value.empty: 

                max_val = np.nanmax(target_value['rate_std'])
                median_val = np.nanmedian(target_value['rate_std'])

                target.loc[i, f'max_{label}'] = max_val
                target.loc[i, f'median_{label}'] = median_val

        return target
    
    df = df.groupby('subject_id',group_keys=False).parallel_apply(operation).reset_index(drop=True)

    del data

    return df

def split_and_prepare(target_subject_id, target_stay_id, stage, stratify_cols):

    def iterative_split(df, test_size, stratify_columns):
        one_hot_cols = pd.get_dummies(df[stratify_columns], columns=stratify_columns)
        stratifier = IterativeStratification(
            n_splits=2,
            order=len(stratify_columns),
            sample_distribution_per_fold=[test_size, 1 - test_size]
        )
        train_idx, test_idx = next(stratifier.split(df.values, one_hot_cols.values))
        return df.iloc[train_idx], df.iloc[test_idx]

    def sort_by_length(df):
        length_col = 'length_x' if 'length_x' in df.columns else 'length'
        return df.sort_values(by=length_col).reset_index(drop=True)

    def make_stay_split(subject_df, target_stay_df):
        return pd.merge(target_stay_df, subject_df[['subject_id']], on='subject_id') \
                 .sort_values('length').reset_index(drop=True)

    def make_stage_split(stage_df, stay_df):
        merged = pd.merge(stage_df, stay_df[['stay_id']], on='stay_id')
        merged['stay_id'] = pd.Categorical(merged['stay_id'], categories=stay_df['stay_id'], ordered=True)
        return merged.sort_values(['stay_id', 'charttime']).reset_index(drop=True)

    train_sub, test_sub = iterative_split(target_subject_id, 0.1, stratify_cols)
    train_sub, valid_sub = iterative_split(train_sub, 1/9, stratify_cols)
    valid_sub, calib_sub = iterative_split(valid_sub, 0.5, stratify_cols)

    splits_subject = {
        'train': sort_by_length(pd.merge(train_sub, target_subject_id[['subject_id', 'length']], on='subject_id')),
        'valid': sort_by_length(pd.merge(valid_sub, target_subject_id[['subject_id', 'length']], on='subject_id')),
        'calibration': sort_by_length(pd.merge(calib_sub, target_subject_id[['subject_id', 'length']], on='subject_id')),
        'test': sort_by_length(pd.merge(test_sub, target_subject_id[['subject_id', 'length']], on='subject_id')),
    }

    splits_stay = {
        k: make_stay_split(v, target_stay_id)
        for k, v in splits_subject.items()
    }

    splits_stage = {
        k: make_stage_split(stage, v)
        for k, v in splits_stay.items()
    }

    return splits_stay, splits_stage

def Dataset(df,numeric_features,presence_features,GT_presence,GT_stage):
    
    X_numerics = []
    X_presences = []
    Y_mains = []
    Y_subs = []
    masks = []

    start = int(min(df['length']))
    end = int(max(df['length']))
    datasets = []

    for i in tqdm(range(start,end+1)):
        
        target = df[df['length'] == i]

        if not target.empty:

            X_presence = target[presence_features]
            X_numeric = target[numeric_features]      

            Y_main = target[GT_presence]
            Y_sub = target[GT_stage]

            X_numeric = X_numeric.values  # Convert DataFrame to NumPy array
            X_numeric = torch.tensor(X_numeric.reshape(-1, i, X_numeric.shape[1]), dtype=torch.float32)
            padding = torch.zeros((X_numeric.shape[0],56-i,X_numeric.shape[2]),dtype=torch.float32)
            X_numeric = torch.cat((X_numeric,padding),dim=1)

            X_presence = X_presence.values  # Convert DataFrame to NumPy array
            X_presence = torch.tensor(X_presence.reshape(-1, i, X_presence.shape[1]), dtype=torch.float32)
            padding = torch.zeros((X_presence.shape[0],56-i,X_presence.shape[2]),dtype=torch.float32)
            X_presence = torch.cat((X_presence,padding),dim=1)

            Y_main = Y_main.values  # Convert DataFrame to NumPy array
            Y_main = torch.tensor(Y_main.reshape(-1, i, Y_main.shape[1]), dtype=torch.float32)
            padding = torch.zeros((Y_main.shape[0],56-i,Y_main.shape[2]),dtype=torch.float32)
            Y_main = torch.cat((Y_main,padding),dim=1)
            Y_main = Y_main.transpose(1, 2)

            Y_sub = Y_sub.values  # Convert DataFrame to NumPy array
            Y_sub = torch.tensor(Y_sub.reshape(-1, i, Y_sub.shape[1]), dtype=torch.float32)
            padding = torch.zeros((Y_sub.shape[0],56-i,Y_sub.shape[2]),dtype=torch.float32)
            Y_sub = torch.cat((Y_sub,padding),dim=1)
            Y_sub = Y_sub.transpose(1, 2)

            mask_valid = torch.zeros((Y_sub.shape[0],i),dtype=torch.float32)
            mask_ones = torch.ones((Y_sub.shape[0],56-i),dtype=torch.float32)
            mask = torch.cat((mask_valid,mask_ones),dim=1)

            X_numerics.append(X_numeric)
            X_presences.append(X_presence)
            Y_mains.append(Y_main)
            Y_subs.append(Y_sub)
            masks.append(mask)

    X_numeric = torch.cat(X_numerics,dim=0)
    X_presence = torch.cat(X_presences,dim=0)
    Y_main = torch.cat(Y_mains,dim=0)
    Y_sub = torch.cat(Y_subs,dim=0)
    mask = torch.cat(masks,dim=0)

    dataset = TensorDataset(X_numeric, X_presence, Y_main, Y_sub, mask) #Y_sub, mask)
    dataloader = DataLoader(dataset, batch_size = X_numeric.shape[0], shuffle=False, drop_last=False)

    for X_numeric, X_presence, Y_main, Y_sub, mask in dataloader:
        batch_dataset = TensorDataset(X_numeric, X_presence, Y_main, Y_sub, mask) #Y_sub, mask)
        datasets.append(batch_dataset)

    return datasets    

def Dataset_test(df,numeric_features,presence_features,GT_presence,GT_stage):
        
    start = int(min(df['length']))
    end = int(max(df['length']))
    datasets = []
    data = []

    for i in tqdm(range(start,end+1)):

        target = df[df['length'] == i]
        data.append(target)
        
        if not target.empty:

            X_presence = target[presence_features]
            X_numeric = target[numeric_features] 
            
            Y_main  = target[GT_presence]
            Y_sub  = target[GT_stage]

            X_numeric = X_numeric.values  # Convert DataFrame to NumPy array
            X_numeric = torch.tensor(X_numeric.reshape(-1, i, X_numeric.shape[1]), dtype=torch.float32)
            padding = torch.zeros((X_numeric.shape[0],56-i,X_numeric.shape[2]),dtype=torch.float32)
            X_numeric = torch.cat((X_numeric,padding),dim=1)

            X_presence = X_presence.values  # Convert DataFrame to NumPy array
            X_presence = torch.tensor(X_presence.reshape(-1, i, X_presence.shape[1]), dtype=torch.float32)
            padding = torch.zeros((X_presence.shape[0],56-i,X_presence.shape[2]),dtype=torch.float32)
            X_presence = torch.cat((X_presence,padding),dim=1)

            Y_main = Y_main.values  # Convert DataFrame to NumPy array
            Y_main = torch.tensor(Y_main.reshape(-1, i, Y_main.shape[1]), dtype=torch.float32)
            padding = torch.zeros((Y_main.shape[0],56-i,Y_main.shape[2]),dtype=torch.float32)
            Y_main = torch.cat((Y_main,padding),dim=1)
            Y_main = Y_main.transpose(1, 2)

            Y_sub = Y_sub.values  # Convert DataFrame to NumPy array
            Y_sub = torch.tensor(Y_sub.reshape(-1, i, Y_sub.shape[1]), dtype=torch.float32)
            padding = torch.zeros((Y_sub.shape[0],56-i,Y_sub.shape[2]),dtype=torch.float32)
            Y_sub = torch.cat((Y_sub,padding),dim=1)
            Y_sub = Y_sub.transpose(1, 2)

            mask_valid = torch.zeros((Y_sub.shape[0],i),dtype=torch.float32)
            mask_ones = torch.ones((Y_sub.shape[0],56-i),dtype=torch.float32)
            mask = torch.cat((mask_valid,mask_ones),dim=1)

            dataset = TensorDataset(X_numeric, X_presence, Y_main, Y_sub, mask)
            dataloader = DataLoader(dataset, batch_size = X_numeric.shape[0], shuffle=False, drop_last=False)

            for X_numeric, X_presence, Y_main, Y_sub, mask in dataloader:
                batch_dataset = TensorDataset(X_numeric, X_presence, Y_main, Y_sub, mask)
                datasets.append(batch_dataset)

    data = pd.concat(data)

    return datasets

def Dataset_CEP(df,numeric_features,presence_features):
    
    X_numerics = []
    X_presences = []
    Y_mains = []
    Y_subs = []
    masks = []

    start = 1
    end = 56 + 1
    datasets = []

    for i in tqdm(range(start,end)):
        
        target = df[df['length'] == i]

        if not target.empty:

            X_presence = target[presence_features]
            X_numeric = target[numeric_features]      

            Y_main  = target[['GT_presence_6','GT_presence_12','GT_presence_18','GT_presence_24','GT_presence_30','GT_presence_36','GT_presence_42','GT_presence_48']]
            Y_sub  = target[['GT_stage_3D','GT_stage_3','GT_stage_2','GT_stage_1']]

            X_numeric = X_numeric.values  # Convert DataFrame to NumPy array
            X_numeric = np.round(X_numeric, decimals=1)
            X_numeric = torch.tensor(X_numeric.reshape(-1, i, X_numeric.shape[1]), dtype=torch.float32)
            padding = torch.zeros((X_numeric.shape[0],56-i,X_numeric.shape[2]),dtype=torch.float32)
            X_numeric = torch.cat((X_numeric,padding),dim=1)

            X_presence = X_presence.values  # Convert DataFrame to NumPy array
            X_presence = torch.tensor(X_presence.reshape(-1, i, X_presence.shape[1]), dtype=torch.float32)
            padding = torch.zeros((X_presence.shape[0],56-i,X_presence.shape[2]),dtype=torch.float32)
            X_presence = torch.cat((X_presence,padding),dim=1)

            Y_main = Y_main.values  # Convert DataFrame to NumPy array
            Y_main = torch.tensor(Y_main.reshape(-1, i, Y_main.shape[1]), dtype=torch.float32)
            padding = torch.zeros((Y_main.shape[0],56-i,Y_main.shape[2]),dtype=torch.float32)
            Y_main = torch.cat((Y_main,padding),dim=1)
            Y_main = Y_main.transpose(1, 2)

            Y_sub = Y_sub.values  # Convert DataFrame to NumPy array
            Y_sub = torch.tensor(Y_sub.reshape(-1, i, Y_sub.shape[1]), dtype=torch.float32)
            padding = torch.zeros((Y_sub.shape[0],56-i,Y_sub.shape[2]),dtype=torch.float32)
            Y_sub = torch.cat((Y_sub,padding),dim=1)
            Y_sub = Y_sub.transpose(1, 2)

            mask_valid = torch.zeros((Y_sub.shape[0],i),dtype=torch.float32)
            mask_ones = torch.ones(Y_sub.shape[0],56-i,dtype=torch.float32)
            mask = torch.cat((mask_valid,mask_ones),dim=1)

            X_numerics.append(X_numeric)
            X_presences.append(X_presence)
            Y_mains.append(Y_main)
            Y_subs.append(Y_sub)
            masks.append(mask)

    X_numeric = torch.cat(X_numerics,dim=0)
    X_presence = torch.cat(X_presences,dim=0)
    Y_main = torch.cat(Y_mains,dim=0)
    Y_sub = torch.cat(Y_subs,dim=0)
    mask = torch.cat(masks,dim=0)

    dataset = TensorDataset(X_numeric, X_presence, Y_main, Y_sub, mask) #Y_sub, mask)
    dataloader = DataLoader(dataset, batch_size = X_numeric.shape[0], shuffle=False, drop_last=False)

    for X_numeric, X_presence, Y_main, Y_sub, mask in dataloader:
        batch_dataset = TensorDataset(X_numeric, X_presence, Y_main, Y_sub, mask) #Y_sub, mask)
        datasets.append(batch_dataset)

    return datasets


def compute_pos_weights_presence(df, prefix='GT_presence_', suffix=''):
    hours = [6, 12, 18, 24, 30, 36, 42, 48]
    weights = [
        (df[f"{prefix}{h}{suffix}"] == 0).sum() / (df[f"{prefix}{h}{suffix}"] == 1).sum()
        for h in hours
    ]
    return torch.tensor(weights, dtype=torch.float32)

def compute_pos_weights_stage(df, stage_cols, rrt_weight=None):
    stage_weights = [
        (df[col] == 0).sum() / (df[col] == 1).sum()
        for col in stage_cols
    ]
    if rrt_weight is not None:
        return torch.tensor([rrt_weight] + stage_weights, dtype=torch.float32)
    else:
        return torch.tensor(stage_weights, dtype=torch.float32)

def compute_rrt_pos_weight(id_df, rrt_col='RRT'):
    num_pos = (id_df[rrt_col] == 1).sum()
    num_neg = len(id_df) - num_pos
    return num_neg / num_pos

def train(
    model,
    train_dataloader,
    valid_dataloader,
    pos_weights,
    batchsize: int,
    learning_rate: float,
    num_epochs: int,
    lr_decay_factor: float,
    lr_decay_steps: int,
    LD: bool,
    CDF: bool,
    path: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    def cdf(t: torch.Tensor) -> torch.Tensor:
        return torch.cummax(t, dim=1)[0]

    LossClass = CustomBCELoss if CDF else nn.BCEWithLogitsLoss

    criterion_main_train = [LossClass(pos_weight=w) for w in pos_weights[0]]
    criterion_sub_train  = [LossClass(pos_weight=w) for w in pos_weights[1]]
    criterion_main_valid = [LossClass(pos_weight=w) for w in pos_weights[2]]
    criterion_sub_valid  = [LossClass(pos_weight=w) for w in pos_weights[3]]

    losses = {"valid_loss": 0, "main_loss": 0, "sub_loss": 0}
    early_stopping = EarlyStopping(
        patience=lr_decay_steps,
        path=path,
        loss_names=list(losses.keys()),
        verbose=False,
    )

    optimizer = PCGrad(optim.Adam(model.parameters(), lr=learning_rate))
    scheduler = ExponentialLR(optimizer.optimizer, gamma=lr_decay_factor)

    step = 0
    model.to(device)

    for epoch in range(num_epochs):
        train_loader = DataLoader(
            train_dataloader.dataset[0],
            batch_size=batchsize,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=4,
        )

        for batch in train_loader:
            step += 1
            model.train()

            inputs_numeric, inputs_presence, tgt_main, tgt_sub, mask = [x.to(device) for x in batch]
            out_main, out_sub = model(inputs_numeric, inputs_presence)

            if CDF:
                out_main = cdf(torch.sigmoid(out_main))
                out_sub  = cdf(torch.sigmoid(out_sub))

            loss_main = sum(c(out_main[:, j], tgt_main[:, j]) for j, c in enumerate(criterion_main_train))
            loss_sub  = sum(c(out_sub[:,  j], tgt_sub[:,  j]) for j, c in enumerate(criterion_sub_train))
            loss_total = loss_main + loss_sub

            optimizer.zero_grad()
            optimizer.pc_backward([loss_main, loss_sub])
            optimizer.step()

            model.eval()
            with torch.no_grad():
                v_loss_main = 0.0
                v_loss_sub  = 0.0

                for v_batch in valid_dataloader.dataset:
                    v_inputs_numeric, v_inputs_presence, v_tgt_main, v_tgt_sub, _ = [
                        x.to(device) for x in v_batch.tensors
                    ]
                    v_out_main, v_out_sub = model(v_inputs_numeric, v_inputs_presence)

                    if CDF:
                        v_out_main = cdf(torch.sigmoid(v_out_main))
                        v_out_sub  = cdf(torch.sigmoid(v_out_sub))

                    v_loss_main += sum(c(v_out_main[:, j], v_tgt_main[:, j]) for j, c in enumerate(criterion_main_valid))
                    v_loss_sub  += sum(c(v_out_sub[:,  j], v_tgt_sub[:,  j]) for j, c in enumerate(criterion_sub_valid))

                v_loss_total = v_loss_main + v_loss_sub

                losses.update(
                    valid_loss=v_loss_total.item(),
                    main_loss=v_loss_main.item(),
                    sub_loss=v_loss_sub.item(),
                )
                early_stopping(
                    losses, model, epoch, num_epochs,
                    loss_total.item(), loss_main.item(), loss_sub.item(),
                    v_loss_total.item(), v_loss_main.item(), v_loss_sub.item()
                )

                gc.collect()
                torch.cuda.empty_cache()

                if early_stopping.early_stop:
                    model.load_state_dict(torch.load(path))
                    break

                if LD and (step % lr_decay_steps == 0):
                    scheduler.step()

            if early_stopping.early_stop:
                break
        if early_stopping.early_stop:
            break

    return model, losses["valid_loss"]

def objective(trial, train_dataloader, valid_dataloader, pos_weights, device):

    numeric_input_size   = train_dataloader.dataset[0].tensors[0].shape[-1]
    presence_input_size  = train_dataloader.dataset[0].tensors[1].shape[-1]

    seq_len   = 56
    num_epochs= 1_000_000

    hidden_size          = trial.suggest_int("hidden_size",        50, 200, step=50)
    embedding_size       = trial.suggest_int("embedding_size",     25, 100, step=25)
    recurrent_num_layers = trial.suggest_int("recurrent_num_layers", 1, 5)
    embedding_num_layers = trial.suggest_int("embedding_num_layers", 1, 5)

    CB              = trial.suggest_categorical("CB",  [0, 1])      # 0: sum, 1: concat
    recurrent_type  = trial.suggest_categorical("recurrent_type",
                                                ["LSTM", "RNN", "GRU"])
    activation_type = trial.suggest_categorical("activation_type",
                                                ["ReLU", "LeakyReLU", "Tanh",
                                                 "ELU", "SELU", "CELU", "GELU"])

    batchsize       = trial.suggest_categorical("batchsize",      [64, 128, 256, 512])
    learning_rate   = trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2])
    lr_decay_steps  = trial.suggest_categorical("lr_decay_steps", [800, 400, 200, 100])
    lr_decay_factor = trial.suggest_categorical("lr_decay_factor",
                                                [0.7, 0.8, 0.85, 0.9, 0.95])

    HN  = bool(trial.suggest_categorical("highway_network", [0, 1]))
    LD  = bool(trial.suggest_categorical("LD",              [0, 1]))
    LN  = bool(trial.suggest_categorical("LN",              [0, 1]))
    CDF = bool(trial.suggest_categorical("CDF",             [0, 1]))

    model = AKIPredictionModel(
        hidden_size, embedding_size,
        recurrent_num_layers, embedding_num_layers,
        activation_type, recurrent_type, seq_len,
        LN, HN,
        numeric_input_size, presence_input_size, CB
    ).to(device)

    ckpt_path = os.path.join("model", f"trial_{trial.number+1}_model.pt")
    os.makedirs("model", exist_ok=True)

    _, valid_loss = train(
        model,
        train_dataloader,
        valid_dataloader,
        pos_weights,
        batchsize           = batchsize,
        learning_rate       = learning_rate,
        num_epochs          = num_epochs,
        lr_decay_factor     = lr_decay_factor,
        lr_decay_steps      = lr_decay_steps,
        LD                  = LD,
        CDF                 = CDF,
        path                = ckpt_path
    )

    return valid_loss

def test(model,dataloader):

    model.eval()

    criterion =  [nn.BCELoss()]

    main_datasets_6h = []
    main_datasets_12h = []
    main_datasets_18h = []
    main_datasets_24h = []
    main_datasets_30h = []
    main_datasets_36h = []
    main_datasets_42h = []
    main_datasets_48h = []

    sub_datasets_1 = []
    sub_datasets_2 = []
    sub_datasets_3 = []
    sub_datasets_3D = []

    with torch.no_grad():

        test_loss = 0.0
      
        for data in dataloader.dataset:
                
            inputs_numeric, inputs_presence, targets_main, targets_sub, mask = [d.to(device) for d in data.tensors]
            
            out_main, out_sub = model(inputs_numeric, inputs_presence)

            test_main_loss = 0.0
            test_sub_loss = 0.0  

            out_main = F.sigmoid(out_main)
            out_sub = F.sigmoid(out_sub)    

            out_main = cdf(out_main)
            out_sub = cdf(out_sub)
            
            for i in range(mask.shape[0]):

                length = (mask[i,:] == 0).sum()
                test_main_loss += sum(criterion(out_main[i, j, :length], targets_main[i, j, :length]) for j, criterion in enumerate(criterion))
                test_sub_loss += sum(criterion(out_sub[i, j, :length], targets_sub[i, j, :length]) for j, criterion in enumerate(criterion))

            test_loss += (test_main_loss + test_sub_loss).item()

            i = (mask[0,:] == 0).sum()

            dataset = TensorDataset(out_main[:,0,:i],out_main[:,1,:i],out_main[:,2,:i],out_main[:,3,:i],out_main[:,4,:i],out_main[:,5,:i],out_main[:,6,:i],out_main[:,7,:i],
                                    out_sub[:,3,:i],out_sub[:,2,:i],out_sub[:,1,:i],out_sub[:,0,:i],
                                    targets_main[:,0,:i],targets_main[:,1,:i],targets_main[:,2,:i],targets_main[:,3,:i],targets_main[:,4,:i],targets_main[:,5,:i],targets_main[:,6,:i],targets_main[:,7,:i],
                                    targets_sub[:,3,:i],targets_sub[:,2,:i],targets_sub[:,1,:i],targets_sub[:,0,:i])
                                    
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

            for out_main_6h,out_main_12h,out_main_18h,out_main_24h,out_main_30h,out_main_36h,out_main_42h,out_main_48h,out_sub_1,out_sub_2,out_sub_3,out_sub_3D,targets_main_6h,targets_main_12h,targets_main_18h,targets_main_24h,targets_main_30h,targets_main_36h,targets_main_42h,targets_main_48h,targets_sub_1,targets_sub_2,targets_sub_3,targets_sub_3D in dataloader:
                
                main_dataset_6h = TensorDataset(out_main_6h, targets_main_6h)
                main_dataset_12h = TensorDataset(out_main_12h, targets_main_12h)
                main_dataset_18h = TensorDataset(out_main_18h, targets_main_18h)
                main_dataset_24h = TensorDataset(out_main_24h, targets_main_24h)
                main_dataset_30h = TensorDataset(out_main_30h, targets_main_30h)
                main_dataset_36h = TensorDataset(out_main_36h, targets_main_36h)
                main_dataset_42h = TensorDataset(out_main_42h, targets_main_42h)
                main_dataset_48h = TensorDataset(out_main_48h, targets_main_48h)

                main_datasets_6h.append(main_dataset_6h)
                main_datasets_12h.append(main_dataset_12h)
                main_datasets_18h.append(main_dataset_18h)
                main_datasets_24h.append(main_dataset_24h)
                main_datasets_30h.append(main_dataset_30h)
                main_datasets_36h.append(main_dataset_36h)
                main_datasets_42h.append(main_dataset_42h)
                main_datasets_48h.append(main_dataset_48h)

                sub_dataset_1 = TensorDataset(out_sub_1, targets_sub_1)
                sub_dataset_2 = TensorDataset(out_sub_2, targets_sub_2)
                sub_dataset_3 = TensorDataset(out_sub_3, targets_sub_3)
                sub_dataset_3D = TensorDataset(out_sub_3D, targets_sub_3D)

                sub_datasets_1.append(sub_dataset_1)
                sub_datasets_2.append(sub_dataset_2)
                sub_datasets_3.append(sub_dataset_3)
                sub_datasets_3D.append(sub_dataset_3D)               

    print(f"Test Loss: {test_loss:.4f}")

    main_datasets = [main_datasets_6h, main_datasets_12h, main_datasets_18h, main_datasets_24h,main_datasets_30h, main_datasets_36h, main_datasets_42h, main_datasets_48h]
    sub_datasets = [sub_datasets_1, sub_datasets_2, sub_datasets_3, sub_datasets_3D]
    
    return main_datasets, sub_datasets

def reshape(dataloader, prob_pos, y_true):

    index = 0
    datasets = []

    prob_pos_tensor = torch.tensor(prob_pos)
    y_true_tensor = torch.tensor(y_true)
    
    for batch in dataloader.dataset:
        
        X, Y = batch.tensors
        X = X.cpu().detach().numpy().tolist()
        Y = Y.cpu().detach().numpy().tolist()

        score = prob_pos_tensor[(index):(index+len(X[0]))]
        true = y_true_tensor[(index):(index+len(Y[0]))]
        
        dataset = TensorDataset(score.unsqueeze(0), true.unsqueeze(0))
        datasets.append(dataset)

        index += len(X[0])

    dataloaders = DataLoader(datasets, batch_size=1, shuffle=False, drop_last=True)

    return dataloaders

def calibration(calibration_dataloader, dataloader):
    y_true, y_scores = step_ROC(calibration_dataloader)
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_scores, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibration Curve")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.show()

    ir = IsotonicRegression(out_of_bounds='clip').fit(y_scores, y_true)

    y_true, prob_pos = step_ROC(dataloader)
    brier_score_before = brier_score_loss(y_true, prob_pos)
    prob_pos_calibrated = ir.transform(prob_pos)
    brier_score_after = brier_score_loss(y_true, prob_pos_calibrated)

    print("Brier Score Before:", round(brier_score_before, 4))
    print("Brier Score After:", round(brier_score_after, 4))

    if brier_score_after > brier_score_before:
        print('[!] Calibration degraded performance. Reverting to original.')
        return dataloader

    print('[✓] Calibration improved performance.')
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, prob_pos_calibrated, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibration Curve")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.show()

    calibrated_dataloader = reshape(dataloader, prob_pos_calibrated, y_true)

    return calibrated_dataloader

def step_ROC(dataloader):

    y_true = []
    y_scores = []
    
    for batch in dataloader.dataset:
        
        X, Y = batch.tensors

        X = X.cpu().detach().numpy().tolist()
        y_scores.extend(X[0])

        Y = Y.cpu().detach().numpy().tolist()
        y_true.extend(Y[0])

    return y_true, y_scores

def AKI_step_ROC(dataloader):

    sum = 0
    y_true = []
    y_scores = []
    
    for batch in dataloader.dataset:
        X, Y = batch.tensors
        if torch.sum(Y) > 0:
            Y = Y.cpu().detach().numpy().tolist()
            y_true.extend(Y[0])

            X = X.cpu().detach().numpy().tolist()
            y_scores.extend(X[0])

            sum += 1

    return y_true, y_scores

def AKI_first_ROC(dataloader):

    sum = 0
    y_true = []
    y_scores = []
    values = float(1)

    for batch in dataloader.dataset:
        X, Y = batch.tensors
        if torch.sum(Y) > 0:
            Y = Y.cpu().detach().numpy().tolist()
            index = Y[0].index(values)
            y_true.extend(Y[0][:index+1])

            X = X.cpu().detach().numpy().tolist()
            y_scores.extend(X[0][:index+1])

            sum+= 1
    
    return y_true, y_scores

def AUROC(y_true,y_scores):

    auroc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    return fpr, tpr, thresholds, auroc

def AUPRC(y_true,y_scores):

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)

    return recall, precision, thresholds, auprc

def calculate_confusion_matrix(y_true, y_scores, threshold):

    y_pred = [1 if score >= threshold else 0 for score in y_scores]

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == 1 and pred_label == 1:
            TP += 1
        elif true_label == 0 and pred_label == 1:
            FP += 1
        elif true_label == 0 and pred_label == 0:
            TN += 1
        elif true_label == 1 and pred_label == 0:
            FN += 1

    evaluation(TP,FP,TN,FN)
    
    return TP, FP, TN, FN

def evaluation(TP,FP,TN,FN):
    Sensitivity = TP / (TP+FN) # Recall
    Specitivity = TN / (FP+TN) 
    Accuaracy = (TP+TN) / (TP+TN+FP+FN)
    Precision = TP / (TP+FP)
    F1 = (2 * Precision * Sensitivity) / (Precision+Sensitivity)

    print('Accuracy :',round(Accuaracy,3)*100,'%')
    print('Precision :',round(Precision,3)*100,'%')
    print('Sensitivity :',round(Sensitivity,3)*100,'%')
    print('Specitivity :',round(Specitivity,3)*100,'%')
    print('F1 score :',round(F1,3))

def bootstrap_auroc(y_true, y_pred, n_bootstraps=200):
    bootstrapped_scores = []
    rng = np.random.RandomState(42)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for _ in range(n_bootstraps):

        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return confidence_lower, confidence_upper

def bootstrap_auprc(y_true, y_pred, n_bootstraps=200):

    bootstrapped_scores = []
    rng = np.random.RandomState(42)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for _ in range(n_bootstraps):

        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue

        score = average_precision_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return confidence_lower, confidence_upper

def Result(dataloader):

    y_true, y_scores = step_ROC(dataloader)

    fpr, tpr, AUROC_thresholds, auroc = AUROC(y_true, y_scores)
    ci_lower, ci_upper = bootstrap_auroc(y_true, y_scores)
    print(f"{round(auroc*100,1)} ({round(ci_lower*100,1)}-{round(ci_upper*100,1)})")

    recall, precision, AUPRC_thresholds, auprc = AUPRC(y_true, y_scores)
    ci_lower, ci_upper = bootstrap_auprc(y_true, y_scores)
    print(f"{round(auprc*100,1)} ({round(ci_lower*100,1)}-{round(ci_upper*100,1)})")
    
    y_true, y_scores = AKI_step_ROC(dataloader)

    fpr, tpr, AUROC_thresholds, auroc = AUROC(y_true, y_scores)
    ci_lower, ci_upper = bootstrap_auroc(y_true, y_scores)
    print(f"{round(auroc*100,1)} ({round(ci_lower*100,1)}-{round(ci_upper*100,1)})")

    recall, precision, AUPRC_thresholds, auprc = AUPRC(y_true, y_scores)
    ci_lower, ci_upper = bootstrap_auprc(y_true, y_scores)
    print(f"{round(auprc*100,1)} ({round(ci_lower*100,1)}-{round(ci_upper*100,1)})")

    y_true, y_scores = AKI_first_ROC(dataloader)

    fpr, tpr, AUROC_thresholds, auroc = AUROC(y_true, y_scores)
    ci_lower, ci_upper = bootstrap_auroc(y_true, y_scores)
    print(f"{round(auroc*100,1)} ({round(ci_lower*100,1)}-{round(ci_upper*100,1)})")

    recall, precision, AUPRC_thresholds, auprc = AUPRC(y_true, y_scores)
    ci_lower, ci_upper = bootstrap_auprc(y_true, y_scores)
=======
import os
import gc
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from pandarallel import pandarallel
from skmultilearn.model_selection import IterativeStratification

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR

from model import AKIPredictionModel, CustomBCELoss, PCGrad, EarlyStopping

tqdm.pandas()
pd.set_option('mode.chained_assignment',  None)
pandarallel.initialize(nb_workers=8,progress_bar=False)

def filter(icu, hosp, RRT_icu, RRT_hosp, KT_hosp):

    icu['RRT'] = 0

    def RRT_in_icu(target):

        target.reset_index(inplace=True,drop=True)

        RRT = RRT_icu[RRT_icu['stay_id'] == target['stay_id'].iloc[0]]

        if not RRT.empty:
            target['outtime'] = RRT['starttime'].iloc[0]
            target = target[target['charttime'] < target['outtime']]
            if not target.empty:
                target['RRT'].iloc[-1] = 1
        return target

    icu = icu.groupby('stay_id').parallel_apply(RRT_in_icu).reset_index(drop=True)

    icu['RRT_icu_history'] = icu.groupby('subject_id')['charttime'].transform(
    lambda x: (x > RRT_icu.loc[RRT_icu['subject_id'] == x.name, 'starttime'].min()).astype(int)
    )

    def RRT_in_hosp(target):

        target.reset_index(inplace=True,drop=True)

        from datetime import timedelta

        RRT = RRT_hosp[RRT_hosp['subject_id'] == target['subject_id'].iloc[0]]

        if not RRT.empty:
            for _, row in RRT.iterrows():
                end_criteria = row['chartdate'] + timedelta(days=1, hours=23, minutes=59)
                start_criteria = row['chartdate']
                target = target[~((target['charttime'] > start_criteria) & (target['charttime'] < end_criteria))]

        return target
    

    icu = icu.assign(RRT_hosp_history=icu.groupby('subject_id')['charttime'].transform(
    lambda x: (x > (RRT_hosp.loc[RRT_hosp['subject_id'] == x.name, 'chartdate'].min() + timedelta(days=1, hours=23, minutes=59))).astype(int)
    ))

    def KT_in_hosp(target):

        target.reset_index(inplace=True,drop=True)

        from datetime import timedelta

        KT = KT_hosp[KT_hosp['subject_id'] == target['subject_id'].iloc[0]]
    
        if not KT.empty:
            criteria = KT['chartdate'].min() + timedelta(days=0,hours=0,minutes=0)
            target = target[target['charttime'] < criteria]

        return target

    icu = icu.groupby('subject_id').parallel_apply(KT_in_hosp).reset_index(drop=True)

    if not hosp.empty:
        hosp = hosp.groupby('subject_id').parallel_apply(RRT_in_hosp).reset_index(drop=True)
        hosp = hosp.groupby('subject_id').parallel_apply(KT_in_hosp).reset_index(drop=True)

    return icu, hosp

def diff(icu, hosp):

    icu[['min', 'max', 'median', 'mean', 'diff']] = np.nan
    hosp[['min', 'max', 'median', 'mean', 'diff']] = np.nan
    SCr = pd.concat([icu, hosp]).dropna(subset=['SCr'])

    def operation(target):

        target.reset_index(inplace=True,drop=True)

        import numpy as np
        import pandas as pd
        from datetime import timedelta
        
        target_SCr = SCr[SCr['subject_id'] == target['subject_id'].iloc[0]]
        
        cri = target['charttime']
        cri_48 = cri - timedelta(hours=48)
        
        for i, charttime in enumerate(cri):
            value_SCr = target_SCr.loc[(target_SCr['charttime'] < charttime) & (target_SCr['charttime'] >= cri_48.iloc[i]), 'SCr']
            
            if not value_SCr.empty:
                
                if pd.isna(target.at[i, 'min']):
                    target.at[i, 'min'] = value_SCr.min()
                    if not pd.isna(target.at[i, 'SCr']):
                        target.at[i, 'diff'] = target.at[i, 'SCr'] - value_SCr.min()
            
                if pd.isna(target.at[i, 'max']):
                    target.at[i, 'max'] = value_SCr.max()

                if pd.isna(target.at[i, 'median']):
                    target.at[i, 'median'] = value_SCr.median()

                if pd.isna(target.at[i, 'mean']):
                    target.at[i, 'mean'] = value_SCr.mean()
            else:

                if pd.isna(target.at[i, 'min']):
                    if not pd.isna(target.at[i, 'SCr']):
                        target.at[i, 'min'] = target.at[i, 'SCr']
                        target.at[i, 'diff'] = 0

                if pd.isna(target.at[i, 'max']):
                    if not pd.isna(target.at[i, 'SCr']):
                        target.at[i, 'max'] = target.at[i, 'SCr']

                if pd.isna(target.at[i, 'median']):
                    if not pd.isna(target.at[i, 'SCr']):
                        target.at[i, 'median'] = target.at[i, 'SCr']

                if pd.isna(target.at[i, 'mean']):
                    if not pd.isna(target.at[i, 'SCr']):
                        target.at[i, 'mean'] = target.at[i, 'SCr']

        return target

    icu = icu.groupby('subject_id', group_keys=False).parallel_apply(operation).reset_index(drop=True)
    hosp = hosp.groupby('subject_id', group_keys=False).parallel_apply(operation).reset_index(drop=True)

    for column in ['min', 'max', 'median', 'mean', 'diff']:
        icu[column] = icu[column].round(1)
        hosp[column] = hosp[column].round(1)

    return icu, hosp

def SCr_gap(target):

    label = 'SCr'
    target[label] = target[label].round(1)
    target[label + '_diff'] = target[label].diff().round(1)

    label = 'charttime'
    target[label + '_diff'] = target[label].diff()
    target['SCr_' + label + '_diff'] = (target[label + '_diff'].dt.total_seconds() / 3600).round(1)
        
    return target

def Pre_admission(icu, hosp):

    common_columns = icu.columns.intersection(hosp.columns).tolist()
    SCr = pd.concat([icu[common_columns], hosp[common_columns]]).drop_duplicates().reset_index(drop=True)

    def MDRD(df):
            
        if df['gender'] == 'F' and df['race'] == 'BLACK': df['baseline'] = (75 / (0.742 * 1.21 * 186 * df['age'] ** (-0.203))) ** (-1 / 1.154)
        elif df['race'] == 'BLACK': df['baseline'] = (75 / (1 * 1.21 * 186 * df['age'] ** (-0.203))) ** (-1 / 1.154)
        elif df['gender'] == 'F': df['baseline'] = (75 / (0.742 * 1 * 186 * df['age'] ** (-0.203))) ** (-1 / 1.154)
        else: df['baseline'] = (75 / (1 * 1 * 186 * df['age'] ** (-0.203))) ** (-1 / 1.154)
    
        return round(df['baseline'], 1)

    def operation(target):

        target.reset_index(inplace=True,drop=True)

        from datetime import timedelta
        import numpy as np
        
        target_SCr = SCr[SCr['subject_id'] == target['subject_id'].iloc[0]]
        target_stay_id = target['stay_id'].drop_duplicates()

        if not target_SCr.empty :
            for i in target_stay_id:    
                intime = target.loc[target['stay_id'] == i, 'intime'].iloc[0]
                intime_7 = intime - timedelta(days=7)
                value_SCr = target_SCr.loc[(target_SCr['charttime'] < intime) & (target_SCr['charttime'] >= intime_7), 'SCr'] 

                if not value_SCr.empty : 
                    target.loc[target['stay_id'] == i, ['baseline', 'method']] = value_SCr.min(), 1
                else : 
                    intime_365 = intime - timedelta(days=365)
                    value_SCr = target_SCr.loc[(target_SCr['charttime'] <= intime_7) & (target_SCr['charttime'] >= intime_365), 'SCr']
                    if not value_SCr.empty : 
                        target.loc[target['stay_id'] == i, ['baseline', 'method']] = np.median(value_SCr), 2
                    else : 
                        target.loc[target['stay_id'] == i, ['baseline', 'method']] = MDRD(target.iloc[0]), 0
        else : 
            for i in target_stay_id:
                target.loc[target['stay_id'] == i, ['baseline', 'method']] = MDRD(target.iloc[0]), 0

        return target

    icu = icu.groupby('subject_id', group_keys=False).parallel_apply(operation).reset_index(drop=True)
    
    return icu

def SCr_AKI_stage(df):
    
    df['ratio'] = df['SCr'] / df['baseline']
    df['SCr_stage'] = 0

    condition_1 = ((df['ratio'] < 2) & (df['ratio'] >= 1.5)) | ((df['diff'] >= 0.3) & (df['diff'] < 4))
    condition_2 = (df['ratio'] < 3) & (df['ratio'] >= 2)
    condition_3 = (df['ratio'] >= 3) | (df['diff'] >= 4) | (df['RRT'] == 1)

    df.loc[condition_1, 'SCr_stage'] = 1
    df.loc[condition_2, 'SCr_stage'] = 2
    df.loc[condition_3, 'SCr_stage'] = 3

    df.loc[(df['SCr'].isnull()) & (df['RRT'] != 1), 'SCr_stage'] = 0

    return df

def SCr_resampling(df, label):

    def operation(group):
        
        import pandas as pd
        import numpy as np
        from datetime import timedelta

        start_frame = group.iloc[[0]].copy()
        end_frame = group.iloc[[0]].copy()

        start_frame['charttime'] = start_frame['intime']
        end_frame['charttime'] = end_frame['outtime']

        na_columns = [label, 'diff', 'min', 'max', 'median', 'mean', 'ratio', 'SCr_diff', 'SCr_stage', 'SCr_charttime_diff']
        start_frame[na_columns] = np.nan
        end_frame[na_columns] = np.nan

        group = pd.concat([start_frame, group, end_frame])
        group.set_index('charttime', inplace=True)

        resampled = group.resample('6h', origin=group['intime'].iloc[0], label='left').last()

        if pd.isna(resampled[label].iloc[-1]):
            last_contribute = group[(group.index < group['outtime']) & (group.index >= group['outtime'] - timedelta(hours=6))]
            if not last_contribute.empty:
                resampled.iloc[-1] = last_contribute.iloc[-1]

        if pd.isna(resampled[label].iloc[0]):
            first_contribute = group[(group.index > group['intime']) & (group.index <= group['intime'] + timedelta(hours=6))]
            if not first_contribute.empty:
                resampled.iloc[0] = first_contribute.iloc[-1]

        for col in ['subject_id', 'hadm_id', 'stay_id', 'first_careunit', 'intime', 'outtime', 'RRT_icu_history', 'RRT_hosp_history', 'los', 'race', 'age', 'gender', 'baseline', 'method']:
            resampled[col] = group[col].iat[0]

        resampled['charttime'] = resampled.index
        resampled['timedelta'] = resampled.index - group['intime'].iloc[0]

        resampled['RRT'] = 0
        if group['RRT'].sum() > 0:
            resampled['RRT'].iloc[-1] = 1
        else:
            resampled['RRT'] = resampled['RRT'].fillna(0)

        return resampled

    df = df.groupby('stay_id', group_keys=False).parallel_apply(operation).reset_index(drop=True)
    df['timedelta'] = pd.to_timedelta(df['timedelta'])
    df['SCr_presence'] = df['SCr'].notna().astype(int)

    return df

def SCr_copy_mask(df, df_icu, df_hosp, label):

    df_SCr = pd.concat([df_icu, df_hosp])
    columns_to_fill = ['min', 'max', 'mean', 'median', 'diff', 'ratio', 'SCr_stage', 'SCr_diff', 'SCr_charttime_diff']

    def SCr_copy(target):

        for col in [label] + columns_to_fill:
            target[col] = target[col].fillna(method='ffill', limit=4)
        return target

    def SCr_copy_hosp(target):

        target.reset_index(inplace=True,drop=True)

        from datetime import timedelta

        if target[label].isnull().iloc[0]:
            target_hosp = df_SCr.loc[df_SCr['subject_id'] == target['subject_id'].iloc[0]]
            forward = target_hosp[(target_hosp['charttime'] < target['charttime'].iloc[0]) & 
                                  (target_hosp['charttime'] > (target['charttime'].iloc[0] - timedelta(days=1)))]
            if not forward.empty:
                target_value = forward.iloc[-1]
                cri = target_value['charttime']
                target_2 = target[(target['charttime'] > cri) & (target['charttime'] < (cri + timedelta(days=1)))]
                if not target_2.empty:
                    for col in [label] + columns_to_fill:
                        target.loc[target_2.index, col] = target_value[col]
        return target

    def SCr_mask(target):

        target.reset_index(inplace=True,drop=True)

        columns = ['SCr', 'ratio', 'min', 'max', 'mean', 'median', 'diff', 'SCr_stage', 'SCr_diff', 'SCr_charttime_diff']
        for col in columns:
            target[f'{col}_mask'] = target[col].notna().astype(int)
            target[col] = target[col].fillna(0)
    
        return target

    df = df.groupby('stay_id', group_keys=False).parallel_apply(SCr_copy).reset_index(drop=True)
    df = df.groupby('stay_id', group_keys=False).parallel_apply(SCr_copy_hosp).reset_index(drop=True)
    df = SCr_mask(df)

    return df

def fetch_outputevents(itemids: str, engine):
    query = f"""
    SELECT subject_id, hadm_id, stay_id, charttime, itemid, value 
    FROM mimiciv_icu.outputevents 
    WHERE value IS NOT NULL
    AND itemid IN ({itemids})
    AND value > 0
    ORDER BY stay_id, charttime
    """
    return pd.read_sql_query(query, engine)


def preprocess_weight_data(df, chartevents, omr):
    
    Weight_icu_kg = chartevents[chartevents['itemid'].isin([224639, 226512])]
    Weight_icu_lbs = chartevents[chartevents['itemid'].isin([226531])]
    Weight_icu_lbs['valuenum'] = Weight_icu_lbs['valuenum'] / 2.20462  # lbs to kg
    Weight_icu = pd.concat([Weight_icu_kg, Weight_icu_lbs])
    Weight_icu = Weight_icu.query('27.2155 <= valuenum <= 317.515').copy()
    Weight_icu['valuenum'] = Weight_icu['valuenum'].round(1)
    Weight_icu = Weight_icu.rename(columns={'valuenum': 'Weight'})

    omr_weight_kg = omr[omr['result_name'] == 'Weight']
    omr_weight_lbs = omr[omr['result_name'] == 'Weight (Lbs)']

    omr_weight_kg['result_value'] = pd.to_numeric(omr_weight_kg['result_value'], errors='coerce')
    omr_weight_lbs['result_value'] = pd.to_numeric(omr_weight_lbs['result_value'], errors='coerce') / 2.20462  # lbs to kg

    omr_weight = pd.concat([omr_weight_kg, omr_weight_lbs])
    omr_weight = omr_weight.dropna(subset=['result_value'])
    omr_weight['chartdate'] = omr_weight['chartdate'].astype(str)
    omr_weight['charttime'] = pd.to_datetime(omr_weight['chartdate'] + ' 23:59:59')
    omr_weight['Weight'] = omr_weight['result_value'].round(1)
    omr_weight = omr_weight.query('27.2155 <= Weight <= 317.515').copy()
    omr_weight = omr_weight.sort_values(['subject_id', 'charttime']).drop_duplicates(subset=['subject_id', 'charttime'], keep='last')

    Weight_pool = pd.concat([omr_weight, Weight_icu]).sort_values(['charttime'])

    def Weight(target_icu):

        import pandas as pd

        target_Weight = Weight_pool[Weight_pool['subject_id'] == target_icu['subject_id'].iloc[0]]
        target_Weight = target_Weight.sort_values('charttime').reset_index(drop=True)
        target_icu = target_icu.sort_values('charttime').reset_index(drop=True)

        target_icu['Weight'] = pd.merge_asof(
            target_icu[['charttime']],
            target_Weight[['charttime', 'Weight']],
            on='charttime',
            direction='backward'
        )['Weight']

        return target_icu

    df = df.groupby('subject_id', group_keys=False).parallel_apply(Weight).reset_index(drop=True)
    
    return df

def Urine(df):

    df = df.sort_values(by=['stay_id', 'charttime'])
    df['charttime_diff'] = df.groupby('stay_id')['charttime'].diff().fillna(timedelta(seconds=0))
    df = df.assign(**{'6h': np.nan, '12h': np.nan, '24h': np.nan, 'Anuria_12h': np.nan, 'Urine_stage': np.nan, 'Urine_output_rate': np.nan})

    anuria_threshold = 50.0

    def operation(target):

        target = target.reset_index(drop=True)
        target['Urine_output_rate'] = target['Urine'] / (target['charttime_diff'].dt.total_seconds() / 3600.0) / target['Weight']
        target['cum_value'] = target['Urine'][::-1].cumsum()
        target['cum_time_diff'] = target['charttime_diff'][::-1].cumsum().dt.total_seconds() / 3600.0

        for i in range(1, len(target)):
            group = target.iloc[1:i+1]
            group['cum_value'] = group['cum_value'] - group['cum_value'].iloc[-1] + group['Urine'].iloc[-1]
            group['cum_time_diff'] = group['cum_time_diff'] - group['cum_time_diff'].iloc[-1] + (group['charttime_diff'].dt.total_seconds() / 3600.0).iloc[-1]

            for threshold_hours_min, threshold_hours_max, rate_threshold, column_name, stage in [
                (6, 12, 0.5, '6h', 1),
                (12, float('inf'), 0.5, '12h', 2),
                (24, float('inf'), 0.3, '24h', 3)]:

                condition = (group['cum_time_diff'] >= threshold_hours_min) & (group['cum_time_diff'] <= threshold_hours_max)
                filtered_group = group.loc[condition]

                if not filtered_group.empty:
                    urine_output_rate = filtered_group['cum_value'] / filtered_group['cum_time_diff'] / target['Weight'].iloc[i]

                    if urine_output_rate.iloc[-1] < rate_threshold:
                        target.at[i, column_name] = 1
                        target.at[i, 'Urine_stage'] = stage

                    if column_name == '12h' and filtered_group['cum_value'].iloc[-1] < anuria_threshold:
                        target.at[i, 'Anuria_12h'] = 1
                        target.at[i, 'Urine_stage'] = 3

                    if column_name in ['6h', '12h', '24h']:
                        target.at[i, f'Urine_volume_{column_name}'] = filtered_group['cum_value'].iloc[-1]
                        target.at[i, f'Urine_output_rate_{column_name}'] = urine_output_rate.iloc[-1]

        return target

    df = df.groupby('stay_id', group_keys=False).parallel_apply(operation).reset_index(drop=True)
    df = df[(~df['Urine_output_rate'].isna()) & (df['Urine_output_rate'] != float('inf'))]
    df['Urine_charttime_diff'] = (df['charttime_diff'].dt.total_seconds() / 3600).round(1)
    df.loc[df['Urine_charttime_diff'] >= 12, ['Urine_stage', 'Anuria_12h']] = 3, 1

    return df

def Urine_resampling(df, label):

    def operation(group):
        
        import pandas as pd
        import numpy as np
        from datetime import timedelta

        start_frame = group.iloc[[0]].copy()
        end_frame = group.iloc[[0]].copy()

        start_frame['charttime'] = start_frame['intime']
        end_frame['charttime'] = end_frame['outtime']

        na_columns = [label, 'Urine', 'Weight', '6h', '12h', '24h', 'Anuria_12h', 'Urine_stage', 'cum_value', 'cum_time_diff', 'Urine_charttime_diff', 'Urine_output_rate_6h', 'Urine_output_rate_12h', 'Urine_output_rate_24h', 'Urine_volume_6h', 'Urine_volume_12h', 'Urine_volume_24h']
        start_frame[na_columns] = np.nan
        end_frame[na_columns] = np.nan

        group = pd.concat([start_frame, group, end_frame])
        group.set_index('charttime', inplace=True)

        resampled = group.resample('6h', origin=group['intime'].iloc[0], label='left').last()

        if pd.isna(resampled[label].iloc[-1]):
            last_contribute = group[(group.index < group['outtime']) & (group.index >= group['outtime'] - timedelta(hours=6))]
            if not last_contribute.empty:
                resampled.iloc[-1] = last_contribute.iloc[-1]

        if pd.isna(resampled[label].iloc[0]):
            first_contribute = group[(group.index > group['intime']) & (group.index <= group['intime'] + timedelta(hours=6))]
            if not first_contribute.empty:
                resampled.iloc[0] = first_contribute.iloc[-1]

        for col in ['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'RRT_icu_history', 'RRT_hosp_history']:
            resampled[col] = group[col].iat[0]

        resampled['charttime'] = resampled.index
        resampled['timedelta'] = resampled.index - group['intime'].iloc[0]

        resampled['RRT'] = 0
        if group['RRT'].sum() > 0:
            resampled['RRT'].iloc[-1] = 1
        else:
            resampled['RRT'] = resampled['RRT'].fillna(0)

        return resampled

    df = df.groupby('stay_id', group_keys=False).parallel_apply(operation).reset_index(drop=True)
    df['timedelta'] = pd.to_timedelta(df['timedelta'])
    df['current_charttime'] = round((df['timedelta'] + timedelta(hours=6)).dt.total_seconds() / 3600, 1)
    df['Urine_presence'] = df['Urine'].notna().astype(int)

    return df

def Urine_copy_mask(df, label):

    def Urine_copy(target):
        columns_to_fill = [
            label, 'Urine_charttime_diff', 'Urine_stage', 'Urine', 
            '6h', '12h', '24h', 'Anuria_12h', 'cum_value', 'cum_time_diff', 
            'Urine_output_rate_6h','Urine_output_rate_12h', 'Urine_output_rate_24h',
            'Urine_volume_6h', 'Urine_volume_12h', 'Urine_volume_24h'
        ]
        
        for col in columns_to_fill:
            target[col] = target[col].fillna(method='ffill', limit=4)
        
        return target

    df = df.groupby('stay_id', group_keys=False).parallel_apply(Urine_copy).reset_index(drop=True)

    def Urine_mask(target):
        columns_to_mask = [
            label, 'Urine_charttime_diff', 'Urine_stage','Urine',
            '6h', '12h', '24h', 'Anuria_12h', 'Weight', 'cum_value', 'cum_time_diff', 
            'Urine_output_rate_6h', 'Urine_output_rate_12h', 'Urine_output_rate_24h', 
            'Urine_volume_6h', 'Urine_volume_12h', 'Urine_volume_24h'
        ]
        
        for col in columns_to_mask:
            mask_col = col + '_mask'
            target[mask_col] = target[col].notna().astype(int)
            target[col] = target[col].fillna(0)

        return target

    df = Urine_mask(df)
    return df

def GT(df, stage_pool_total, stage_pool_SCr_total, RRT_pool_total):
    presence_cols = [f'GT_presence_{i}' for i in range(6, 49, 6)]
    presence_cols_SCr = [f'GT_presence_{i}_SCr' for i in range(6, 49, 6)]

    stage_cols = ['GT_stage_1', 'GT_stage_2', 'GT_stage_3', 'GT_stage_3D']
    stage_cols_SCr = ['GT_stage_1_SCr', 'GT_stage_2_SCr', 'GT_stage_3_SCr', 'GT_stage_3D_SCr']

    for col in presence_cols + stage_cols + presence_cols_SCr + stage_cols_SCr:
        df[col] = 0

    df['charttime_end'] = df['charttime'] + timedelta(hours=6)

    def operation(target):

        from datetime import timedelta

        target = target.sort_values('charttime').reset_index(drop=True)
        stage_pool = stage_pool_total[stage_pool_total['stay_id'].isin(target['stay_id'].unique())]
        stage_pool_SCr = stage_pool_SCr_total[stage_pool_SCr_total['stay_id'].isin(target['stay_id'].unique())]
        RRT_pool = RRT_pool_total[RRT_pool_total['stay_id'].isin(target['stay_id'].unique())]

        for hours in range(6, 49, 6):
            target[f'charttime_{hours}'] = target['charttime_end'] + timedelta(hours=hours)

        for i, row in target.iterrows():
            for hours in range(6, 49, 6):
                area = stage_pool[(stage_pool['charttime'] > row['charttime_end']) & 
                                  (stage_pool['charttime'] <= row[f'charttime_{hours}'])]['stage']
                area_SCr = stage_pool_SCr[(stage_pool_SCr['charttime'] > row['charttime_end']) & 
                                  (stage_pool_SCr['charttime'] <= row[f'charttime_{hours}'])]['stage']
                RRT_count = len(RRT_pool[(RRT_pool['starttime'] >= row['charttime_end']) & 
                                         (RRT_pool['starttime'] <= row[f'charttime_{hours}'])]) + target['RRT'].iloc[i]

                if area.sum() > 0:
                    target.loc[i, f'GT_presence_{hours}'] = 1
                    if hours == 48:
                        max_stage = area.max()
                        if max_stage >= 1:
                            target.loc[i, 'GT_stage_1'] = 1
                        if max_stage >= 2:
                            target.loc[i, 'GT_stage_2'] = 1
                        if max_stage >= 3:
                            target.loc[i, 'GT_stage_3'] = 1

                if area_SCr.sum() > 0:
                    target.loc[i, f'GT_presence_{hours}_SCr'] = 1
                    if hours == 48:
                        max_stage = area_SCr.max()
                        if max_stage >= 1:
                            target.loc[i, 'GT_stage_1_SCr'] = 1
                        if max_stage >= 2:
                            target.loc[i, 'GT_stage_2_SCr'] = 1
                        if max_stage >= 3:
                            target.loc[i, 'GT_stage_3_SCr'] = 1
                            
                if RRT_count > 0:
                    target.loc[i, [f'GT_presence_{hours}'] + stage_cols] = 1
                    target.loc[i, [f'GT_presence_{hours}_SCr'] + stage_cols_SCr] = 1

        return target

    df = df.groupby('stay_id', group_keys=False).parallel_apply(operation)
    df = df.sort_values(['subject_id', 'stay_id', 'charttime']).reset_index(drop=True)

    return df

def MAX_AKI(df,SCr_icu,Urine_icu):

    def operation(target):

        SCr = SCr_icu[SCr_icu['stay_id'] == target['stay_id'].iloc[0]]
        Urine = Urine_icu[Urine_icu['stay_id'] == target['stay_id'].iloc[0]]
        
        max_SCr = max(SCr['SCr_stage'])
        max_Urine = max(Urine['Urine_stage'])
        
        target['max_stage'] = max(max_SCr,max_Urine)

        return target

    df = df.groupby('stay_id',group_keys=False).parallel_apply(operation).reset_index(drop=True)
    
    return df

def onehot(df):
      
    careunit_columns = ['Surgical', 'Medical', 'Medical/Surgical', 'Other']

    for col in careunit_columns:
        df.loc[df['first_careunit'] == col, col] = 1
        df[col] = df[col].fillna(0)
        
    race_columns = ['WHITE', 'UNKNOWN', 'BLACK', 'HISPANIC OR LATINO', 'OTHER', 'ASIAN']
    
    for col in race_columns:

        if col == 'BLACK':
            df.loc[df['race'] == col, col] = 1
            df[col] = df[col].fillna(0)

        else :
            df.loc[df['race'] == col, 'BLACK'] = 0
            df['BLACK'] = df['BLACK'].fillna(0)

    df.loc[df['gender'] == 'F','gender'] = 0
    df.loc[df['gender'] == 'M','gender'] = 1
    
    df['length'] = df.groupby('stay_id')['stay_id'].transform('size')

    return df

def ICD(df, icd_list):
    df_icd = df[df['icd_code'].str.startswith(tuple(icd_list))]
    return df_icd

def check_ICD(stage,df9,df10,label):

    check = pd.concat([df9,df10])
    check.drop_duplicates(subset='hadm_id',inplace=True)
    check[label] = 1
    check = check[['hadm_id',label]]
    check = pd.merge(stage,check,on='hadm_id',how='left')
    check[label] = check[label].fillna(0)

    return check 

def extract_icd_matches(df9, df10, codes_9, codes_10):
    matched_9 = ICD(df9, codes_9)
    matched_10 = ICD(df10, codes_10)
    return matched_9, matched_10

def add_comorbidity(stage_df, df9, df10, codes_9, codes_10, label):
    disease_9, disease_10 = extract_icd_matches(df9, df10, codes_9, codes_10)
    return check_ICD(stage_df, disease_9, disease_10, label)

def gap(target, label='valuenum', use_label_column=True):
    if use_label_column and label in target.columns:
        target[label] = target[label].round(1)
        target[label + '_diff'] = target[label].diff().round(1)
    else:
        target[label] = target['valuenum'].round(1)
        target[label + '_diff'] = target['valuenum'].diff().round(1)

    return target

def Mapping(df,df_icu,df_hosp,label,copy):
    
    if not df_hosp.empty :
        common_columns = df_icu.columns.intersection(df_hosp.columns).tolist()
        df_data = pd.concat([df_icu[common_columns], df_hosp[common_columns]]).drop_duplicates().sort_values(['subject_id', 'charttime']).reset_index(drop=True)
    
    else :
        df_data = df_icu.drop_duplicates().sort_values(['subject_id', 'charttime']).reset_index(drop=True)
    
    if not label in df.columns :
        df[label] = np.nan
        df[label + '_diff'] = np.nan

    if not label in df_data.columns:
        df_data.rename(columns = {'valuenum_diff': label + '_diff'}, inplace = True)    
        df_data.rename(columns = {'valuenum': label}, inplace = True)    

    def operation(target):

        import pandas as pd
        from datetime import timedelta

        target.reset_index(inplace=True,drop=True)
        target_data = df_data[df_data['subject_id'] == target['subject_id'].iloc[0]]

        for i in range(len(target)):

            target_lower = target['charttime'].iloc[i]
            target_upper = target_lower  + timedelta(hours=6)
            target_value = target_data[(target_data['charttime'] < target_upper) & (target_data['charttime'] >= target_lower)].sort_values(['charttime'])

            if not target_value.empty: 
                if pd.isna(target[label].iloc[i]):
                    target[label].iloc[i] = target_value[label].iloc[-1]
                    target[label + '_diff'].iloc[i] = target_value[label + '_diff'].iloc[-1]

        if copy:

            target[label] = target[label].fillna(method='ffill', limit=4)
            target[label + '_diff'] = target[label + '_diff'].fillna(method='ffill', limit=4)

            if target[label].isnull().iloc[0] == True:
                
                target_hosp = df_hosp.loc[df_hosp['subject_id'] == target['subject_id'].iloc[0]]
                
                forward = target_hosp[(target_hosp['charttime'] < target['charttime'].iloc[0]) & (target_hosp['charttime'] >= (target['charttime'].iloc[0] - timedelta(days=1)))]
                forward = forward.sort_values(['charttime'])

                if not forward.empty:
                
                    target_value = forward.iloc[-1]
                    cri = target_value['charttime']
                    target_2 = target[(target['charttime'] > cri) & (target['charttime'] < (cri + timedelta(days=1)))]
                    target_2 = target_2[label].isnull().sum()

                    if target_2 != 0:
                        target.loc[target.index[:target_2], label] = target_value[label]
                        target.loc[target.index[:target_2], label + '_diff'] = target_value[label + '_diff']

        return target
    
    df = df.groupby('stay_id',group_keys=False).parallel_apply(operation).reset_index(drop=True)
    print(df[label].isnull().sum())
    
    df[label + '_mask']  = df[label].notna().astype(int)
    df[label + '_diff_mask']  = df[label + '_diff'].notna().astype(int)

    return df

def Vital(df, stage, vitalsign):

    def convert_temperature(df):
        df['valuenum'] = (df['valuenum'] - 32) * 5 / 9
        df['itemid'] = 223762
        df['valueuom'] = '°C'
        return df

    item_config = {
        'temperature': {'itemids': [223762, 223761], 'range': (32, 43), 'unit': '°C'},
        'heartrate': {'itemids': [220045], 'range': (0, 300), 'unit': 'bpm'},
        'sbp': {'itemids': [220050, 220179], 'range': (0, 300), 'unit': 'mmHg'},
        'dbp': {'itemids': [220051, 220180], 'range': (10, 175), 'unit': 'mmHg'},
        'resprate': {'itemids': [220210], 'range': (0, 60), 'unit': 'insp/min'},
        'o2sat': {'itemids': [220227, 220277], 'range': (0, 100), 'unit': '%'},
    }

    subject_id = stage[['subject_id']].drop_duplicates()

    df_all = {}
    for label, info in item_config.items():
        dfs = []
        for idx, itemid in enumerate(info['itemids']):
            df_v = df[df['itemid'] == itemid].copy()
            if label == 'temperature' and itemid == 223761:
                df_v = convert_temperature(df_v)
            df_v = df_v[(df_v['valuenum'] >= info['range'][0]) & (df_v['valuenum'] <= info['range'][1])]
            df_v['valueuom'] = info['unit']
            df_v = pd.merge(df_v, subject_id, on='subject_id', how='inner')
            df_v = df_v.sort_values(['subject_id', 'charttime'])
            df_v[label] = df_v['valuenum'].round(1)
            df_v[label + '_diff'] = df_v.groupby('subject_id')[label].diff().round(1)
            if len(info['itemids']) > 1:
                df_v['Art'] = int(idx == 0)
            dfs.append(df_v)

        df_all[label] = pd.concat(dfs).drop_duplicates().reset_index(drop=True)

        if label in vitalsign.columns:
            vitalsign[label] = vitalsign[label].where(
                (vitalsign[label] >= info['range'][0]) & (vitalsign[label] <= info['range'][1])
            )

    vitalsign_results = {}
    for label in item_config.keys():
        if label in vitalsign.columns:
            vdf = vitalsign[vitalsign[label].notna()][['subject_id', 'charttime', label]]
            vdf = vdf.sort_values(['subject_id', 'charttime']).reset_index(drop=True)
            vdf = vdf.groupby('subject_id', group_keys=False).apply(lambda x: gap(x, label)).reset_index(drop=True)
            vitalsign_results[label] = vdf

    combined_results = {}
    for label in ['sbp', 'dbp', 'o2sat']:
        df_combined = df_all[label]
        df_combined = df_combined.sort_values(['subject_id', 'charttime']).reset_index(drop=True)
        df_combined = df_combined.groupby('subject_id', group_keys=False).apply(lambda x: gap(x, label)).reset_index(drop=True)
        combined_results[label] = df_combined

    mapping_info = [
        ('temperature', df_all['temperature'], vitalsign_results['temperature'], True),
        ('heartrate', df_all['heartrate'], vitalsign_results['heartrate'], True),
        ('sbp', df_all['sbp'][df_all['sbp']['Art'] == 1], vitalsign_results['sbp'], False),
        ('sbp', df_all['sbp'][df_all['sbp']['Art'] == 0], vitalsign_results['sbp'], True),
        ('dbp', df_all['dbp'][df_all['dbp']['Art'] == 1], vitalsign_results['dbp'], False),
        ('dbp', df_all['dbp'][df_all['dbp']['Art'] == 0], vitalsign_results['dbp'], True),
        ('resprate', df_all['resprate'], vitalsign_results['resprate'], True),
        ('o2sat', df_all['o2sat'][df_all['o2sat']['Art'] == 1], vitalsign_results['o2sat'], False),
        ('o2sat', df_all['o2sat'][df_all['o2sat']['Art'] == 0], vitalsign_results['o2sat'], True),
    ]

    del df

    for label, df_src, df_vital, copy_flag in mapping_info:
        stage = Mapping(stage, df_src, df_vital, label, copy=copy_flag)

    return stage

def process_lab_data(variables, subject_ids, engine):

    results = []
    
    for var_name, icu_itemid, hosp_itemids, left_bound, right_bound, left_inclusive, right_inclusive in variables:

        if hosp_itemids is not None:

            query = f"""
            select subject_id, stay_id, charttime, itemid, valuenum
            from mimiciv_icu.chartevents
            where valuenum is not null and valuenum != 999999 and stay_id is not null and 
            itemid in ({', '.join(map(str, icu_itemid))})
            order by subject_id, itemid, charttime
            """

            var_icu = pd.read_sql(query,engine)
            var_icu = var_icu[var_icu['subject_id'].isin(subject_ids)]

            query = f"""
            select subject_id, charttime, itemid, valuenum
            from mimiciv_hosp.labevents
            where valuenum is not null and valuenum != 999999 and
            itemid in ({', '.join(map(str, hosp_itemids))})
            order by subject_id, itemid, charttime
            """
    
            var_hosp = pd.read_sql(query,engine)
            var_hosp = var_hosp[var_hosp['subject_id'].isin(subject_ids)]

            var_icu['ICU'] = 1
            var_hosp['ICU'] = 0

            var_data = pd.concat([var_icu, var_hosp]).sort_values(by=['subject_id', 'charttime'])

            if left_bound is not None:
                if left_inclusive:
                    var_data = var_data[var_data.valuenum >= left_bound]
                else:
                    var_data = var_data[var_data.valuenum > left_bound]

            if right_bound is not None:
                if right_inclusive:
                    var_data = var_data[var_data.valuenum <= right_bound]
                else:
                    var_data = var_data[var_data.valuenum < right_bound]
        else :
            query = f"""
            select subject_id, stay_id, itemid, starttime, amount
            from mimiciv_icu.inputevents
            Where amount is not null and amount != 999999 and
            itemid in ({', '.join(map(str, icu_itemid))})
            order by subject_id, starttime
            """

            var_icu = pd.read_sql(query,engine)
            var_icu = var_icu[var_icu['subject_id'].isin(subject_ids)]
            var_icu['ICU'] = 1
            var_data = var_icu
            var_data.rename(columns={'amount': 'valuenum','starttime':'charttime'}, inplace=True)

        var_data = var_data.groupby('subject_id', group_keys=False).parallel_apply(gap).reset_index(drop=True)
        var_data.rename(columns={'valuenum': var_name, 'valuenum_diff': f'{var_name}_diff'}, inplace=True)
        results.append(var_data)
        results = pd.concat(results).reset_index(drop=True)

        icu = results[results['ICU']==1]
        hosp = results[results['ICU']==0]

    return icu, hosp

def Anti_Mapping(df,ICU,label):

    data = ICU[['subject_id','starttime']]

    df[f'{label}'] = 0
    
    def operation(target):

        from datetime import timedelta

        target = target.reset_index(drop=True)
        subject_id = target['subject_id'].iloc[0]
        target_data = data[data['subject_id'] == subject_id]

        for i in range(len(target)):
            
            target_lower = target['charttime'].iloc[i]
            target_upper = target_lower + timedelta(hours=6)

            case = target_data[(target_data['starttime'] >= target_lower) & (target_data['starttime'] <= target_upper)]
            if not case.empty :
                target[f'{label}'].iloc[i] = 1

        return target
    
    df = df.groupby('subject_id',group_keys=False).parallel_apply(operation).reset_index(drop=True)
    print(df[f'{label}'].value_counts())

    return df

def merge_anti(target):
    merge_dict = {
        'Others': [
            'Aztreonam', 'Doxycycline', 'Tigecycline', 'Bactrim (SMX/TMP)', 'Azithromycin',
            'Erythromycin', 'Colistin', 'Daptomycin', 'Linezolid', 'Clindamycin',
            'Acyclovir', 'Rifampin', 'Amikacin', 'Gentamicin', 'Tobramycin'
        ],
        'Fluoroquinolones': ['Ciprofloxacin', 'Levofloxacin', 'Moxifloxacin'],
        'Penicillins': ['Ampicillin', 'Nafcillin', 'Penicillin G potassium', 'Penicillin gen4'],
        'Betalactam': ['Ampicillin/Sulbactam (Unasyn)', 'Piperacillin/Tazobactam (Zosyn)'],
        'Cephalosporins': ['Cefazolin', 'Cefepime', 'Ceftazidime', 'Ceftriaxone', 'Keflex', 'Ceftaroline'],
        'Carbapenems': ['Imipenem/Cilastatin', 'Meropenem', 'Ertapenem sodium (Invanz)'],
        
    }

    for new_col, cols_to_merge in merge_dict.items():
        target[new_col] = target[cols_to_merge].max(axis=1)
        target.drop(columns=cols_to_merge, inplace=True)

    return target

def MV(df, engine):

    query = """
        select subject_id, hadm_id, stay_id, itemid, value, starttime, endtime
        from mimiciv_icu.procedureevents
        where value is not null
        and itemid in (225792, 225794)
        order by subject_id, itemid, starttime
        """

    MV_icu = pd.read_sql_query(query,engine)

    def operation(target):

        target_value = MV_icu[MV_icu['stay_id'] == target['stay_id'].iloc[0]].reset_index(drop=True)

        if ~target_value.empty :
            for i in range(len(target_value)):
                target.loc[(target['charttime'] >= target_value['starttime'].iloc[i]) & (target['charttime'] < target_value['endtime'].iloc[i]),'MV'] = 1
        
        return target

    df['MV'] = 0
    df = df.groupby('stay_id',group_keys=False).parallel_apply(operation).reset_index(drop=True)

    return df

def Fluid_Mapping(df,ICU,pre_adm,value):

    data = ICU[['subject_id','starttime','endtime','tev','rate',value]]

    df['input_total'] = np.nan   #total fluid given
    df['input_6hr'] = np.nan  #fluid given at this step
    
    def operation(target):

        from datetime import timedelta
        import numpy as np

        target = target.reset_index(drop=True)
        subject_id = target['subject_id'].iloc[0]
        target_data = data[data['subject_id'] == subject_id]
        pread = pre_adm[pre_adm['stay_id'] == target['stay_id'].iloc[0]]

        if len(pread) > 0:           
            totvol = np.nansum(pread['inputpreadm'])
        else:
            totvol = np.nan

        for i in range(len(target)):
            
            target_lower = target['charttime'].iloc[i]
            target_upper = target_lower + timedelta(hours=6)

            case_1_value, case_2_value, case_3_value, case_4_value = 0,0,0,0

            case_1 = target_data[(target_data['starttime'] >= target_lower) & (target_data['endtime'] <= target_upper)]
            if not case_1.empty:  case_1_value = np.nansum((case_1[value] * (target_data['endtime']-target_data['starttime'])).dt.total_seconds() / 3600)

            case_2 = target_data[(target_data['starttime'] <= target_lower) & (target_data['endtime'] >= target_lower) & (target_data['endtime'] <= target_upper)]
            if not case_2.empty:  case_2_value = np.nansum((case_2[value] * (target_data['endtime']-target_lower)).dt.total_seconds() / 3600)

            case_3 = target_data[(target_data['starttime'] >= target_lower) & (target_data['starttime'] <= target_upper) & (target_data['endtime'] >= target_upper)]
            if not case_3.empty:  case_3_value = np.nansum((case_3[value] * (target_upper-target_data['starttime']).dt.total_seconds() / 3600))

            case_4 = target_data[(target_data['starttime'] <= target_lower) & (target_data['endtime'] >= target_upper)]
            if not case_4.empty:  case_4_value = np.nansum((case_4[value] * (target_upper-target_lower)).dt.total_seconds() / 3600)

            infu = np.nansum([case_1_value,case_2_value,case_3_value,case_4_value])
            bolus = np.nansum(target_data[(np.isnan(target_data['rate'])) & (target_data['starttime'] >= target_lower) & (target_data['starttime'] <= target_upper)]['tev'])

            totvol = np.nansum([totvol, infu, bolus])
            target.loc[i,'input_total'] = totvol    #total fluid given
            target.loc[i,'input_6hr'] = np.nansum([infu, bolus])  #fluid given at this step
            target.loc[i,'input_6hr_bolus'] = np.nansum([bolus])  #fluid given at this step only bolus

        return target
    
    df = df.groupby('subject_id',group_keys=False).parallel_apply(operation).reset_index(drop=True)

    df.loc[df['input_total'] < 0, 'input_total'] = 0
    df.loc[df['input_total'].isna(),'input_total'] = 0
    
    df.loc[df['input_6hr'] < 0, 'input_6hr'] = 0
    df.loc[df['input_6hr'].isna(),'input_6hr'] = 0

    df.loc[df['input_6hr_bolus'] < 0, 'input_6hr_bolus'] = 0
    df.loc[df['input_6hr_bolus'].isna(),'input_6hr_bolus'] = 0

    del data

    return df

def Vaso_Mapping(df,ICU,label):

    data = ICU[['subject_id','itemid','starttime','endtime','rate_std']]

    def operation(target):

        from datetime import timedelta
        import numpy as np

        target = target.reset_index(drop=True)
        subject_id = target['subject_id'].iloc[0]
        target_data = data[data['subject_id'] == subject_id]
    
        target[f'max_{label}'] = 0
        target[f'median_{label}'] = 0

        for i in range(len(target)):
            
            target_lower = target['charttime'].iloc[i]
            target_upper = target_lower + timedelta(hours=6)

            #v = ((endv >= t0) & (endv <= t1)) | ((startv >= t0) & (endv<=t1)) | ((startv >= t0) & (startv <= t1))| ((startv <= t0) & (endv>=t1))

            # VASOPRESSORS
            # for MV: 4 possibles cases, each one needing a different way to compute the dose of VP actually administered:
            #----t0---start----end-----t1----
            #----start---t0----end----t1----
            #-----t0---start---t1---end
            #----start---t0----t1---end----

            target_value = target_data[((target_data['endtime'] <= target_upper) & (target_data['endtime'] >= target_lower)) | 
                                       ((target_data['endtime'] <= target_upper) & (target_data['starttime'] >= target_lower)) |
                                       ((target_data['starttime'] <= target_upper) & (target_data['starttime'] >= target_lower)) |
                                       ((target_data['endtime'] >= target_upper) & (target_data['starttime'] <= target_lower))].sort_values(['starttime'])
            
            if not target_value.empty: 

                max_val = np.nanmax(target_value['rate_std'])
                median_val = np.nanmedian(target_value['rate_std'])

                target.loc[i, f'max_{label}'] = max_val
                target.loc[i, f'median_{label}'] = median_val

        return target
    
    df = df.groupby('subject_id',group_keys=False).parallel_apply(operation).reset_index(drop=True)

    del data

    return df

def split_and_prepare(target_subject_id, target_stay_id, stage, stratify_cols):

    def iterative_split(df, test_size, stratify_columns):
        one_hot_cols = pd.get_dummies(df[stratify_columns], columns=stratify_columns)
        stratifier = IterativeStratification(
            n_splits=2,
            order=len(stratify_columns),
            sample_distribution_per_fold=[test_size, 1 - test_size]
        )
        train_idx, test_idx = next(stratifier.split(df.values, one_hot_cols.values))
        return df.iloc[train_idx], df.iloc[test_idx]

    def sort_by_length(df):
        length_col = 'length_x' if 'length_x' in df.columns else 'length'
        return df.sort_values(by=length_col).reset_index(drop=True)

    def make_stay_split(subject_df, target_stay_df):
        return pd.merge(target_stay_df, subject_df[['subject_id']], on='subject_id') \
                 .sort_values('length').reset_index(drop=True)

    def make_stage_split(stage_df, stay_df):
        merged = pd.merge(stage_df, stay_df[['stay_id']], on='stay_id')
        merged['stay_id'] = pd.Categorical(merged['stay_id'], categories=stay_df['stay_id'], ordered=True)
        return merged.sort_values(['stay_id', 'charttime']).reset_index(drop=True)

    train_sub, test_sub = iterative_split(target_subject_id, 0.1, stratify_cols)
    train_sub, valid_sub = iterative_split(train_sub, 1/9, stratify_cols)
    valid_sub, calib_sub = iterative_split(valid_sub, 0.5, stratify_cols)

    splits_subject = {
        'train': sort_by_length(pd.merge(train_sub, target_subject_id[['subject_id', 'length']], on='subject_id')),
        'valid': sort_by_length(pd.merge(valid_sub, target_subject_id[['subject_id', 'length']], on='subject_id')),
        'calibration': sort_by_length(pd.merge(calib_sub, target_subject_id[['subject_id', 'length']], on='subject_id')),
        'test': sort_by_length(pd.merge(test_sub, target_subject_id[['subject_id', 'length']], on='subject_id')),
    }

    splits_stay = {
        k: make_stay_split(v, target_stay_id)
        for k, v in splits_subject.items()
    }

    splits_stage = {
        k: make_stage_split(stage, v)
        for k, v in splits_stay.items()
    }

    return splits_stay, splits_stage

def Dataset(df,numeric_features,presence_features,GT_presence,GT_stage):
    
    X_numerics = []
    X_presences = []
    Y_mains = []
    Y_subs = []
    masks = []

    start = int(min(df['length']))
    end = int(max(df['length']))
    datasets = []

    for i in tqdm(range(start,end+1)):
        
        target = df[df['length'] == i]

        if not target.empty:

            X_presence = target[presence_features]
            X_numeric = target[numeric_features]      

            Y_main = target[GT_presence]
            Y_sub = target[GT_stage]

            X_numeric = X_numeric.values  # Convert DataFrame to NumPy array
            X_numeric = torch.tensor(X_numeric.reshape(-1, i, X_numeric.shape[1]), dtype=torch.float32)
            padding = torch.zeros((X_numeric.shape[0],56-i,X_numeric.shape[2]),dtype=torch.float32)
            X_numeric = torch.cat((X_numeric,padding),dim=1)

            X_presence = X_presence.values  # Convert DataFrame to NumPy array
            X_presence = torch.tensor(X_presence.reshape(-1, i, X_presence.shape[1]), dtype=torch.float32)
            padding = torch.zeros((X_presence.shape[0],56-i,X_presence.shape[2]),dtype=torch.float32)
            X_presence = torch.cat((X_presence,padding),dim=1)

            Y_main = Y_main.values  # Convert DataFrame to NumPy array
            Y_main = torch.tensor(Y_main.reshape(-1, i, Y_main.shape[1]), dtype=torch.float32)
            padding = torch.zeros((Y_main.shape[0],56-i,Y_main.shape[2]),dtype=torch.float32)
            Y_main = torch.cat((Y_main,padding),dim=1)
            Y_main = Y_main.transpose(1, 2)

            Y_sub = Y_sub.values  # Convert DataFrame to NumPy array
            Y_sub = torch.tensor(Y_sub.reshape(-1, i, Y_sub.shape[1]), dtype=torch.float32)
            padding = torch.zeros((Y_sub.shape[0],56-i,Y_sub.shape[2]),dtype=torch.float32)
            Y_sub = torch.cat((Y_sub,padding),dim=1)
            Y_sub = Y_sub.transpose(1, 2)

            mask_valid = torch.zeros((Y_sub.shape[0],i),dtype=torch.float32)
            mask_ones = torch.ones((Y_sub.shape[0],56-i),dtype=torch.float32)
            mask = torch.cat((mask_valid,mask_ones),dim=1)

            X_numerics.append(X_numeric)
            X_presences.append(X_presence)
            Y_mains.append(Y_main)
            Y_subs.append(Y_sub)
            masks.append(mask)

    X_numeric = torch.cat(X_numerics,dim=0)
    X_presence = torch.cat(X_presences,dim=0)
    Y_main = torch.cat(Y_mains,dim=0)
    Y_sub = torch.cat(Y_subs,dim=0)
    mask = torch.cat(masks,dim=0)

    dataset = TensorDataset(X_numeric, X_presence, Y_main, Y_sub, mask) #Y_sub, mask)
    dataloader = DataLoader(dataset, batch_size = X_numeric.shape[0], shuffle=False, drop_last=False)

    for X_numeric, X_presence, Y_main, Y_sub, mask in dataloader:
        batch_dataset = TensorDataset(X_numeric, X_presence, Y_main, Y_sub, mask) #Y_sub, mask)
        datasets.append(batch_dataset)

    return datasets    

def Dataset_test(df,numeric_features,presence_features,GT_presence,GT_stage):
        
    start = int(min(df['length']))
    end = int(max(df['length']))
    datasets = []
    data = []

    for i in tqdm(range(start,end+1)):

        target = df[df['length'] == i]
        data.append(target)
        
        if not target.empty:

            X_presence = target[presence_features]
            X_numeric = target[numeric_features] 
            
            Y_main  = target[GT_presence]
            Y_sub  = target[GT_stage]

            X_numeric = X_numeric.values  # Convert DataFrame to NumPy array
            X_numeric = torch.tensor(X_numeric.reshape(-1, i, X_numeric.shape[1]), dtype=torch.float32)
            padding = torch.zeros((X_numeric.shape[0],56-i,X_numeric.shape[2]),dtype=torch.float32)
            X_numeric = torch.cat((X_numeric,padding),dim=1)

            X_presence = X_presence.values  # Convert DataFrame to NumPy array
            X_presence = torch.tensor(X_presence.reshape(-1, i, X_presence.shape[1]), dtype=torch.float32)
            padding = torch.zeros((X_presence.shape[0],56-i,X_presence.shape[2]),dtype=torch.float32)
            X_presence = torch.cat((X_presence,padding),dim=1)

            Y_main = Y_main.values  # Convert DataFrame to NumPy array
            Y_main = torch.tensor(Y_main.reshape(-1, i, Y_main.shape[1]), dtype=torch.float32)
            padding = torch.zeros((Y_main.shape[0],56-i,Y_main.shape[2]),dtype=torch.float32)
            Y_main = torch.cat((Y_main,padding),dim=1)
            Y_main = Y_main.transpose(1, 2)

            Y_sub = Y_sub.values  # Convert DataFrame to NumPy array
            Y_sub = torch.tensor(Y_sub.reshape(-1, i, Y_sub.shape[1]), dtype=torch.float32)
            padding = torch.zeros((Y_sub.shape[0],56-i,Y_sub.shape[2]),dtype=torch.float32)
            Y_sub = torch.cat((Y_sub,padding),dim=1)
            Y_sub = Y_sub.transpose(1, 2)

            mask_valid = torch.zeros((Y_sub.shape[0],i),dtype=torch.float32)
            mask_ones = torch.ones((Y_sub.shape[0],56-i),dtype=torch.float32)
            mask = torch.cat((mask_valid,mask_ones),dim=1)

            dataset = TensorDataset(X_numeric, X_presence, Y_main, Y_sub, mask)
            dataloader = DataLoader(dataset, batch_size = X_numeric.shape[0], shuffle=False, drop_last=False)

            for X_numeric, X_presence, Y_main, Y_sub, mask in dataloader:
                batch_dataset = TensorDataset(X_numeric, X_presence, Y_main, Y_sub, mask)
                datasets.append(batch_dataset)

    data = pd.concat(data)

    return datasets

def Dataset_CEP(df,numeric_features,presence_features):
    
    X_numerics = []
    X_presences = []
    Y_mains = []
    Y_subs = []
    masks = []

    start = 1
    end = 56 + 1
    datasets = []

    for i in tqdm(range(start,end)):
        
        target = df[df['length'] == i]

        if not target.empty:

            X_presence = target[presence_features]
            X_numeric = target[numeric_features]      

            Y_main  = target[['GT_presence_6','GT_presence_12','GT_presence_18','GT_presence_24','GT_presence_30','GT_presence_36','GT_presence_42','GT_presence_48']]
            Y_sub  = target[['GT_stage_3D','GT_stage_3','GT_stage_2','GT_stage_1']]

            X_numeric = X_numeric.values  # Convert DataFrame to NumPy array
            X_numeric = np.round(X_numeric, decimals=1)
            X_numeric = torch.tensor(X_numeric.reshape(-1, i, X_numeric.shape[1]), dtype=torch.float32)
            padding = torch.zeros((X_numeric.shape[0],56-i,X_numeric.shape[2]),dtype=torch.float32)
            X_numeric = torch.cat((X_numeric,padding),dim=1)

            X_presence = X_presence.values  # Convert DataFrame to NumPy array
            X_presence = torch.tensor(X_presence.reshape(-1, i, X_presence.shape[1]), dtype=torch.float32)
            padding = torch.zeros((X_presence.shape[0],56-i,X_presence.shape[2]),dtype=torch.float32)
            X_presence = torch.cat((X_presence,padding),dim=1)

            Y_main = Y_main.values  # Convert DataFrame to NumPy array
            Y_main = torch.tensor(Y_main.reshape(-1, i, Y_main.shape[1]), dtype=torch.float32)
            padding = torch.zeros((Y_main.shape[0],56-i,Y_main.shape[2]),dtype=torch.float32)
            Y_main = torch.cat((Y_main,padding),dim=1)
            Y_main = Y_main.transpose(1, 2)

            Y_sub = Y_sub.values  # Convert DataFrame to NumPy array
            Y_sub = torch.tensor(Y_sub.reshape(-1, i, Y_sub.shape[1]), dtype=torch.float32)
            padding = torch.zeros((Y_sub.shape[0],56-i,Y_sub.shape[2]),dtype=torch.float32)
            Y_sub = torch.cat((Y_sub,padding),dim=1)
            Y_sub = Y_sub.transpose(1, 2)

            mask_valid = torch.zeros((Y_sub.shape[0],i),dtype=torch.float32)
            mask_ones = torch.ones(Y_sub.shape[0],56-i,dtype=torch.float32)
            mask = torch.cat((mask_valid,mask_ones),dim=1)

            X_numerics.append(X_numeric)
            X_presences.append(X_presence)
            Y_mains.append(Y_main)
            Y_subs.append(Y_sub)
            masks.append(mask)

    X_numeric = torch.cat(X_numerics,dim=0)
    X_presence = torch.cat(X_presences,dim=0)
    Y_main = torch.cat(Y_mains,dim=0)
    Y_sub = torch.cat(Y_subs,dim=0)
    mask = torch.cat(masks,dim=0)

    dataset = TensorDataset(X_numeric, X_presence, Y_main, Y_sub, mask) #Y_sub, mask)
    dataloader = DataLoader(dataset, batch_size = X_numeric.shape[0], shuffle=False, drop_last=False)

    for X_numeric, X_presence, Y_main, Y_sub, mask in dataloader:
        batch_dataset = TensorDataset(X_numeric, X_presence, Y_main, Y_sub, mask) #Y_sub, mask)
        datasets.append(batch_dataset)

    return datasets


def compute_pos_weights_presence(df, prefix='GT_presence_', suffix=''):
    hours = [6, 12, 18, 24, 30, 36, 42, 48]
    weights = [
        (df[f"{prefix}{h}{suffix}"] == 0).sum() / (df[f"{prefix}{h}{suffix}"] == 1).sum()
        for h in hours
    ]
    return torch.tensor(weights, dtype=torch.float32)

def compute_pos_weights_stage(df, stage_cols, rrt_weight=None):
    stage_weights = [
        (df[col] == 0).sum() / (df[col] == 1).sum()
        for col in stage_cols
    ]
    if rrt_weight is not None:
        return torch.tensor([rrt_weight] + stage_weights, dtype=torch.float32)
    else:
        return torch.tensor(stage_weights, dtype=torch.float32)

def compute_rrt_pos_weight(id_df, rrt_col='RRT'):
    num_pos = (id_df[rrt_col] == 1).sum()
    num_neg = len(id_df) - num_pos
    return num_neg / num_pos

def train(
    model,
    train_dataloader,
    valid_dataloader,
    pos_weights,
    batchsize: int,
    learning_rate: float,
    num_epochs: int,
    lr_decay_factor: float,
    lr_decay_steps: int,
    LD: bool,
    CDF: bool,
    path: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    def cdf(t: torch.Tensor) -> torch.Tensor:
        return torch.cummax(t, dim=1)[0]

    LossClass = CustomBCELoss if CDF else nn.BCEWithLogitsLoss

    criterion_main_train = [LossClass(pos_weight=w) for w in pos_weights[0]]
    criterion_sub_train  = [LossClass(pos_weight=w) for w in pos_weights[1]]
    criterion_main_valid = [LossClass(pos_weight=w) for w in pos_weights[2]]
    criterion_sub_valid  = [LossClass(pos_weight=w) for w in pos_weights[3]]

    losses = {"valid_loss": 0, "main_loss": 0, "sub_loss": 0}
    early_stopping = EarlyStopping(
        patience=lr_decay_steps,
        path=path,
        loss_names=list(losses.keys()),
        verbose=False,
    )

    optimizer = PCGrad(optim.Adam(model.parameters(), lr=learning_rate))
    scheduler = ExponentialLR(optimizer.optimizer, gamma=lr_decay_factor)

    step = 0
    model.to(device)

    for epoch in range(num_epochs):
        train_loader = DataLoader(
            train_dataloader.dataset[0],
            batch_size=batchsize,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=4,
        )

        for batch in train_loader:
            step += 1
            model.train()

            inputs_numeric, inputs_presence, tgt_main, tgt_sub, mask = [x.to(device) for x in batch]
            out_main, out_sub = model(inputs_numeric, inputs_presence)

            if CDF:
                out_main = cdf(torch.sigmoid(out_main))
                out_sub  = cdf(torch.sigmoid(out_sub))

            loss_main = sum(c(out_main[:, j], tgt_main[:, j]) for j, c in enumerate(criterion_main_train))
            loss_sub  = sum(c(out_sub[:,  j], tgt_sub[:,  j]) for j, c in enumerate(criterion_sub_train))
            loss_total = loss_main + loss_sub

            optimizer.zero_grad()
            optimizer.pc_backward([loss_main, loss_sub])
            optimizer.step()

            model.eval()
            with torch.no_grad():
                v_loss_main = 0.0
                v_loss_sub  = 0.0

                for v_batch in valid_dataloader.dataset:
                    v_inputs_numeric, v_inputs_presence, v_tgt_main, v_tgt_sub, _ = [
                        x.to(device) for x in v_batch.tensors
                    ]
                    v_out_main, v_out_sub = model(v_inputs_numeric, v_inputs_presence)

                    if CDF:
                        v_out_main = cdf(torch.sigmoid(v_out_main))
                        v_out_sub  = cdf(torch.sigmoid(v_out_sub))

                    v_loss_main += sum(c(v_out_main[:, j], v_tgt_main[:, j]) for j, c in enumerate(criterion_main_valid))
                    v_loss_sub  += sum(c(v_out_sub[:,  j], v_tgt_sub[:,  j]) for j, c in enumerate(criterion_sub_valid))

                v_loss_total = v_loss_main + v_loss_sub

                losses.update(
                    valid_loss=v_loss_total.item(),
                    main_loss=v_loss_main.item(),
                    sub_loss=v_loss_sub.item(),
                )
                early_stopping(
                    losses, model, epoch, num_epochs,
                    loss_total.item(), loss_main.item(), loss_sub.item(),
                    v_loss_total.item(), v_loss_main.item(), v_loss_sub.item()
                )

                gc.collect()
                torch.cuda.empty_cache()

                if early_stopping.early_stop:
                    model.load_state_dict(torch.load(path))
                    break

                if LD and (step % lr_decay_steps == 0):
                    scheduler.step()

            if early_stopping.early_stop:
                break
        if early_stopping.early_stop:
            break

    return model, losses["valid_loss"]

def objective(trial, train_dataloader, valid_dataloader, pos_weights, device):

    numeric_input_size   = train_dataloader.dataset[0].tensors[0].shape[-1]
    presence_input_size  = train_dataloader.dataset[0].tensors[1].shape[-1]

    seq_len   = 56
    num_epochs= 1_000_000

    hidden_size          = trial.suggest_int("hidden_size",        50, 200, step=50)
    embedding_size       = trial.suggest_int("embedding_size",     25, 100, step=25)
    recurrent_num_layers = trial.suggest_int("recurrent_num_layers", 1, 5)
    embedding_num_layers = trial.suggest_int("embedding_num_layers", 1, 5)

    CB              = trial.suggest_categorical("CB",  [0, 1])      # 0: sum, 1: concat
    recurrent_type  = trial.suggest_categorical("recurrent_type",
                                                ["LSTM", "RNN", "GRU"])
    activation_type = trial.suggest_categorical("activation_type",
                                                ["ReLU", "LeakyReLU", "Tanh",
                                                 "ELU", "SELU", "CELU", "GELU"])

    batchsize       = trial.suggest_categorical("batchsize",      [64, 128, 256, 512])
    learning_rate   = trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2])
    lr_decay_steps  = trial.suggest_categorical("lr_decay_steps", [800, 400, 200, 100])
    lr_decay_factor = trial.suggest_categorical("lr_decay_factor",
                                                [0.7, 0.8, 0.85, 0.9, 0.95])

    HN  = bool(trial.suggest_categorical("highway_network", [0, 1]))
    LD  = bool(trial.suggest_categorical("LD",              [0, 1]))
    LN  = bool(trial.suggest_categorical("LN",              [0, 1]))
    CDF = bool(trial.suggest_categorical("CDF",             [0, 1]))

    model = AKIPredictionModel(
        hidden_size, embedding_size,
        recurrent_num_layers, embedding_num_layers,
        activation_type, recurrent_type, seq_len,
        LN, HN,
        numeric_input_size, presence_input_size, CB
    ).to(device)

    ckpt_path = os.path.join("model", f"trial_{trial.number+1}_model.pt")
    os.makedirs("model", exist_ok=True)

    _, valid_loss = train(
        model,
        train_dataloader,
        valid_dataloader,
        pos_weights,
        batchsize           = batchsize,
        learning_rate       = learning_rate,
        num_epochs          = num_epochs,
        lr_decay_factor     = lr_decay_factor,
        lr_decay_steps      = lr_decay_steps,
        LD                  = LD,
        CDF                 = CDF,
        path                = ckpt_path
    )

    return valid_loss

def test(model,dataloader):

    model.eval()

    criterion =  [nn.BCELoss()]

    main_datasets_6h = []
    main_datasets_12h = []
    main_datasets_18h = []
    main_datasets_24h = []
    main_datasets_30h = []
    main_datasets_36h = []
    main_datasets_42h = []
    main_datasets_48h = []

    sub_datasets_1 = []
    sub_datasets_2 = []
    sub_datasets_3 = []
    sub_datasets_3D = []

    with torch.no_grad():

        test_loss = 0.0
      
        for data in dataloader.dataset:
                
            inputs_numeric, inputs_presence, targets_main, targets_sub, mask = [d.to(device) for d in data.tensors]
            
            out_main, out_sub = model(inputs_numeric, inputs_presence)

            test_main_loss = 0.0
            test_sub_loss = 0.0  

            out_main = F.sigmoid(out_main)
            out_sub = F.sigmoid(out_sub)    

            out_main = cdf(out_main)
            out_sub = cdf(out_sub)
            
            for i in range(mask.shape[0]):

                length = (mask[i,:] == 0).sum()
                test_main_loss += sum(criterion(out_main[i, j, :length], targets_main[i, j, :length]) for j, criterion in enumerate(criterion))
                test_sub_loss += sum(criterion(out_sub[i, j, :length], targets_sub[i, j, :length]) for j, criterion in enumerate(criterion))

            test_loss += (test_main_loss + test_sub_loss).item()

            i = (mask[0,:] == 0).sum()

            dataset = TensorDataset(out_main[:,0,:i],out_main[:,1,:i],out_main[:,2,:i],out_main[:,3,:i],out_main[:,4,:i],out_main[:,5,:i],out_main[:,6,:i],out_main[:,7,:i],
                                    out_sub[:,3,:i],out_sub[:,2,:i],out_sub[:,1,:i],out_sub[:,0,:i],
                                    targets_main[:,0,:i],targets_main[:,1,:i],targets_main[:,2,:i],targets_main[:,3,:i],targets_main[:,4,:i],targets_main[:,5,:i],targets_main[:,6,:i],targets_main[:,7,:i],
                                    targets_sub[:,3,:i],targets_sub[:,2,:i],targets_sub[:,1,:i],targets_sub[:,0,:i])
                                    
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

            for out_main_6h,out_main_12h,out_main_18h,out_main_24h,out_main_30h,out_main_36h,out_main_42h,out_main_48h,out_sub_1,out_sub_2,out_sub_3,out_sub_3D,targets_main_6h,targets_main_12h,targets_main_18h,targets_main_24h,targets_main_30h,targets_main_36h,targets_main_42h,targets_main_48h,targets_sub_1,targets_sub_2,targets_sub_3,targets_sub_3D in dataloader:
                
                main_dataset_6h = TensorDataset(out_main_6h, targets_main_6h)
                main_dataset_12h = TensorDataset(out_main_12h, targets_main_12h)
                main_dataset_18h = TensorDataset(out_main_18h, targets_main_18h)
                main_dataset_24h = TensorDataset(out_main_24h, targets_main_24h)
                main_dataset_30h = TensorDataset(out_main_30h, targets_main_30h)
                main_dataset_36h = TensorDataset(out_main_36h, targets_main_36h)
                main_dataset_42h = TensorDataset(out_main_42h, targets_main_42h)
                main_dataset_48h = TensorDataset(out_main_48h, targets_main_48h)

                main_datasets_6h.append(main_dataset_6h)
                main_datasets_12h.append(main_dataset_12h)
                main_datasets_18h.append(main_dataset_18h)
                main_datasets_24h.append(main_dataset_24h)
                main_datasets_30h.append(main_dataset_30h)
                main_datasets_36h.append(main_dataset_36h)
                main_datasets_42h.append(main_dataset_42h)
                main_datasets_48h.append(main_dataset_48h)

                sub_dataset_1 = TensorDataset(out_sub_1, targets_sub_1)
                sub_dataset_2 = TensorDataset(out_sub_2, targets_sub_2)
                sub_dataset_3 = TensorDataset(out_sub_3, targets_sub_3)
                sub_dataset_3D = TensorDataset(out_sub_3D, targets_sub_3D)

                sub_datasets_1.append(sub_dataset_1)
                sub_datasets_2.append(sub_dataset_2)
                sub_datasets_3.append(sub_dataset_3)
                sub_datasets_3D.append(sub_dataset_3D)               

    print(f"Test Loss: {test_loss:.4f}")

    main_datasets = [main_datasets_6h, main_datasets_12h, main_datasets_18h, main_datasets_24h,main_datasets_30h, main_datasets_36h, main_datasets_42h, main_datasets_48h]
    sub_datasets = [sub_datasets_1, sub_datasets_2, sub_datasets_3, sub_datasets_3D]
    
    return main_datasets, sub_datasets

def reshape(dataloader, prob_pos, y_true):

    index = 0
    datasets = []

    prob_pos_tensor = torch.tensor(prob_pos)
    y_true_tensor = torch.tensor(y_true)
    
    for batch in dataloader.dataset:
        
        X, Y = batch.tensors
        X = X.cpu().detach().numpy().tolist()
        Y = Y.cpu().detach().numpy().tolist()

        score = prob_pos_tensor[(index):(index+len(X[0]))]
        true = y_true_tensor[(index):(index+len(Y[0]))]
        
        dataset = TensorDataset(score.unsqueeze(0), true.unsqueeze(0))
        datasets.append(dataset)

        index += len(X[0])

    dataloaders = DataLoader(datasets, batch_size=1, shuffle=False, drop_last=True)

    return dataloaders

def calibration(calibration_dataloader, dataloader):
    y_true, y_scores = step_ROC(calibration_dataloader)
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_scores, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibration Curve")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.show()

    ir = IsotonicRegression(out_of_bounds='clip').fit(y_scores, y_true)

    y_true, prob_pos = step_ROC(dataloader)
    brier_score_before = brier_score_loss(y_true, prob_pos)
    prob_pos_calibrated = ir.transform(prob_pos)
    brier_score_after = brier_score_loss(y_true, prob_pos_calibrated)

    print("Brier Score Before:", round(brier_score_before, 4))
    print("Brier Score After:", round(brier_score_after, 4))

    if brier_score_after > brier_score_before:
        print('[!] Calibration degraded performance. Reverting to original.')
        return dataloader

    print('[✓] Calibration improved performance.')
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, prob_pos_calibrated, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibration Curve")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.show()

    calibrated_dataloader = reshape(dataloader, prob_pos_calibrated, y_true)

    return calibrated_dataloader

def step_ROC(dataloader):

    y_true = []
    y_scores = []
    
    for batch in dataloader.dataset:
        
        X, Y = batch.tensors

        X = X.cpu().detach().numpy().tolist()
        y_scores.extend(X[0])

        Y = Y.cpu().detach().numpy().tolist()
        y_true.extend(Y[0])

    return y_true, y_scores

def AKI_step_ROC(dataloader):

    sum = 0
    y_true = []
    y_scores = []
    
    for batch in dataloader.dataset:
        X, Y = batch.tensors
        if torch.sum(Y) > 0:
            Y = Y.cpu().detach().numpy().tolist()
            y_true.extend(Y[0])

            X = X.cpu().detach().numpy().tolist()
            y_scores.extend(X[0])

            sum += 1

    return y_true, y_scores

def AKI_first_ROC(dataloader):

    sum = 0
    y_true = []
    y_scores = []
    values = float(1)

    for batch in dataloader.dataset:
        X, Y = batch.tensors
        if torch.sum(Y) > 0:
            Y = Y.cpu().detach().numpy().tolist()
            index = Y[0].index(values)
            y_true.extend(Y[0][:index+1])

            X = X.cpu().detach().numpy().tolist()
            y_scores.extend(X[0][:index+1])

            sum+= 1
    
    return y_true, y_scores

def AUROC(y_true,y_scores):

    auroc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    return fpr, tpr, thresholds, auroc

def AUPRC(y_true,y_scores):

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)

    return recall, precision, thresholds, auprc

def calculate_confusion_matrix(y_true, y_scores, threshold):

    y_pred = [1 if score >= threshold else 0 for score in y_scores]

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == 1 and pred_label == 1:
            TP += 1
        elif true_label == 0 and pred_label == 1:
            FP += 1
        elif true_label == 0 and pred_label == 0:
            TN += 1
        elif true_label == 1 and pred_label == 0:
            FN += 1

    evaluation(TP,FP,TN,FN)
    
    return TP, FP, TN, FN

def evaluation(TP,FP,TN,FN):
    Sensitivity = TP / (TP+FN) # Recall
    Specitivity = TN / (FP+TN) 
    Accuaracy = (TP+TN) / (TP+TN+FP+FN)
    Precision = TP / (TP+FP)
    F1 = (2 * Precision * Sensitivity) / (Precision+Sensitivity)

    print('Accuracy :',round(Accuaracy,3)*100,'%')
    print('Precision :',round(Precision,3)*100,'%')
    print('Sensitivity :',round(Sensitivity,3)*100,'%')
    print('Specitivity :',round(Specitivity,3)*100,'%')
    print('F1 score :',round(F1,3))

def bootstrap_auroc(y_true, y_pred, n_bootstraps=200):
    bootstrapped_scores = []
    rng = np.random.RandomState(42)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for _ in range(n_bootstraps):

        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return confidence_lower, confidence_upper

def bootstrap_auprc(y_true, y_pred, n_bootstraps=200):

    bootstrapped_scores = []
    rng = np.random.RandomState(42)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for _ in range(n_bootstraps):

        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue

        score = average_precision_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return confidence_lower, confidence_upper

def Result(dataloader):

    y_true, y_scores = step_ROC(dataloader)

    fpr, tpr, AUROC_thresholds, auroc = AUROC(y_true, y_scores)
    ci_lower, ci_upper = bootstrap_auroc(y_true, y_scores)
    print(f"{round(auroc*100,1)} ({round(ci_lower*100,1)}-{round(ci_upper*100,1)})")

    recall, precision, AUPRC_thresholds, auprc = AUPRC(y_true, y_scores)
    ci_lower, ci_upper = bootstrap_auprc(y_true, y_scores)
    print(f"{round(auprc*100,1)} ({round(ci_lower*100,1)}-{round(ci_upper*100,1)})")
    
    y_true, y_scores = AKI_step_ROC(dataloader)

    fpr, tpr, AUROC_thresholds, auroc = AUROC(y_true, y_scores)
    ci_lower, ci_upper = bootstrap_auroc(y_true, y_scores)
    print(f"{round(auroc*100,1)} ({round(ci_lower*100,1)}-{round(ci_upper*100,1)})")

    recall, precision, AUPRC_thresholds, auprc = AUPRC(y_true, y_scores)
    ci_lower, ci_upper = bootstrap_auprc(y_true, y_scores)
    print(f"{round(auprc*100,1)} ({round(ci_lower*100,1)}-{round(ci_upper*100,1)})")

    y_true, y_scores = AKI_first_ROC(dataloader)

    fpr, tpr, AUROC_thresholds, auroc = AUROC(y_true, y_scores)
    ci_lower, ci_upper = bootstrap_auroc(y_true, y_scores)
    print(f"{round(auroc*100,1)} ({round(ci_lower*100,1)}-{round(ci_upper*100,1)})")

    recall, precision, AUPRC_thresholds, auprc = AUPRC(y_true, y_scores)
    ci_lower, ci_upper = bootstrap_auprc(y_true, y_scores)
>>>>>>> 721a46a72ac12c15e84b7395aefc902162a33939
    print(f"{round(auprc*100,1)} ({round(ci_lower*100,1)}-{round(ci_upper*100,1)})")