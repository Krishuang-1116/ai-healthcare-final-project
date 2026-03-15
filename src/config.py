# src/config.py
'''
Configuration file for the project. 
Contains constants and settings used throughout the codebase.
'''
TARGET_COL = 'hospital_expire_flag'

ID_COLS = ['subject_id', 'hadm_id', 'stay_id', 'intime']

VITAL_COLS = [
    "heart_rate_mean",
    "sbp_mean",
    "dbp_mean",
    "mbp_mean",
    "resp_rate_mean",
    "temperature_mean",
    "spo2_mean"
]

LAB_COLS = [
    "wbc_min",
    "wbc_max",
    "platelets_min",
    "hemoglobin_min",
    "creatinine_max",
    "sodium_min",
    "sodium_max",
    "glucose_max",
    "albumin_min",
]

BG_COLS = [
    "lactate_max",
    "ph_min",
    "po2_min",
    "pco2_max"
]

SOFA_COLS = ['sofa']

FEATURE_COLS = VITAL_COLS + LAB_COLS + BG_COLS + SOFA_COLS

HIGH_MISSING_COLS = [
    "albumin_min",
    "lactate_max",
    "ph_min",
    "po2_min",
    "pco2_max"
]

N_SPLITS = 5
RANDOM_STATE = 42
