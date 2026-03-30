CREATE OR REPLACE TABLE `ai4health-488910.mimic4_features.pneumonia_firstday_features`
AS
WITH
  pneumonia_icu AS (
    SELECT temp.subject_id, temp.hadm_id, a.stay_id, a.intime
    FROM
      (
        SELECT DISTINCT subject_id, hadm_id
        FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
        WHERE
          (
            icd_version = 9
            AND (icd_code BETWEEN '480' AND '486' OR icd_code = '4870'))
          OR (
            icd_version = 10 AND SUBSTR(icd_code, 1, 3) BETWEEN 'J12' AND 'J18')
      ) temp
    JOIN `physionet-data.mimiciv_3_1_icu.icustays` a
      ON temp.hadm_id = a.hadm_id
    QUALIFY ROW_NUMBER() OVER (PARTITION BY temp.hadm_id ORDER BY a.intime) = 1
  )
SELECT
  p.subject_id,
  p.hadm_id,
  p.stay_id,
  p.intime,
  v.heart_rate_mean,
  v.sbp_mean,
  v.dbp_mean,
  v.mbp_mean,
  v.resp_rate_mean,
  v.temperature_mean,
  v.spo2_mean,
  l.wbc_min,
  l.wbc_max,
  l.platelets_min,
  l.hemoglobin_min,
  l.creatinine_max,
  l.sodium_min,
  l.sodium_max,
  l.glucose_max,
  l.albumin_min,
  bg.lactate_max,
  bg.ph_min,
  bg.po2_min,
  bg.pco2_max,
  s.sofa,
  adm.hospital_expire_flag
FROM pneumonia_icu p
LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_vitalsign` v
  ON p.stay_id = v.stay_id
LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_lab` l
  ON p.stay_id = l.stay_id
LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_bg` bg
  ON p.stay_id = bg.stay_id
LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_sofa` s
  ON p.stay_id = s.stay_id
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` adm
  ON p.hadm_id = adm.hadm_id;
