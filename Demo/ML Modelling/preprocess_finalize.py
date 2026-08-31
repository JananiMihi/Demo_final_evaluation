"""
Full preprocessing pipeline for Full_Data_Set (1).xlsx
Outputs a clean, analysis-ready DataFrame saved as:
  D:/Downloads/Full_Data_Set_preprocessed.xlsx
"""

import pandas as pd
import numpy as np
import warnings
import re
warnings.filterwarnings('ignore')

DATA_PATH = 'D:/Downloads/Full_Data_Set (1).xlsx'
OUT_PATH  = 'D:/Downloads/Full_Data_Set_preprocessed.xlsx'

print('='*70)
print('LOADING DATA')
print('='*70)

bp   = pd.read_excel(DATA_PATH, sheet_name='bp_data')
ctrl = pd.read_excel(DATA_PATH, sheet_name='control')

print(f'bp_data  : {bp.shape[0]} rows x {bp.shape[1]} cols')
print(f'control  : {ctrl.shape[0]} rows x {ctrl.shape[1]} cols')

# ── STEP 1: Fix participant IDs (case + strip whitespace) ─────────────────
print('\n--- STEP 1: Fix participant IDs ---')

# Rename messy column headers consistently in both sheets
rename_map = {
    "PSS Total\n(0\x9640)":  "PSS_Total",
    "PSS Total\n(0–40)":     "PSS_Total",
    "PSS\nCategory":         "PSS_Category",
    "Stress Index\n(%)":     "Stress_Index_pct",
    "PSQI Global\n(0\x9621)": "PSQI_Global",
    "PSQI Global\n(0–21)":   "PSQI_Global",
    "PSQI\nCategory":        "PSQI_Category",
    "Sleep Index\n(%)":      "Sleep_Index_pct",
    "Nutrition\nIndex (%)":  "Nutrition_Index_pct",
    "Protein\n(g/kg)":       "Protein_g_kg",
    "Calories\n(kcal/day)":  "Calories_kcal",
    "Weight (kg)":           "Weight_kg",
    "Height (cm)":           "Height_cm",
    "Training Period (yrs)": "Training_yrs",
    "Unnamed: 1":            "Name",
}
bp.rename(columns=rename_map, inplace=True)
bp.columns = [c.strip() for c in bp.columns]
bp['participant'] = bp['participant'].astype(str).str.strip().str.upper()
print(f'bp_data participants: {bp["participant"].nunique()}')

# ── STEP 2: Merge control if it has data ─────────────────────────────────
print('\n--- STEP 2: Merge bp_data + control ---')
if ctrl.shape[0] > 0 and ctrl.shape[1] > 0:
    ctrl.rename(columns=rename_map, inplace=True)
    ctrl.columns = [c.strip() for c in ctrl.columns]
    ctrl['participant'] = ctrl['participant'].astype(str).str.strip().str.upper()
    all_cols = list(dict.fromkeys(list(bp.columns) + list(ctrl.columns)))
    for col in all_cols:
        if col not in bp.columns:
            bp[col] = np.nan
        if col not in ctrl.columns:
            ctrl[col] = np.nan
    ctrl = ctrl[bp.columns]
    df = pd.concat([bp, ctrl], ignore_index=True)
    print(f'Control sheet merged. New shape: {df.shape[0]} rows x {df.shape[1]} cols')
else:
    df = bp.copy()
    print('Control sheet is empty — using bp_data only.')

print(f'Total unique participants: {df["participant"].nunique()}')
print(f'Participant list: {sorted(df["participant"].unique())}')

# ── STEP 3: Drop completely empty columns ────────────────────────────────
print('\n--- STEP 3: Drop 100% empty columns ---')
empty_cols = [c for c in df.columns if df[c].isna().all()]
print(f'Dropping {len(empty_cols)} empty columns: {empty_cols}')
df.drop(columns=empty_cols, inplace=True)

# ── STEP 4: Drop duplicate columns (same name after rename) ──────────────
print('\n--- STEP 4: Drop exact duplicate columns ---')
df = df.loc[:, ~df.columns.duplicated()]
print(f'Shape after dedup columns: {df.shape}')

# ── STEP 5: Fix object-type numeric columns ───────────────────────────────
print('\n--- STEP 5: Fix object-type numeric columns ---')

def fix_numeric_col(series):
    """Coerce to numeric; handle Unicode minus, Excel date objects."""
    def _fix(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        # Excel datetime object parsed as date (e.g. n_csa_valid "1/1" → date)
        if hasattr(val, 'month') and hasattr(val, 'day'):
            # Return month value as numerator (best guess for "x/x" format)
            return float(val.month)
        s = str(val).strip()
        # Unicode minus → ASCII minus
        s = s.replace('\u2212', '-').replace('\u2013', '-')
        # Handle fraction strings like "0/0", "3/5" → keep as is, return NaN for fractions
        if '/' in s:
            parts = s.split('/')
            try:
                num = float(parts[0])
                den = float(parts[1])
                return num if den == 0 else num  # just take numerator for csa_valid count
            except:
                return np.nan
        try:
            return float(s)
        except:
            return np.nan
    return series.apply(_fix)

# Identify object columns that should be numeric (exclude known categoricals)
known_cats = {'participant', 'Muscle', 'Group', 'Gender', 'PSS_Category',
              'PSQI_Category', 'shape_class', 'csa_method', 'Name',
              'length_status', 'csa_flag'}
window_cols = {c for c in df.columns if 'window' in c.lower()}
unnamed_cols = {c for c in df.columns if 'Unnamed' in str(c)}
skip = known_cats | window_cols | unnamed_cols

obj_cols = [c for c in df.select_dtypes('object').columns if c not in skip]
fixed = []
for col in obj_cols:
    df[col] = fix_numeric_col(df[col])
    fixed.append(col)
print(f'Fixed {len(fixed)} object columns → numeric: {fixed}')

# ── STEP 6: Drop rows with no Muscle assigned ─────────────────────────────
print('\n--- STEP 6: Drop rows with missing Muscle ---')
before = len(df)
df = df[df['Muscle'].notna()].copy()
after = len(df)
print(f'Dropped {before - after} rows with no Muscle (before={before}, after={after})')

# ── STEP 7: Drop TR muscle (not in 8-muscle protocol) ────────────────────
print('\n--- STEP 7: Drop TR muscle rows ---')
tr_rows = (df['Muscle'] == 'TR').sum()
df = df[df['Muscle'] != 'TR'].copy()
print(f'Dropped {tr_rows} TR rows')

# ── STEP 8: Fix group assignment errors ───────────────────────────────────
print('\n--- STEP 8: Check & fix group assignment ---')
part_groups = df.groupby('participant')['Group'].nunique()
multi_grp   = part_groups[part_groups > 1]
print(f'Participants with >1 group: {len(multi_grp)}')
for p in multi_grp.index:
    grps_rows = df[df['participant'] == p][['participant', 'Muscle', 'Group']]
    print(f'\n  {p}:')
    print(grps_rows.to_string())
# Auto-fix: assign modal group (most common group per participant)
if len(multi_grp) > 0:
    modal_group = df.groupby('participant')['Group'].agg(lambda x: x.mode()[0])
    df['Group'] = df['participant'].map(modal_group)
    print('\n  Auto-fixed: assigned each participant their most-frequent group.')
    print('  Verify manually: ', dict(df[df['participant'].isin(multi_grp.index)]
                                        .groupby('participant')['Group'].first()))

# ── STEP 8b: Strip whitespace from Group values ───────────────────────────
df['Group'] = df['Group'].astype(str).str.strip()
df['Muscle'] = df['Muscle'].astype(str).str.strip()

# ── STEP 9: Encode categorical columns ───────────────────────────────────
print('\n--- STEP 9: Encode categorical columns ---')
df['Gender_bin']  = (df['Gender'].astype(str).str.strip().str.lower() == 'male').astype(int)
group_map = {'G1': 1, 'G2': 2, 'G3': 3, 'G4': 4}
df['Group_code']  = df['Group'].map(group_map).fillna(0).astype(int)
print('Gender_bin and Group_code created.')
print(f'Group distribution:\n{df.groupby("Group")["participant"].nunique().sort_index()}')

# ── STEP 10: Coerce all remaining feature columns to numeric ──────────────
print('\n--- STEP 10: Force all feature columns to numeric ---')
non_num_cols = {'participant', 'Name', 'Muscle', 'Group', 'Gender',
                'PSS_Category', 'PSQI_Category', 'shape_class',
                'csa_method', 'length_status', 'csa_flag', 'Gender_bin'}
non_num_cols |= window_cols | unnamed_cols

for col in df.columns:
    if col not in non_num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ── STEP 11: Missing value imputation ────────────────────────────────────
print('\n--- STEP 11: Missing value imputation ---')

# Define feature columns for imputation (all numeric except target + IDs)
id_cols     = ['participant', 'Name', 'Muscle', 'Group', 'Gender',
               'PSS_Category', 'PSQI_Category', 'shape_class',
               'csa_method', 'length_status', 'csa_flag',
               'Gender_bin', 'Group_code']
target_col  = 'volume_corrected_cm3'

num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in id_cols + [target_col]]

missing_before = df[num_cols].isna().sum().sum()
print(f'Missing cells in feature columns (before): {missing_before}')

# Impute: per-muscle median, fallback to global median
for col in num_cols:
    if df[col].isna().any():
        per_muscle_median = df.groupby('Muscle')[col].transform('median')
        df[col] = df[col].fillna(per_muscle_median)
        global_median = df[col].median()
        if pd.notna(global_median):
            df[col] = df[col].fillna(global_median)

missing_after = df[num_cols].isna().sum().sum()
print(f'Missing cells in feature columns (after) : {missing_after}')

# Volume: report which participants still have no volume (do NOT impute — it is the target)
vol_miss = df[target_col].isna()
print(f'\nRows still missing volume (target — NOT imputed): {vol_miss.sum()}')
no_vol_parts = sorted(df[vol_miss]['participant'].unique())
print(f'Participants with no volume: {no_vol_parts}')

# ── STEP 12: Final summary ────────────────────────────────────────────────
print('\n--- STEP 12: FINAL SUMMARY ---')
print(f'Final shape                : {df.shape}')
print(f'Total unique participants  : {df["participant"].nunique()}')
print(f'Participants WITH volume   : {df[df[target_col].notna()]["participant"].nunique()}')
print(f'Participants WITHOUT volume: {df[df[target_col].isna()]["participant"].nunique()}')
print(f'Muscles present            : {sorted(df["Muscle"].dropna().unique())}')
print(f'Groups present             : {sorted(df["Group"].dropna().unique())}')
print(f'\nGroup x participant count:')
print(df.groupby('Group')['participant'].nunique().sort_index())
print(f'\nMuscle row counts:')
print(df['Muscle'].value_counts().sort_index())
print(f'\nRemaining missing values (all columns): {df.isna().sum().sum()}')
print(f'Remaining missing — feature cols only : {df[num_cols].isna().sum().sum()}')
print(f'Remaining missing — volume (target)   : {df[target_col].isna().sum()}')

# ── STEP 13: Save to Excel ────────────────────────────────────────────────
print('\n--- STEP 13: Save preprocessed file ---')
with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='preprocessed', index=False)
    # Summary sheet
    summary = pd.DataFrame({
        'Item': [
            'Total rows', 'Total columns',
            'Unique participants (all)',
            'Participants WITH volume',
            'Participants WITHOUT volume',
            'Missing feature cells',
            'Missing volume (target) rows',
            'Muscles included',
            'Groups present',
        ],
        'Value': [
            len(df), df.shape[1],
            df['participant'].nunique(),
            df[df[target_col].notna()]['participant'].nunique(),
            df[df[target_col].isna()]['participant'].nunique(),
            int(df[num_cols].isna().sum().sum()),
            int(df[target_col].isna().sum()),
            ', '.join(sorted(df['Muscle'].dropna().unique())),
            ', '.join(sorted(df['Group'].dropna().unique())),
        ]
    })
    summary.to_excel(writer, sheet_name='summary', index=False)

    # Per-participant volume status
    vol_status = df.groupby('participant').agg(
        Group=('Group', 'first'),
        Muscles_present=('Muscle', lambda x: ', '.join(sorted(x.dropna().unique()))),
        Has_volume=('volume_corrected_cm3', lambda x: 'YES' if x.notna().any() else 'NO'),
        Volume_rows=('volume_corrected_cm3', 'count'),
    ).reset_index()
    vol_status.to_excel(writer, sheet_name='participant_status', index=False)

print(f'Saved → {OUT_PATH}')
print('\n' + '='*70)
print('PREPROCESSING COMPLETE')
print('='*70)
