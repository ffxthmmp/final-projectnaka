import pandas as pd
import numpy as np

df = pd.read_excel("ces.xlsx")

# Separate features and target
X = df[['S_RANK', 'P_ID', 'R_ID', 'FEE_NAME', 'S_PARENT', 'GPA', 'GPA_MATCH', 'GPA_SCI']]
y = df['BRANCH']

# ตรวจสอบจำนวนตัวอย่างในแต่ละคลาสของ "BRANCH"
branch_counts = df['BRANCH'].value_counts()
print('Original counts:')
print(branch_counts)

# หาจำนวนตัวอย่างมากที่สุดในคลาสใดคลาสหนึ่ง
max_samples = max(branch_counts)

# สร้างตัวอย่างใหม่เพื่อสุ่ม
X_resampled = pd.DataFrame()
y_resampled = pd.Series(dtype='int')

for class_number in np.unique(df['BRANCH']):
    class_subset = df[df['BRANCH'] == class_number]
    resampled_subset = class_subset.sample(max_samples, replace=True, random_state=42)
    X_resampled = pd.concat([X_resampled, resampled_subset.drop('BRANCH', axis=1)], axis=0)
    y_resampled = pd.concat([y_resampled, resampled_subset['BRANCH']], axis=0)

# ตรวจสอบจำนวนตัวอย่างใหม่ในแต่ละคลาส
resampled_counts_manual = y_resampled.value_counts()
print('\nResampled counts:')
print(resampled_counts_manual)
