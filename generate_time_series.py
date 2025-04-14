import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from tqdm import tqdm
import pickle

# Function to clean columns by adding 'side' and removing R_ and L_ prefixes
def clean_columns(df):
    side = 'R' if any('R_' in col for col in df.columns) else 'L' if any('L_' in col for col in df.columns) else None
    df['side'] = side
    df.columns = [col.replace('R_', '').replace('L_', '') for col in df.columns]
    return df

# Function to align trial to a reference using DTW and interpolation
def align_trial(reference, trial):
    aligned_trial = {}
    for col in trial.columns:
        try:
            ref_series = reference[col]
            trial_series = trial[col]

            # Skip if either series is too short or NaN
            if len(ref_series) < 2 or len(trial_series) < 2:
                print(f"[Warning] Skipping column '{col}' due to insufficient length.")
                continue

            distance, path = fastdtw(ref_series, trial_series)
            path = np.array(path)

            # Interpolate to align with reference
            interp = interp1d(path[:, 1], trial_series.iloc[path[:, 1]],
                              kind='linear', bounds_error=False, fill_value="extrapolate")
            aligned_trial[col] = interp(np.arange(len(ref_series)))
        except Exception as e:
            print(f"[Error] Failed to align column '{col}': {e}")
            continue

    return pd.DataFrame(aligned_trial)

# Function to process and align data for all patients
def process_patients(patient_dir):
    folder_names = [f for f in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, f))]
    
    references, references2, overall_dfs = [], [], []  # Lists to store references and dataframes
    
    for patient in tqdm(folder_names, desc="Processing patients"):
        trials = []
        aligned_trials = []
        if patient not in ['P24', 'P34']:
            name = os.path.join(os.getcwd(), patient)
            combined_data_with_kinematics = pd.read_pickle(os.path.join(name, "combined_data_with_kinematics.pkl"))
            
            for index, row in combined_data_with_kinematics.iterrows():
                trials.append(row.kinematics)
            
            trials = [clean_columns(trial_df).drop(columns='side') for trial_df in trials]
            reference_trial = trials[0]
            aligned_trials = [align_trial(reference_trial, trial) for trial in trials]
            
            # Assuming references2 and references are being populated here from combined_data_with_kinematics or other source
            references.append([trial['side'].iloc[0] for trial in trials])  # Example side reference
            references2.append([trial['condition'].iloc[0] for trial in trials])  # Example condition reference
        else:
            print(f'Skipped patient {patient}!')
        
        overall_dfs.append(aligned_trials)
    
    return references, references2, overall_dfs

# Add additional side and condition information for each trial
def add_side_condition_information(overall_dfs, references, references2):
    sides = [[[inner_table[0]] for inner_table in outer_table] for outer_table in references]
    conditions = [inner_table for inner_table in references2]
    
    for patient in range(len(sides)):
        for trial in range(len(references[patient])):
            if patient not in [17, 23]:  # Skip certain patients if needed
                print(f"Patient {patient}, Trial {trial}")
                overall_dfs[patient][trial]['side'] = sides[patient][trial][0]
                overall_dfs[patient][trial]['condition'] = conditions[patient][trial]

    return overall_dfs

# Function to combine all aligned trials into a single DataFrame
def combine_trials_into_dataframe(time_series):
    all_columns = set()
    for i in range(len(time_series)):
        for j in range(len(time_series[i])):
            all_columns.update(time_series[i][j].columns)

    data = []
    for i in range(len(time_series)):
        for j in range(len(time_series[i])):
            time_series[i][j]['patient_number'] = i
            time_series[i][j]['trial_number'] = j
            df = time_series[i][j]
            missing_columns = all_columns - set(df.columns)
            
            # Add missing columns with NaN values
            for col in missing_columns:
                df[col] = pd.NA
            
            data.append(df)

    df_combined = pd.concat(data, ignore_index=True, join='outer')
    return df_combined

# Main processing workflow
def main():
    patient_dir = r"C:\Users\shrinidhi.velan\Documents\Data"
    references, references2, overall_dfs = process_patients(patient_dir)
    
    # Add side and condition information
    overall_dfs = add_side_condition_information(overall_dfs, references, references2)
    
    # Combine all trials into a single DataFrame
    df_combined = combine_trials_into_dataframe(overall_dfs)
    
    # Save the combined DataFrame
    df_combined.to_csv('df_combined_time_series.csv', index=False)
    print("Combined DataFrame saved as 'df_combined_time_series.csv'")
    
    # Optionally load the saved DataFrame
    df_combined = pd.read_csv('df_combined_time_series.csv')
    return df_combined

# Run the main function
df_combined = main()
