import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from fastdtw import fastdtw

# -------- Utility Functions -------- #
def process_kinematics(kinematics_df):
    new_columns = {}
    detected_sides = set()

    for col in kinematics_df.columns:
        if "L_" in col:
            detected_sides.add("L")
            new_columns[col] = col.replace("L_", "")
        elif "R_" in col:
            detected_sides.add("R")
            new_columns[col] = col.replace("R_", "")

    kinematics_df = kinematics_df.rename(columns=new_columns)
    return kinematics_df

def align_trial(reference, trial):
    aligned_trial = {}
    for col in trial.columns:
        try:
            ref_series = reference[col]
            trial_series = trial[col]

            if len(ref_series) < 2 or len(trial_series) < 2:
                print(f"[Warning] Skipping column '{col}' due to insufficient length.")
                continue

            distance, path = fastdtw(ref_series, trial_series)
            path = np.array(path)

            interp = interp1d(path[:, 1], trial_series.iloc[path[:, 1]],
                              kind='linear', bounds_error=False, fill_value="extrapolate")
            aligned_trial[col] = interp(np.arange(len(ref_series)))
        except Exception as e:
            print(f"[Error] Failed to align column '{col}': {e}")
            continue

    return pd.DataFrame(aligned_trial)


# -------- Main Processing Function -------- #
def process_all_patients(patient_dir_data, patient_current_dir, mode='raw'):
    participants = [f for f in os.listdir(patient_dir_data) if os.path.isdir(os.path.join(patient_dir_data, f))]
    all_data = []

    for participant in tqdm(participants, desc="Processing participants"):
        file_path = os.path.join(patient_current_dir, participant, "combined_data_with_kinematics.pkl")

        try:
            df = pd.read_pickle(file_path).reset_index()
            df["kinematics"] = df["kinematics"].apply(process_kinematics)

            meta_cols = ["participant_id", "trial_number", "side", "condition", "validity"]

            if mode == 'dtw':
                trials = [row.kinematics.drop(columns='side', errors='ignore') for _, row in df.iterrows()]
                if not trials:
                    continue
                reference_trial = trials[0]
                aligned_trials = [align_trial(reference_trial, trial) for trial in trials]

                for i, row in enumerate(df.itertuples()):
                    trial_df = aligned_trials[i]
                    trial_df['side'] = getattr(row, 'side')
                    trial_df['condition'] = getattr(row, 'condition')
                    trial_df['participant_id'] = getattr(row, 'participant_id')
                    trial_df['trial_number'] = getattr(row, 'trial_number')
                    trial_df['validity'] = getattr(row, 'validity')
                    all_data.append(trial_df)

            elif mode == 'raw':
                for _, row in df.iterrows():
                    kinematic_df = row["kinematics"]
                    if not kinematic_df.empty:
                        repeated_metadata = pd.DataFrame([row[meta_cols]] * len(kinematic_df)).reset_index(drop=True)
                        merged_df = pd.concat([repeated_metadata, kinematic_df.reset_index(drop=True)], axis=1)
                        all_data.append(merged_df)

            else:
                raise ValueError("Invalid mode. Use 'dtw' or 'raw'.")

        except FileNotFoundError:
            print(f"⚠️ File not found: {file_path}, skipping...")
        except Exception as e:
            print(f"⚠️ Unexpected error processing {participant}: {e}")

    if all_data:

        df_final = pd.concat(all_data, ignore_index=True)
        df_final['log_movement_smoothness'] = np.log(df_final['movement_smoothness'] + 1)
        df_final['frame_number'] = (df_final.groupby(['participant_id', 'trial_number']).cumcount())
        os.makedirs('data', exist_ok=True)

        if mode == 'dtw':
            df_final.to_pickle(os.path.join("data","aligned_dtw_data.pkl"))
            df_final.to_csv(os.path.join("data","aligned_dtw_data.csv"), index=False)
        elif mode == 'raw':
            df_final.to_pickle(os.path.join("data","raw_appended_data.pkl"))
            df_final.to_csv(os.path.join("data","raw_appended_data.csv"), index=False)

        print(f"✅ Successfully processed data in '{mode}' mode. Total rows: {len(df_final)}")
    else:
        print("❌ No valid data found.")


# -------- Main Entry Point -------- #
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process kinematic data with optional DTW alignment.")
    parser.add_argument('--patient_dir_data', type=str, required=True, help='Directory containing participant folders')
    parser.add_argument('--patient_current_dir', type=str, required=True, help='Directory containing participant folders with created pkl files')
    parser.add_argument('--mode', type=str, choices=['raw', 'dtw'], default='raw', help='Processing mode')
    args = parser.parse_args()

    print(f"▶️ Running unified pipeline in '{args.mode}' mode for directory: {args.patient_dir_data}")
    process_all_patients(args.patient_dir_data, patient_current_dir=args.patient_current_dir, mode=args.mode)


if __name__ == '__main__':
    main()



#python unified_pipeline.py --patient_dir_data "C:\Users\shrinidhi.velan\Documents\Data" --patient_current_dir "C:\Users\shrinidhi.velan\Documents\AutoMQ" --mode dtw
