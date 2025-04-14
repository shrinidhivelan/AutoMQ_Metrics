from measures_helper import *
from read_files import *
import pandas as pd
import numpy as np
import os
import pickle

def append_dataframes(df1, df2):
    # Ensure all columns in both DataFrames are the same
    all_columns = df1.columns.union(df2.columns)
    
    # Reindex both DataFrames to include all columns, filling missing values with NaN
    df1 = df1.reindex(columns=all_columns)
    df2 = df2.reindex(columns=all_columns)
    
    # Append the DataFrames
    return pd.concat([df1, df2], ignore_index=True)



def process_patients(folder_names, patient_dir, force_recompute=False):
    first = 0
    main_df = None  # Initialize main DataFrame
    
    for patient_pid in folder_names:
        print(f"Processing Patient: {patient_pid}")
        patient_path = os.path.join(os.getcwd(), patient_pid)

        # Decide whether to reuse or recompute data
        if not force_recompute and os.path.exists(patient_path):
            print("Already done! Loading existing data.")
            combined_data_with_kinematics = pd.read_pickle(os.path.join(patient_path, "combined_data_with_kinematics.pkl"))
            phases_df = pd.read_pickle(os.path.join(patient_path, "phases_df.pkl"))
        
        elif patient_pid not in ['P24', 'P34']:  # Process only if not in the excluded list
            print("Processing from scratch...")
            data_dir = os.path.join(patient_dir, patient_pid, "c3d")
            
            # Initialize patient data
            init_file(patient_pid, data_dir)
            combined_data_with_kinematics = load_dataframe(patient_pid, "combined_data_with_kinematics")
            phases_df = load_dataframe(patient_pid, "phases_df")
        
        else:
            print(f"Skipping {patient_pid} (excluded)")
            continue  # Skip excluded patients

        # Compute kinematic measures
        df = calculate_combined_measures(combined_data_with_kinematics, phases_df)

        # Merge results into main DataFrame
        if first == 0:
            main_df = df
            first = 1
        else:
            main_df = append_dataframes(main_df, df)
    
    return main_df  # Return final merged DataFrame






if __name__ == "__main__":
    

    patient_dir = r"C:\Users\shrinidhi.velan\Documents\Data"

    # List only directories (folders) in the specified directory
    folder_names = [f for f in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, f))]
    # folder_names = ['P01', 'P02']#, 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P17', 'P19', 'P23', 'P24', 'P25', 'P27', 'P28', 'P30', 'P31', 'P34']
    print(folder_names)

    main_df = process_patients(folder_names, patient_dir, True)

    #os.makedirs("overall_patients", exist_ok=True)  # Create folder if it doesn't exist^
    patient_paths = os.path.join("Participants","overall_patients")
    os.makedirs(patient_paths, exist_ok=True)  # Create folder if it doesn't exist
    main_df.to_csv(os.path.join(patient_paths, "overall_measures.csv"), index=False)  # Save DataFrame as CSV