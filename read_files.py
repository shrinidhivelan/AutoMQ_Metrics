import pandas as pd
import numpy as np
import re
import ezc3d
import os
from measures_helper import *
from scipy.signal import butter, filtfilt

def check_for_nans_in_trials(df_12):
    for i in range(len(df_12)):  # 0 to 80 inclusive 
        try:
            df = df_12.iloc[i].kinematics
            if df.isna().any().any():  # Check if any NaN values exist in the dataframe
                print(f"⚠️ Trial {i}: NaNs found")
                return True
            else:
                print(f"✅ Trial {i}: No NaNs found")
        except Exception as e:
            print(f"⚠️ Error in trial {i}: {e}")
    return False



def phases(combined_data):

    # Butterworth filter function
    def apply_butterworth_filter(data, cutoff=6, fs=100, order=2):
        nyquist = fs / 2.0
        b, a = butter(order, cutoff / nyquist, btype='low')
        return filtfilt(b, a, data)

    # Function to identify phases
    def identify_phases(df, velocity_threshold=20):
        # Determine side-specific markers
        side = df.index.get_level_values('side')[0]  # Access side from index
        hand_marker = 'hand_R' if side == 'R' else 'hand_L'
        glass_marker = 'glass'
        face_marker = 'face'

        # Extract the markers DataFrame
        markers_df = df['markers'].iloc[0]  # Get the markers DataFrame

        # Initialize phases dictionary
        phases = {
            'Reaching': None,
            'Forward Transport': None,
            'Drinking': None,
            'Back Transport': None,
            'Returning': None
        }

        # Calculate hand velocities using vectorization
        frames = markers_df.index.get_level_values('frame').unique()
        hand_positions = markers_df.xs(hand_marker, level='marker')[['x', 'y', 'z']].values
        hand_velocities = np.linalg.norm(hand_positions[1:] - hand_positions[:-1], axis=1) * 100  # Assuming 100 Hz frame rate
        hand_velocities = np.insert(hand_velocities, 0, 0)  # No velocity for the first frame

        # Filter hand velocities
        filtered_hand_velocities = apply_butterworth_filter(hand_velocities)

        # Set reaching_start based on filtered hand velocities
        reaching_start = None
        for i in range(len(filtered_hand_velocities) - 30):
            if filtered_hand_velocities[i] > velocity_threshold and \
            all(filtered_hand_velocities[i:i + 30] > velocity_threshold):
                reaching_start = i
                break

        # Calculate glass velocities
        glass_positions = markers_df.xs(glass_marker, level='marker')[['x', 'y', 'z']].values
        glass_velocities = np.linalg.norm(glass_positions[1:] - glass_positions[:-1], axis=1) * 100  # Assuming 100 Hz frame rate
        glass_velocities = np.insert(glass_velocities, 0, 0)  # No velocity for the first frame

        # Set forward_start based on filtered glass velocities
        forward_start = None
        for i in range(len(glass_velocities) - 30):
            if glass_velocities[i] > 50 and all(glass_velocities[i:i + 30] > 50):
                forward_start = i
                break
                
        phases['Reaching'] = (reaching_start, forward_start)

        # Identify drinking phase using distances to face
        hand_positions = markers_df.xs(hand_marker, level='marker')[['x', 'y', 'z']].values
        face_positions = markers_df.xs(face_marker, level='marker')[['x', 'y', 'z']].values
        face_distances = np.linalg.norm(hand_positions - face_positions, axis=1)

        # Determine steady state distance
        min_distance_index = np.argmin(face_distances)
        steady_state_start = max(min_distance_index - 20, 0)
        steady_state_end = min(min_distance_index + 20, len(face_distances))
        steady_state_distance = np.mean(face_distances[steady_state_start:steady_state_end])
        drinking_distance_threshold = steady_state_distance * 1.2  # Tim Adjust threshold from 1.15 to 1.2

        # Look for the first value after forward_start where d < drinking_distance_threshold
        drinking_start = None
        if forward_start is not None:
            drinking_start = np.where(face_distances[forward_start:] < drinking_distance_threshold)[0]
            if len(drinking_start) > 0:
                drinking_start = drinking_start[0] + forward_start

        phases['Forward Transport'] = (forward_start, drinking_start)

        if drinking_start is not None:
            drinking_end = np.where(face_distances[drinking_start:] > drinking_distance_threshold)[0]
            if len(drinking_end) > 0:
                drinking_end = drinking_end[0] + drinking_start
        else:
            drinking_end = None
        phases['Drinking'] = (drinking_start, drinking_end)

        if drinking_end is not None and not drinking_end.size == 0 : # Tim added the second condition catch the case where drinking_end is empty
            back_transport_start = drinking_end
            back_transport_end = np.where(glass_velocities[drinking_end:] < 50)[0]
            if len(back_transport_end) > 0:
                back_transport_end = back_transport_end[0] + drinking_end
        else:
            back_transport_start = None
            back_transport_end = None
        phases['Back Transport'] = (back_transport_start, back_transport_end)

        if back_transport_end is not None:
            returning_start = back_transport_end
            max_velocity_after_start = np.max(filtered_hand_velocities[returning_start:]) if returning_start < len(filtered_hand_velocities) else 0
            
            # Get the index of the maximum velocity after returning_start
            idx_max_velocity_after_start = returning_start + np.argmax(filtered_hand_velocities[returning_start:]) if returning_start < len(filtered_hand_velocities) else None 
            
            # Now, search for the first index after the max velocity index where the velocity falls below 0.02 * max_velocity
            returning_end = np.where(filtered_hand_velocities[idx_max_velocity_after_start:] < 0.02 * max_velocity_after_start)[0]
            if len(returning_end) > 0:
                returning_end = returning_end[0] + idx_max_velocity_after_start
            else:
                returning_end = len(filtered_hand_velocities) - 1  # Default to end of trial if criteria is not met
        else:
            returning_start = None
            returning_end = len(filtered_hand_velocities) - 1  # Default to end of trial if criteria is not met

        phases['Returning'] = (returning_start, returning_end)

        # Assess trial validity
        valid_trial = (
            all(v is not None for phase in phases.values() for v in phase) and  # Check that all phases are defined
            all(velocity < velocity_threshold for velocity in filtered_hand_velocities[0:10]) and  # Check first 10 velocities for stationarity
            all(start < end for start, end in phases.values() if start is not None and end is not None)  # Check sequential order
        )
        valid_trial = 1 if valid_trial else 0  # Convert boolean to integer (1 for valid, 0 for invalid)

        return phases, valid_trial

    # Dictionary to store phases for each trial
    all_trial_phases = {}

    # Loop through each trial in combined_data
    for (participant_id, trial_number, side, condition), trial_df in combined_data.groupby(['participant_id', 'trial_number', 'side', 'condition']):
        print(trial_number)
        # Check validity
        validity = trial_df.index.get_level_values('validity')[0]  # Access validity from index
        if validity == 0:
            print(f"Skipping trial {trial_number} due to invalidity.")
            continue  # Skip invalid trials

        # Identify phases for each trial
        trial_phases, valid_trial = identify_phases(trial_df)
        if trial_phases is not None:  # Only add to dictionary if phases were successfully identified
            all_trial_phases[(participant_id, trial_number, side, condition)] = {'phases': trial_phases, 'validity': valid_trial}

    # Convert the phases dictionary to a DataFrame for easier handling
    phases_df = pd.DataFrame.from_dict(all_trial_phases, orient='index')

    # Display the DataFrame with phases for all trials
    print("Identified Phases for All Trials:")
    print(phases_df)
    return phases_df

def init_file(patient_pid, data_dir):
    # Read the lab log CSV file
    lab_log_df = pd.read_csv('lab_log.csv')

    # Check the lab for the given PID
    lab_name = lab_log_df.loc[lab_log_df['PID'] == patient_pid, 'lab'].values
    rotate_markers = lab_name[0] == "Hertenstein" if lab_name.size > 0 else False

    # Default marker mapping for renaming and standardizing markers
    DEFAULT_MARKER_MAP = {
        'hand_R': 'body_Rhand',
        'hand_L': 'body_Lhand',
        'elbow_R': 'body_Relbow',
        'elbow_L': 'body_Lelbow',
        'shoulder_R': 'body_Rshoulder',
        'shoulder_L': 'body_Lshoulder',
        'wrist_R': 'wrist_outer_R',
        'wrist_L': 'wrist_outer_L',
        'glass': 'cluster_cup_1',
        'face': 'head',
        'trunk': 'chest',
        'indexfinger_R': 'indexfinger_R',
        'indexfinger_L': 'indexfinger_L',
        'middlefinger_R': 'middlefinger_R',
        'middlefinger_L': 'middlefinger_L',
        'thumb_R': 'thumb_R',
        'thumb_L': 'thumb_L',
        'hip_R': 'hip_R',
        'hip_L': 'hip_L',
        'arm_R': 'arm_R',
        'arm_L': 'arm_L'
    }

    # Default marker mapping for renaming and standardizing markers for P01
    if patient_pid == 'P01':
        DEFAULT_MARKER_MAP = {
            'hand_R': 'body_Rhand',
            'hand_L': 'body_Lhand',
            'elbow_R': 'body_Relbow',
            'elbow_L': 'body_Lelbow',
            'shoulder_R': 'body_Rshoulder',
            'shoulder_L': 'body_Lshoulder',
            'wrist_R': 'body_Rwrist',
            'wrist_L': 'body_Lwrist',
            'glass': 'cluster_cup_1',
            'face': 'body_head',
            'trunk': 'body_chest',
            'indexfinger_R': 'body_Rindexfinger',
            'indexfinger_L': 'body_Lindexfinger',
            'middlefinger_R': 'body_Rmiddlefinger',
            'middlefinger_L': 'body_Lmiddlefinger',
            'thumb_R': 'body_Rthumb',
            'thumb_L': 'body_Lthumb',
            'hip_R': 'hip_R',
            'hip_L': 'hip_L',
            'arm_R': 'body_Rarm',
            'arm_L': 'body_Larm'
        }

    # Default marker mapping for renaming and standardizing markers for P19
    if patient_pid == 'P19':
        DEFAULT_MARKER_MAP = {
            'hand_R': 'body_Rhand',
            'hand_L': 'body_Lhand',
            'elbow_R': 'body_Relbow',
            'elbow_L': 'body_Lelbow',
            'shoulder_R': 'body_Rshoulder',
            'shoulder_L': 'body_Lshoulder',
            'wrist_R': 'wrist_outer_R',
            'wrist_L': 'wirst_outer_L', # Typo in the original mapping
            'glass': 'cluster_cup_1',
            'face': 'head',
            'trunk': 'chest',
            'indexfinger_R': 'indexfinger_R',
            'indexfinger_L': 'indexfinger_L',
            'middlefinger_R': 'middlefinger_R',
            'middlefinger_L': 'middlefinger_L',
            'thumb_R': 'thumb_R',
            'thumb_L': 'thumb_L',
            'hip_R': 'hip_R',
            'hip_L': 'hip_L',
            'arm_R': 'arm_R',
            'arm_L': 'arm_L'
        }


    # Essential markers for validity checking only (use standardized names here)
    VALIDITY_MARKERS = [
        'hand_R', 'hand_L', 'elbow_R', 'elbow_L', 'shoulder_R', 'shoulder_L',
        'wrist_R', 'wrist_L', 'glass', 'face', 'trunk'
    ]

    # Function to map markers based on side and standardize names
    def get_standardized_marker_map(side, marker_map=DEFAULT_MARKER_MAP):
        return {
            'hand': marker_map['hand_R'] if side == 'R' else marker_map['hand_L'],
            'elbow': marker_map['elbow_R'] if side == 'R' else marker_map['elbow_L'],
            'shoulder': marker_map['shoulder_R'] if side == 'R' else marker_map['shoulder_L'],
            'wrist': marker_map['wrist_R'] if side == 'R' else marker_map['wrist_L'],
            'glass': marker_map['glass'],
            'face': marker_map['face'],
            'trunk': marker_map['trunk']
        }

    # Function to load a C3D file and return a structured DataFrame with markers only
    # Function to load a C3D file and return a structured DataFrame with markers only
    def load_trial(file_path, marker_map=DEFAULT_MARKER_MAP):
        # Extract metadata from filename
        match = re.match(r'trial_(\d+)_([RL])_(affected|unaffected)\.c3d', os.path.basename(file_path))
        if match:
            trial_number = int(match.group(1))
            side = match.group(2)
            condition = match.group(3)
        else:
            raise ValueError("Filename does not match expected format: trial_#_Side_Condition.c3d")

        try:
            # Load the C3D file
            c3d = ezc3d.c3d(file_path)

            # Standardize marker names based on provided mapping
            standardized_marker_map = get_standardized_marker_map(side, marker_map)

            # Extract marker data
            points = c3d['data']['points'][:3, :, :].transpose(2, 1, 0)  # Reshape to (num_frames, num_markers, 3)
            frames = range(points.shape[0])
            markers = c3d['parameters']['POINT']['LABELS']['value']

            # Create DataFrame for marker coordinates
            markers_df = pd.DataFrame(
                points.reshape(-1, 3),
                index=pd.MultiIndex.from_product([frames, markers], names=['frame', 'marker']),
                columns=['x', 'y', 'z']
            )

            # Rename markers according to the standard map
            renamed_df = markers_df.reset_index()
            renamed_df['marker'] = renamed_df['marker'].apply(
                lambda m: next((k for k, v in marker_map.items() if v == m), m)
            )
            renamed_df = renamed_df.set_index(['frame', 'marker']).sort_index()

            # Rotate markers if necessary # they will be rotated to match COS of P01
            if rotate_markers:
                print(f"Rotating markers for trial {trial_number} due to lab Hertenstein.")
                # Rotation matrix for -90 degrees around the Z-axis
                angle = -90  # degrees
                theta = np.radians(angle)  # convert to radians
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                original_positions = renamed_df[['x', 'y', 'z']].values  # Get all positions in a single array
                rotated_positions = original_positions @ rotation_matrix.T  # Use the transpose of the rotation matrix
                renamed_df[['x', 'y', 'z']] = rotated_positions

            # Check validity based on essential markers
            nan_free = 1
            for marker in VALIDITY_MARKERS:
                if marker in renamed_df.index.get_level_values('marker'):
                    if renamed_df.loc[pd.IndexSlice[:, marker], :].isna().values.any():
                        nan_free = 0
                        break
                else:
                    nan_free = 0
                    break

            # Create trial-level DataFrame with metadata and validity flag
            trial_df = pd.DataFrame({
                'trial_number': [trial_number],
                'side': [side],
                'condition': [condition],
                'validity': [nan_free],
                'markers': [renamed_df],  # Store the original marker data only
            })
            
        except (OSError, ValueError) as e:
            # Handle empty or invalid C3D files
            print(f"Error loading trial file {file_path}: {e}")
            # Set validity flag to 0 (invalid)
            trial_df = pd.DataFrame({
                'trial_number': [trial_number],
                'side': [side],
                'condition': [condition],
                'validity': [0],  # Mark as invalid
                'markers': [None],  # No valid data
            })

        return trial_df


    # Load each trial and append it to a list of DataFrames
    all_trials = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".c3d"):
            file_path = os.path.join(data_dir, filename)
            trial_df = load_trial(file_path)
            trial_df['participant_id'] = patient_pid  # Use the manually set patient PID
            all_trials.append(trial_df)

    # Concatenate all trials into one DataFrame
    combined_data = pd.concat(all_trials, ignore_index=True)

    # Set 'participant_id' and trial information as higher-level indices
    combined_data.set_index(['participant_id', 'trial_number', 'side', 'condition', 'validity'], inplace=True)
    combined_data.sort_index(level='trial_number', inplace=True)

    # Display the resulting structure
    print(combined_data.head())


    all_trials_with_kinematics = []

    # Calculate kinematics for each valid trial in combined_data
    for idx, row in combined_data.iterrows():
        # Unpack the necessary information
        validity = idx[-1]  # Validity is the last level in the index
        if validity == 0:
            continue  # Skip trials marked as invalid

        # Access markers DataFrame
        markers_df = row['markers']  # Change to row['markers_filtered'] to use filtered data
        side = idx[2]  # Side is the third level in the index

        # Calculate kinematics and add to trial DataFrame if successful
        kinematics_df = calculate_kinematics(markers_df, side)
        row_with_kinematics = row.copy()
        row_with_kinematics['kinematics'] = kinematics_df  # Store kinematics DataFrame as nested DataFrame

        # Add the index information explicitly as columns in the row data
        row_with_kinematics['participant_id'], row_with_kinematics['trial_number'], row_with_kinematics['side'], row_with_kinematics['condition'], row_with_kinematics['validity'] = idx

        # Append row with nested kinematics to all_trials_with_kinematics
        all_trials_with_kinematics.append(row_with_kinematics)

    # Create a DataFrame from the list of rows, explicitly specifying index columns
    combined_data_with_kinematics = pd.DataFrame(all_trials_with_kinematics)
    combined_data_with_kinematics.set_index(['participant_id', 'trial_number', 'side', 'condition', 'validity'], inplace=True)

    phases_df = phases(combined_data)

    ## Before saving the file, check for NaN values 
    # Call the function
    there_is_nan = check_for_nans_in_trials(combined_data_with_kinematics)
    print(there_is_nan)

    save_dataframe(combined_data_with_kinematics, patient_pid, "combined_data_with_kinematics")
    save_dataframe(phases_df, patient_pid, "phases_df")

