import matplotlib.pyplot
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import entropy
import os
import pickle
import re
#import pyentrp
#from pyentrp.entropy import approximate_entropy
from scipy.stats import entropy as shannon_entropy
from pyentrp.entropy import sample_entropy
from scipy.signal import welch


def save_dataframe(df, participant_id, file_name):
    os.makedirs('data', exist_ok=True)
    folder_name = f"{participant_id}"
    participants_path = os.path.join('data','Participants', folder_name)

    os.makedirs(participants_path, exist_ok=True)  # Create the folder if it doesn't exist
    
    file_path = os.path.join(participants_path, f"{file_name}.pkl")
    
    with open(file_path, 'wb') as f:
        pickle.dump(df, f)
    print(f"DataFrame saved to {file_path}")

# Function to load DataFrame from a pickle file
def load_dataframe(participant_id, file_name):
    file_path = os.path.join('Participants',participant_id, f"{file_name}.pkl")
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        print(f"DataFrame loaded from {file_path}")
        return df
    else:
        print(f"No file found at {file_path}")
        return None

# Butterworth filter function
def apply_butterworth_filter(data, cutoff=6, fs=100, order=2):
    nyquist = fs / 2.0
    b, a = butter(order, cutoff / nyquist, btype='low')
    return filtfilt(b, a, data, axis=0)

# Calculate 3D angle between two vectors
def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = dot_product / norm_product
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle_rad)

# Compute elbow angle
def calculate_elbow_angle(shoulder, elbow, wrist):
    upper_arm = shoulder - elbow
    forearm = wrist - elbow
    return calculate_angle(upper_arm, forearm)

# Compute shoulder angles
def calculate_shoulder_angles(shoulder, elbow):
    upper_arm_vector = shoulder - elbow
    z_axis = np.array([0, 0, 1])
    pure_shoulder_angle = calculate_angle(upper_arm_vector, z_axis)
    
    upper_arm_proj_sagittal = np.array([0, upper_arm_vector[1], upper_arm_vector[2]])
    shoulder_flexion = calculate_angle(upper_arm_proj_sagittal, z_axis)
    
    upper_arm_proj_coronal = np.array([upper_arm_vector[0], 0, upper_arm_vector[2]])
    shoulder_abduction = calculate_angle(upper_arm_proj_coronal, z_axis)
    
    return pure_shoulder_angle, shoulder_flexion, shoulder_abduction

# Compute end-effector velocity
def calculate_end_effector_velocity(hand_positions, frame_rate=100):
    velocity = np.linalg.norm(hand_positions[1:] - hand_positions[:-1], axis=1) * frame_rate
    return np.insert(velocity, 0, 0)

# Compute jerk (smoothness)
def calculate_jerk(hand_positions, frame_rate=100):
    dt = 1.0 / frame_rate
    velocity_vectors = np.diff(hand_positions, axis=0) / dt
    acceleration_vectors = np.diff(velocity_vectors, axis=0) / dt
    jerk_vectors = np.diff(acceleration_vectors, axis=0) / dt
    jerk_magnitude = np.linalg.norm(jerk_vectors, axis=1)
    return np.concatenate((np.zeros(3), jerk_magnitude))


# Shannon Entropy calculation
def shannon_entropy(signal, bins=10):
    hist, bin_edges = np.histogram(signal, bins=bins, density=True)
    prob_dist = hist / np.sum(hist)
    return entropy(prob_dist)

# Approximate Entropy (ApEn) calculation
def approximate_entropy(signal, m=2, r=0.2):
    return approximate_entropy(signal, m, r)

# Sample Entropy (SampEn) calculation
def entropy_sample(signal, m=2, r=0.2):
    return sample_entropy(signal, m, r)

# Compute trajectory metrics
def calculate_trajectory_metrics(hand_positions):
    path_length = np.sum(np.linalg.norm(np.diff(hand_positions, axis=0), axis=1))
    jerk_magnitude = calculate_jerk(hand_positions)
    movement_smoothness = np.sum(jerk_magnitude ** 2)
    return path_length, movement_smoothness

# Compute trunk displacement
def calculate_trunk_displacement(trunk_positions):
    return trunk_positions[0] - trunk_positions

# Example kinematic data calculation functions (like velocity, acceleration)
def calculate_velocity(positions, frame_rate=100):
    velocity = np.linalg.norm(positions[1:] - positions[:-1], axis=1) * frame_rate
    return np.insert(velocity, 0, 0)

# Compute kinematics for a trial
# Main function to calculate all kinematics for each trial
def calculate_kinematics(df, side):
    # Set side-specific markers
    markers = {
        'hand': 'hand_R' if side == 'R' else 'hand_L',
        'elbow': 'elbow_R' if side == 'R' else 'elbow_L',
        'shoulder': 'shoulder_R' if side == 'R' else 'shoulder_L',
        'wrist': 'wrist_R' if side == 'R' else 'wrist_L',
        'trunk': 'trunk'
    }

    # Get frame indices
    frames = df.index.get_level_values('frame').unique()

    # Get marker positions for shoulder, elbow, wrist, and hand in bulk
    shoulder_positions = df.loc[pd.IndexSlice[:, markers['shoulder']], ['x', 'y', 'z']].values
    elbow_positions = df.loc[pd.IndexSlice[:, markers['elbow']], ['x', 'y', 'z']].values
    wrist_positions = df.loc[pd.IndexSlice[:, markers['wrist']], ['x', 'y', 'z']].values
    hand_positions = df.loc[pd.IndexSlice[:, markers['hand']], ['x', 'y', 'z']].values
    trunk_positions = df.loc[pd.IndexSlice[:, markers['trunk']], 'y'].values

    # Calculate angles
    elbow_angles = np.array([calculate_elbow_angle(shoulder, elbow, wrist) for shoulder, elbow, wrist in zip(shoulder_positions, elbow_positions, wrist_positions)])
    pure_shoulder_angles, shoulder_flexions, shoulder_abductions = zip(*[calculate_shoulder_angles(shoulder, elbow) for shoulder, elbow in zip(shoulder_positions, elbow_positions)])

    dt = 1.0 / 100  # 100 Hz sampling rate - trajectory metrics


    # End-effector (hand) velocity calculation
    end_effector_velocity = np.linalg.norm(hand_positions[1:] - hand_positions[:-1], axis=1) * 100  # Assuming 100 Hz frame rate
    end_effector_velocity = np.insert(end_effector_velocity, 0, 0)  # No velocity for the first frame

    # End-effector (hand) acceleration calculation
    # end_effector_acceleration = np.linalg.norm(end_effector_velocity[1:] - end_effector_velocity[:-1], axis=1) * 100  # Assuming 100 Hz frame rate
    end_effector_acceleration = np.linalg.norm(end_effector_velocity[1:] - end_effector_velocity[:-1]) * 100
    end_effector_acceleration = np.insert(end_effector_acceleration, 0, 0)  # No velocity for the first frame


    # Trunk displacement
    initial_trunk_y = trunk_positions[0]
    trunk_displacement = initial_trunk_y - trunk_positions  # Positive values indicate forward movement


    # NEW: Add raw hand positions as separate columns
    hand_x, hand_y, hand_z = hand_positions[:, 0], hand_positions[:, 1], hand_positions[:, 2]


    # Add a jerk measure :
    """ 
    #### JERK calculated from hand movements ####
    
    # Jerk is the third derivative of position with respect to time.
    # We'll compute finite differences along the time axis.
    """
    
    dt = 1.0 / 100  # 100 Hz sampling rate
    
    #1) First derivative: velocity vectors
    velocity_vectors = np.diff(hand_positions, axis=0) / dt

    #2) Second derivative: acceleration vectors
    acceleration_vectors = np.diff(velocity_vectors, axis=0) / dt
    
    #3) Third derivative: jerk vectors
    jerk_vectors = np.diff(acceleration_vectors, axis=0) / dt
    
    # Compute the magnitude of jerk vectors for each time step
    jerk_magnitude = np.linalg.norm(jerk_vectors, axis=1)
    # Pad jerk_magnitude so that its length matches the number of frames.
    # We lose 3 frames when taking three differences.
    jerk_magnitude = np.concatenate((np.zeros(3), jerk_magnitude))

    # let's add trajectory metrics :
    path_length = np.sum(np.linalg.norm(np.diff(hand_positions, axis=0), axis=1))
    movement_smoothness = np.sum(jerk_magnitude ** 2)

    # CURVATURE: frame-wise curvature
    def compute_curvature(x, y, z):
        dx = np.gradient(x)
        dy = np.gradient(y)
        dz = np.gradient(z)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        ddz = np.gradient(dz)

        num = np.sqrt((dy * ddz - dz * ddy)**2 + (dz * ddx - dx * ddz)**2 + (dx * ddy - dy * ddx)**2)
        denom = (dx**2 + dy**2 + dz**2)**1.5 + 1e-8
        return num / denom
    
    curvature = compute_curvature(hand_x, hand_y, hand_z)

    # SPECTRAL ARC LENGTH (SAL) in rolling windows of velocity
    def spectral_arc_length(signal, fs=100):
        freqs, power = welch(signal, fs=fs, nperseg=len(signal))
        normalized_power = power / np.sum(power)
        sal = -np.sum(np.sqrt(1 + np.diff(normalized_power)**2))
        return sal
    
    window_size = 20
    sal_velocity = pd.Series(end_effector_velocity).rolling(window=window_size, center=True, min_periods=10).apply(lambda x: spectral_arc_length(x), raw=True)

    
    # VELOCITY CHANGE (optional)
    velocity_change = np.insert(np.diff(end_effector_velocity), 0, 0)


    """
    Acceleration magnitude : (new!)
    """
    acceleration_magnitude = np.linalg.norm(acceleration_vectors, axis=1)

    """
    Range Of motion - for joints such as shoulder or elbow - careful as this is in fact a summary statistics
    """
    shoulder_rom = np.max(pure_shoulder_angles) - np.min(pure_shoulder_angles)
    elbow_rom = np.max(elbow_angles) - np.min(elbow_angles)

    """ 
    Entropy : Shannon, Approximate and Sample Entropy :
    """
    hand_velocity = calculate_velocity(hand_positions)


    # Calculate Shannon Entropy for hand velocity
    hand_velocity_entropy = shannon_entropy(hand_velocity)

    # Calculate Approximate Entropy for wrist position
    #hand_position_apen = approximate_entropy(hand_velocity)

    # Calculate Sample Entropy for trunk velocity
    hand_velocity_sampen = entropy_sample(hand_velocity)


    """ 
    # Create DataFrame for kinematics with frame index
    kinematics_df = pd.DataFrame({
        f'{markers["hand"]}_velocity': end_effector_velocity,
        f'{markers["trunk"]}_displacement': trunk_displacement,
        'elbow_angle': elbow_angles,
        'pure_shoulder_angle': pure_shoulder_angles,
        'shoulder_flexion': shoulder_flexions,
        'shoulder_abduction': shoulder_abductions
    }, index=frames)
    """

    kinematics_df = pd.DataFrame({
    f'{markers["hand"]}_velocity': end_effector_velocity,
    f'{markers["hand"]}_acceleration': end_effector_acceleration,
    f'{markers["trunk"]}_displacement': trunk_displacement,
    'elbow_angle': elbow_angles,
    'pure_shoulder_angle': pure_shoulder_angles,
    'shoulder_flexion': shoulder_flexions,
    'shoulder_abduction': shoulder_abductions,
    #'shoulder_ROM': shoulder_rom,
    #'elbow_ROM': elbow_rom,
    #'hand_velocity_entropy' : hand_velocity_entropy,
    #'hand_velocity_sampen' : hand_velocity_sampen,
    # New columns for hand marker positions:
    f'{markers["hand"]}_x': hand_x,
    f'{markers["hand"]}_y': hand_y,
    f'{markers["hand"]}_z': hand_z,
    # New column for instantaneous jerk magnitude:
    f'{markers["hand"]}_jerk': jerk_magnitude,
    'path_length': path_length,
    'movement_smoothness': movement_smoothness,
    'curvature': curvature,
    'sal_velocity': sal_velocity,
    'velocity_change': velocity_change,
    #'acceleration_magnitude': acceleration_magnitude
    }, index=frames)

    return kinematics_df

# Process trials
def process_trials(combined_data):
    all_trials_with_kinematics = []
    for idx, row in combined_data.iterrows():
        if idx[-1] == 0:
            continue
        
        markers_df = row['markers']
        side = idx[2]
        kinematics_df = calculate_kinematics(markers_df, side)
        row_with_kinematics = row.copy()
        row_with_kinematics['kinematics'] = kinematics_df
        row_with_kinematics['participant_id'], row_with_kinematics['trial_number'], row_with_kinematics['side'], row_with_kinematics['condition'], row_with_kinematics['validity'] = idx
        all_trials_with_kinematics.append(row_with_kinematics)
    
    combined_data_with_kinematics = pd.DataFrame(all_trials_with_kinematics)
    combined_data_with_kinematics.set_index(['participant_id', 'trial_number', 'side', 'condition', 'validity'], inplace=True)
    return combined_data_with_kinematics

def calculate_movement_units(velocities):
    inverted_velocities = -velocities
    min_peaks, _ = find_peaks(inverted_velocities)
    max_peaks, _ = find_peaks(velocities)

    movement_units = 0
    amplitude_threshold = 20  # mm/s
    time_threshold = 15  # corresponding to 150 ms at 100 Hz

    for min_index in min_peaks:
        for max_index in max_peaks:
            if max_index > min_index:  # Only consider maxima after the minimum
                if velocities[max_index] > amplitude_threshold:
                    if (max_index - min_index) >= time_threshold:
                        movement_units += 1
                    break  # Stop after finding the first valid maximum

    return movement_units

def compute_derivative(data, dt):
    return np.diff(data, axis=0) / dt

def calculate_jerk_metrics(position_data, fs=100, filter_data=True):
    """
    Calculates jerk metrics from a given time series of position data.
    
    Parameters:
        position_data (np.ndarray): Array of shape (N, D) where N is the number of frames 
                                    and D is the spatial dimension (e.g. 3 for x, y, z).
        fs (float): Sampling frequency in Hz.
        filter_data (bool): Whether to apply a Butterworth filter to smooth the data.
    
    Returns:
        dict: Dictionary containing mean jerk, peak jerk, and normalized jerk.
    """
    dt = 1.0 / fs
    
    # Optionally filter the position data to reduce noise
    if filter_data:
        position_data = apply_butterworth_filter(position_data, fs=fs)
    
    # First derivative: velocity
    velocity = compute_derivative(position_data, dt)
    
    # Second derivative: acceleration
    acceleration = compute_derivative(velocity, dt)
    
    # Third derivative: jerk
    jerk = compute_derivative(acceleration, dt)
    
    # Compute the magnitude of jerk at each time step (if data is multidimensional)
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    
    # Compute summary metrics:
    mean_jerk = np.mean(np.abs(jerk_magnitude))
    peak_jerk = np.max(np.abs(jerk_magnitude))
    movement_duration = position_data.shape[0] * dt
    normalized_jerk = np.sum(np.abs(jerk_magnitude)) / (movement_duration**3)
    
    return {
        'mean_jerk': mean_jerk,
        'peak_jerk': peak_jerk,
        'normalized_jerk': normalized_jerk
    }

def calculate_trajectory_metrics(position_data, fs=100, filter_data=True):
    """
    Calculates movement trajectory metrics from a given time series of position data.

    Parameters:
        position_data (np.ndarray): Array of shape (N, D), where N is the number of frames 
                                    and D is the spatial dimension (e.g., 3 for x, y, z).
        fs (float): Sampling frequency in Hz.
        filter_data (bool): Whether to apply a Butterworth filter to smooth the data.

    Returns:
        dict: Dictionary containing path length, movement efficiency, and smoothness.
    """
    dt = 1.0 / fs

    # Optionally filter the position data to reduce noise
    if filter_data:
        position_data = apply_butterworth_filter(position_data, fs=fs)

    # Compute displacement vectors between consecutive frames
    displacement_vectors = np.diff(position_data, axis=0)

    # Compute Euclidean distances between consecutive positions
    segment_lengths = np.linalg.norm(displacement_vectors, axis=1)

    # Compute total path length
    path_length = np.sum(segment_lengths)

    # Compute movement efficiency (straight-line distance / total path length)
    start_position = position_data[0]
    end_position = position_data[-1]
    straight_line_distance = np.linalg.norm(end_position - start_position)
    movement_efficiency = straight_line_distance / path_length if path_length > 0 else 0

    # Compute smoothness using velocity profile (mean absolute velocity change)
    velocity = compute_derivative(position_data, dt)
    velocity_magnitude = np.linalg.norm(velocity, axis=1)
    smoothness = -np.mean(np.abs(np.diff(velocity_magnitude)))  # Negative for interpretation: higher = smoother

    return {
        'path_length': path_length,
        'movement_efficiency': movement_efficiency,
        'smoothness': smoothness
    }

"""
Murphy measures and jerk measures :

def calculate_trajectory_metrics(position_data, fs=100, filter_data=True):
    dt = 1.0 / fs
    if filter_data:
        position_data = apply_butterworth_filter(position_data, fs=fs)
    displacement_vectors = np.diff(position_data, axis=0)
    segment_lengths = np.linalg.norm(displacement_vectors, axis=1)
    path_length = np.sum(segment_lengths)
    movement_efficiency = np.linalg.norm(position_data[-1] - position_data[0]) / path_length if path_length > 0 else 0
    velocity = compute_derivative(position_data, dt)
    velocity_magnitude = np.linalg.norm(velocity, axis=1)
    smoothness = -np.mean(np.abs(np.diff(velocity_magnitude)))
    return {
        'path_length': path_length,
        'movement_efficiency': movement_efficiency,
        'smoothness': smoothness
    }

def calculate_trajectory_metrics(position_data, fs=100, filter_data=True):
    dt = 1.0 / fs
    if filter_data:
        position_data = apply_butterworth_filter(position_data, fs=fs)
    displacement_vectors = np.diff(position_data, axis=0)
    segment_lengths = np.linalg.norm(displacement_vectors, axis=1)
    path_length = np.sum(segment_lengths)
    movement_efficiency = np.linalg.norm(position_data[-1] - position_data[0]) / path_length if path_length > 0 else 0
    velocity = compute_derivative(position_data, dt)
    velocity_magnitude = np.linalg.norm(velocity, axis=1)
    smoothness = -np.mean(np.abs(np.diff(velocity_magnitude)))
    return {
        'path_length': path_length,
        'movement_efficiency': movement_efficiency,
        'smoothness': smoothness
    }
"""


def calculate_basic_statistics(combined_data_with_kinematics):
    statistics_list = []

    for (participant_id, trial_number, side, condition), trial_df in combined_data_with_kinematics.groupby(
        ['participant_id', 'trial_number', 'side', 'condition']):
        
        # Extract kinematics DataFrame
        if trial_df['kinematics'].values[0] is None:
            print(f"Skipping empty kinematics data for trial {trial_number}")
            continue
        
        kinematics_df = pd.DataFrame(trial_df['kinematics'].values[0])  # Extract first row safely

        # Ensure kinematics_df is not empty
        if kinematics_df.empty:
            print(f"Skipping empty kinematics data for trial {trial_number}")
            continue

        # Standardize column names by removing "_L" and "_R"
        standardized_columns = {col: re.sub(r'(_L|_R)', '', col) for col in kinematics_df.columns}
 
        kinematics_df.rename(columns=standardized_columns, inplace=True)

        # Merge left and right columns by averaging their values
        kinematics_df = kinematics_df.groupby(level=0, axis=1).mean()

        # Initialize stats dictionary
        stats_dict = {
            'participant_id': participant_id,
            'trial_number': trial_number,
            'side': side,
            'condition': condition
        }

        # Compute statistics for each kinematic variable
        for column in kinematics_df.columns:
            stats_dict[f"Mean_{column}"] = kinematics_df[column].mean()
            stats_dict[f"Median_{column}"] = kinematics_df[column].median()
            stats_dict[f"Std_{column}"] = kinematics_df[column].std()
            stats_dict[f"Max_{column}"] = kinematics_df[column].max()
            stats_dict[f"Min_{column}"] = kinematics_df[column].min()
            stats_dict[f"Range_{column}"] = stats_dict[f"Max_{column}"] - stats_dict[f"Min_{column}"]

        # Append statistics to list
        statistics_list.append(stats_dict)

    # Convert list to DataFrame
    statistics_df = pd.DataFrame(statistics_list)

    # Fill NaNs with 0 (optional)
    statistics_df.fillna(0, inplace=True)

    return statistics_df



def compute_shannon_entropy(signal, bins=10):
    hist, _ = np.histogram(signal, bins=bins, density=True)
    prob_dist = hist / np.sum(hist)
    return shannon_entropy(prob_dist)

def compute_sample_entropy(signal, m=2, r=None):
    if r is None:
        r = 0.2 * np.std(signal)
    return sample_entropy(signal, m, r)[0]  # [0] gives the result for a single time series



def calculate_spectral_arc_length(trajectory):
    """ Compute the SPARC : Spectral Arc length """
    # Perform Fourier transform
    freqs = np.fft.fftfreq(len(trajectory))
    fft_vals = np.fft.fft(trajectory)
    
    # Calculate the Spectral Arc Length as the sum of magnitudes
    sal = np.sum(np.abs(fft_vals))
    return sal

def calculate_index_of_curvature(trajectory):
    # Calculate the second derivative (curvature)
    first_derivative = np.diff(trajectory)
    second_derivative = np.diff(first_derivative)
    
    # Index of Curvature is the sum of squared second derivatives
    ioc = np.sum(second_derivative ** 2)
    return ioc



def calculate_combined_measures(combined_data_with_kinematics, phases_df, fs=100):
    measures_list = []
    
    for (participant_id, trial_number, side, condition), trial_df in combined_data_with_kinematics.groupby(
        ['participant_id', 'trial_number', 'side', 'condition']):
        
        validity = trial_df.index.get_level_values('validity')[0]
        if validity == 0:
            continue
        
        phases = phases_df.loc[(participant_id, trial_number, side, condition)]
        if phases['validity'] == 0:
            continue
        
        kinematics_df = trial_df['kinematics']
        hand_velocity_column = 'hand_R_velocity' if side == 'R' else 'hand_L_velocity'
        hand_velocities = kinematics_df.iloc[0][hand_velocity_column].values
        filtered_hand_velocities = apply_butterworth_filter(hand_velocities)

        # Calculate acceleration magnitude
        acceleration_vectors = np.gradient(np.gradient(filtered_hand_velocities)) * (fs ** 2)  # second derivative of velocity
        acceleration_magnitude = np.linalg.norm(acceleration_vectors)
        
        peak_velocity = np.max(filtered_hand_velocities[phases['phases']['Reaching'][0]:phases['phases']['Reaching'][1]])
        total_movement_time = (phases['phases']['Returning'][1] - phases['phases']['Reaching'][0]) / 100
        peak_velocity_index = np.argmax(filtered_hand_velocities[phases['phases']['Reaching'][0]:phases['phases']['Reaching'][1]]) + phases['phases']['Reaching'][0]
        time_to_peak_velocity = peak_velocity_index / 100
        time_to_peak_velocity_percent = (peak_velocity_index - phases['phases']['Reaching'][0]) / (phases['phases']['Reaching'][1] - phases['phases']['Reaching'][0]) * 100
        
        first_peak_indices, _ = find_peaks(filtered_hand_velocities[phases['phases']['Reaching'][0]:phases['phases']['Reaching'][1]])
        if len(first_peak_indices) > 0:
            first_peak_index = first_peak_indices[0] + phases['phases']['Reaching'][0]
            time_to_first_peak_velocity = first_peak_index / 100
            time_to_first_peak_velocity_percent = (first_peak_index - phases['phases']['Reaching'][0]) / (phases['phases']['Reaching'][1] - phases['phases']['Reaching'][0]) * 100
        else:
            time_to_first_peak_velocity = None
            time_to_first_peak_velocity_percent = None
        
        relevant_phases = ['Reaching', 'Forward Transport', 'Back Transport', 'Returning']
        movement_units = sum(calculate_movement_units(filtered_hand_velocities[phases['phases'][phase][0]:phases['phases'][phase][1]]) for phase in relevant_phases)
        
        elbow_angle = kinematics_df.iloc[0]['elbow_angle'].values
        # Entropy-based features
        try:
            shannon_entropy_hand_velocity = compute_shannon_entropy(filtered_hand_velocities)
        except:
            shannon_entropy_hand_velocity = None

        try:
            sample_entropy_hand_velocity = compute_sample_entropy(filtered_hand_velocities)
        except:
            sample_entropy_hand_velocity = None

        try:
            sample_entropy_elbow_angle = compute_sample_entropy(elbow_angle)
        except:
            sample_entropy_elbow_angle = None


        filtered_elbow_angle = apply_butterworth_filter(elbow_angle)
        peak_elbow_angular_velocity = np.max(np.abs(np.diff(filtered_elbow_angle[phases['phases']['Reaching'][0]:phases['phases']['Reaching'][1]]) * 100))
        
        shoulder_flexion = kinematics_df.iloc[0]['shoulder_flexion'].values
        shoulder_abduction = kinematics_df.iloc[0]['shoulder_abduction'].values
        max_shoulder_flexion = np.max(shoulder_flexion[phases['phases']['Reaching'][0]:phases['phases']['Drinking'][1]])
        max_shoulder_abduction = np.max(shoulder_abduction[phases['phases']['Reaching'][0]:phases['phases']['Drinking'][1]])
  
        # Range of motion (ROM)
        shoulder_rom = np.max(shoulder_flexion) - np.min(shoulder_flexion)  # Optional: use filtered values if needed
        elbow_rom = np.max(elbow_angle) - np.min(elbow_angle)
        
        interjoint_coordination = np.corrcoef(shoulder_flexion[phases['phases']['Reaching'][0]:phases['phases']['Reaching'][1]], elbow_angle[phases['phases']['Reaching'][0]:phases['phases']['Reaching'][1]])[0, 1]
        
        trunk_displacement = kinematics_df.iloc[0]['trunk_displacement'].values
        max_trunk_displacement = np.max(np.abs(trunk_displacement))
        
        pos_keys = ['hand_R_x', 'hand_R_y', 'hand_R_z'] if side == 'R' else ['hand_L_x', 'hand_L_y', 'hand_L_z']
        if all(key in kinematics_df.iloc[0].keys() for key in pos_keys):
            hand_positions = np.column_stack([kinematics_df.iloc[0][key].values for key in pos_keys])
            jerk_metrics = calculate_jerk_metrics(hand_positions, fs=fs, filter_data=True)
        else:
            jerk_metrics = {'mean_jerk': None, 'peak_jerk': None, 'normalized_jerk': None}

        # SPARC and Index of curvature :
        # Calculate Spectral Arc Length (SAL)
        sal = calculate_spectral_arc_length(filtered_hand_velocities)
        
        # Calculate Index of Curvature (IoC)
        ioc = calculate_index_of_curvature(filtered_hand_velocities)
        
        
        measures = {
            'participant_id': participant_id,
            'trial_number': trial_number,
            'side': side,
            'condition': condition,
            'total_movement_time': total_movement_time,
            'peak_velocity': peak_velocity,
            'time_to_peak_velocity': time_to_peak_velocity,
            'time_to_first_peak_velocity': time_to_first_peak_velocity,
            'time_to_peak_velocity_percent': time_to_peak_velocity_percent,
            'time_to_first_peak_velocity_percent': time_to_first_peak_velocity_percent,
            'number_of_movement_units': movement_units,
            'interjoint_coordination': interjoint_coordination,
            'max_trunk_displacement': max_trunk_displacement,
            'max_shoulder_flexion': max_shoulder_flexion,
            'max_shoulder_abduction': max_shoulder_abduction,
            'max_elbow_angle': np.max(elbow_angle),
            'peak_elbow_angular_velocity': peak_elbow_angular_velocity,
            'mean_jerk': jerk_metrics['mean_jerk'],
            'peak_jerk': jerk_metrics['peak_jerk'],
            'normalized_jerk': jerk_metrics['normalized_jerk'], 
            'shoulder_rom': shoulder_rom,
            'elbow_rom': elbow_rom,
            'acceleration_magnitude': acceleration_magnitude,
            'shannon_entropy_hand_velocity': shannon_entropy_hand_velocity,
            'sample_entropy_hand_velocity': sample_entropy_hand_velocity,
            'sample_entropy_elbow_angle': sample_entropy_elbow_angle,
            'spectral_arc_length': sal,
            'index_of_curvature': ioc,
        }
        
        measures_list.append(measures)

        df1 = pd.DataFrame(measures_list)
        df2 = calculate_basic_statistics(combined_data_with_kinematics)
        merged_df = pd.merge(df1, df2, on=['participant_id', 'trial_number', 'side', 'condition'], how='inner')

    
    return merged_df #pd.DataFrame(measures_list)


