import gymnasium as gym
import numpy as np
import pandas as pd
import joblib
import os
import warnings
from gymnasium import spaces
from datetime import datetime, timedelta
from collections import deque
from pyswmm import Simulation, Nodes, Links, RainGages
from tensorflow.keras.models import load_model

GAMMA = 62.4; SLOPE = 0.0028108
TRANSECT_POINTS = np.array([[1.083,909.0],[2.0,908.708],[3.0,908.313],[4.0,908.0],[5.0,907.75],[6.0,907.125],[7.0,906.875],[8.0,906.875],[9.0,906.667],[10.0,906.438],[11.0,906.417],[12.0,906.25],[13.0,906.167],[13.5,906.042],[14.0,905.937],[14.333,905.875],[14.5,905.167],[15.0,905.125],[15.5,905.125],[16.0,905.042],[16.5,905.0],[17.0,905.021],[17.5,905.042],[18.0,905.063],[18.5,905.167],[19.0,905.229],[19.5,905.25],[20.0,905.333],[20.5,905.604],[20.667,905.938],[21.0,906.271],[22.0,906.333],[23.0,906.583],[24.0,906.667],[25.0,906.792],[26.0,907.146],[27.0,907.354],[28.0,907.563],[29.0,907.875],[30.0,908.583],[31.0,908.833],[31.5,909.0]])
INVERT_ELEVATION = np.min(TRANSECT_POINTS[:, 1])

def calculate_hydraulic_properties(depth):
    if depth <= 1e-6: return 0.0, 0.0
    water_surface_elevation = INVERT_ELEVATION + depth
    wet_points = TRANSECT_POINTS[TRANSECT_POINTS[:, 1] < water_surface_elevation]
    intersections = [];
    for i in range(1, len(TRANSECT_POINTS)):
        p1, p2 = TRANSECT_POINTS[i-1], TRANSECT_POINTS[i]
        y1, y2 = p1[1], p2[1]
        if (y1 < water_surface_elevation and y2 >= water_surface_elevation) or \
           (y2 < water_surface_elevation and y1 >= water_surface_elevation):
            x1, x2 = p1[0], p2[0]
            if abs(y2 - y1) > 1e-6:
                x_intersect = x1 + (x2 - x1) * (water_surface_elevation - y1) / (y2 - y1)
                intersections.append([x_intersect, water_surface_elevation])
    if len(intersections) > 0: all_wet_points = np.vstack([wet_points, intersections])
    else: all_wet_points = wet_points
    if len(all_wet_points) < 2: return 0.0, 0.0
    all_wet_points = all_wet_points[all_wet_points[:, 0].argsort()]
    x, y = all_wet_points[:, 0], all_wet_points[:, 1]
    area = np.trapz(water_surface_elevation - y, x)
    wetted_perimeter = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    return area, wetted_perimeter

def calculate_shear_stress(predicted_depth):
    try:
        if predicted_depth <= 1e-6: return 0.0
        area, wetted_perimeter = calculate_hydraulic_properties(predicted_depth)
        if wetted_perimeter <= 1e-6: return 0.0
        hydraulic_radius = area / wetted_perimeter
        shear_stress = GAMMA * hydraulic_radius * SLOPE
        return shear_stress
    except Exception: return 0.0

class ConnorCreekEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, mode='surrogate', 
                 surrogate_model_path=None, scaler_path=None, rainfall_csv_path=None,
                 inp_file=None, start_time=None, end_time=None):
        super().__init__()
        
        self.mode = mode
        self.W_EROSION = 250000.0
        self.W_ACTION = 0.0
        self.W_DEPTH_SQ = 0.02 
        self.CRITICAL_SHEAR_STRESS = 0.024
        
        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3])
        self.action_mapping = {0: 0.0, 1: 0.5, 2: 1.0}

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)

        self.POND_DEPTH_LIMITS = {'depth_pond1': 9.5, 'depth_pond2': 8.6, 'depth_pond3': 4.5, 'depth_pond4': 2.5}

        self.surrogate_input_order = [
            'depth_pond1', 'depth_pond2', 'depth_pond3', 'depth_pond4', 'depth_C1833_1',
            'Pond_1_setting', 'Pond_2_setting', 'Pond_3_setting', 'Pond_4_setting', 'precipitation'
        ]

        if not scaler_path or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        self.scaler_feature_order = list(self.scaler.feature_names_in_)

        if self.mode == 'surrogate':
            self._init_surrogate(surrogate_model_path, rainfall_csv_path, start_time, end_time)
        elif self.mode == 'swmm':
            self._init_swmm(inp_file, start_time, end_time)

    def _init_surrogate(self, surrogate_model_path, rainfall_csv_path, start_time, end_time):
        self.surrogate_output_order = ['depth_pond1', 'depth_pond2', 'depth_pond3', 'depth_pond4', 'depth_C1833_1']
        self.surrogate_model = load_model(surrogate_model_path, compile=False)
        self.start_time, self.end_time = start_time, end_time
        self.T = int((self.end_time - self.start_time).total_seconds() / 3600)
        self.rainfall_df = pd.read_csv(rainfall_csv_path, parse_dates=['datetime'], index_col='datetime')
        self.sequence_length = self.surrogate_model.input_shape[1]
        self.raw_state_history = deque(maxlen=self.sequence_length)

    def _init_swmm(self, inp_file, start_time, end_time):
        self.inp_file = inp_file
        self.start_time = start_time
        self.end_time = end_time
        self.T = int((self.end_time - self.start_time).total_seconds() / 3600)

    def _connect_to_swmm(self):
        self.sim = Simulation(self.inp_file); self.sim.start_time, self.sim.end_time = self.start_time, self.end_time
        self.sim.step_advance(3600); nodes = Nodes(self.sim); self.ponds = {i: nodes[f"Pond__{i}"] for i in range(1, 5)}
        links = Links(self.sim); self.valves = {i: links[f"Pond{i}_Valve"] for i in range(1, 5)}
        self.conduit_c1833_1 = links["C1833_1"]; self.raingage = RainGages(self.sim)["Raingage1"]; self.sim.start()

    def _get_swmm_state_as_dict(self):
        s = {'precipitation': self.raingage.rainfall}
        s.update({f'depth_pond{i}': self.ponds[i].depth for i in range(1,5)})
        s['depth_C1833_1'] = self.conduit_c1833_1.depth
        return s
    
    def _get_observation_from_raw_state(self):
        raw = self.current_raw_state
        obs_unscaled_dict = {
            'depth_pond1': raw.get('depth_pond1', 0.0), 'depth_pond2': raw.get('depth_pond2', 0.0),
            'depth_pond3': raw.get('depth_pond3', 0.0), 'depth_pond4': raw.get('depth_pond4', 0.0),
            'depth_C1833_1': raw.get('depth_C1833_1', 0.0),
            'Pond_1_setting': self.current_action_setting[0], 'Pond_2_setting': self.current_action_setting[1],
            'Pond_3_setting': self.current_action_setting[2], 'Pond_4_setting': self.current_action_setting[3],
            'precipitation': raw.get('precipitation', 0.0)
        }
        obs_df = pd.DataFrame([obs_unscaled_dict])[self.scaler_feature_order]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            scaled_array = self.scaler.transform(obs_df)
        scaled_df = pd.DataFrame(scaled_array, columns=self.scaler_feature_order)
        final_obs_vector = scaled_df[self.surrogate_input_order].values.flatten()
        return np.clip(final_obs_vector, 0, 1).astype(np.float32)

    def _calculate_reward(self, next_raw_state):
        depth_penalty_raw = sum(next_raw_state.get(f'depth_pond{i}', 0)**2 for i in range(1, 5))
        depth_term = depth_penalty_raw * self.W_DEPTH_SQ
        predicted_depth = next_raw_state.get('depth_C1833_1', 0)
        predicted_shear_stress = calculate_shear_stress(predicted_depth)
        erosion_term = ((max(0, predicted_shear_stress - self.CRITICAL_SHEAR_STRESS))**2) * self.W_EROSION
        action_term = np.sum(np.abs(self.last_action_setting - self.current_action_setting)) * self.W_ACTION
        overflow_term = 0
        if next_raw_state.get('depth_pond1', 0) > self.POND_DEPTH_LIMITS['depth_pond1']: overflow_term += 100
        if next_raw_state.get('depth_pond2', 0) > self.POND_DEPTH_LIMITS['depth_pond2']: overflow_term += 100
        if next_raw_state.get('depth_pond3', 0) > self.POND_DEPTH_LIMITS['depth_pond3']: overflow_term += 100
        if next_raw_state.get('depth_pond4', 0) > self.POND_DEPTH_LIMITS['depth_pond4']: overflow_term += 100
        return -(depth_term + erosion_term + action_term + overflow_term)

    def step(self, action):
        self.last_action_setting = self.current_action_setting
        self.current_action_setting = np.array([self.action_mapping[a] for a in action])
        
        terminated = False; truncated = False

        if self.mode == 'swmm':
            for i in range(1, 5): self.valves[i].target_setting = self.current_action_setting[i-1]
            try:
                self.sim.step_advance(3600); next_raw_state = self._get_swmm_state_as_dict()
            except Exception:
                truncated = True; next_raw_state = self.current_raw_state
                
        elif self.mode == 'surrogate':
            history_list = list(self.raw_state_history);
            history_list[-1]['Pond_1_setting'] = self.current_action_setting[0]
            history_list[-1]['Pond_2_setting'] = self.current_action_setting[1]
            history_list[-1]['Pond_3_setting'] = self.current_action_setting[2]
            history_list[-1]['Pond_4_setting'] = self.current_action_setting[3]
            history_df = pd.DataFrame(history_list)[self.scaler_feature_order]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                scaled_history = self.scaler.transform(history_df)
            scaled_df_for_model = pd.DataFrame(scaled_history, columns=self.scaler_feature_order)
            scaled_input_array = scaled_df_for_model[self.surrogate_input_order].values
            scaled_input = scaled_input_array.reshape(1, self.sequence_length, -1)
            scaled_prediction = self.surrogate_model.predict(scaled_input, verbose=0)[0]
            dummy_df = pd.DataFrame(np.zeros((1, len(self.scaler_feature_order))), columns=self.scaler_feature_order)
            dummy_df[self.surrogate_output_order] = scaled_prediction
            descaled_prediction_row = self.scaler.inverse_transform(dummy_df)[0]
            descaled_prediction = pd.Series(descaled_prediction_row, index=self.scaler_feature_order)
            next_raw_state = {}
            for col in self.surrogate_output_order:
                next_raw_state[col] = max(0, descaled_prediction[col])
            next_time = self.current_time + timedelta(hours=1)
            next_raw_state['precipitation'] = self.rainfall_df.loc[next_time, 'precipitation'] if next_time in self.rainfall_df.index else 0.0

        reward = self._calculate_reward(next_raw_state); self.t += 1; self.current_time += timedelta(hours=1)
        if self.t >= self.T:
            truncated = True
        self.current_raw_state = next_raw_state
        if self.mode == 'surrogate':
            history_append_dict = self.current_raw_state.copy()
            history_append_dict.update({f'Pond_{i}_setting': self.current_action_setting[i-1] for i in range(1,5)})
            self.raw_state_history.append(history_append_dict)
        return self._get_observation_from_raw_state(), reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self.t = 0; self.current_action_setting = np.zeros(4); self.last_action_setting = np.zeros(4)
        self.current_time = self.start_time

        if self.mode == 'swmm':
            if hasattr(self, 'sim'): self.sim.close()
            self._connect_to_swmm()
            for i in range(1, 5): self.valves[i].target_setting = 0.0
            self.current_raw_state = self._get_swmm_state_as_dict()
            
        elif self.mode == 'surrogate':
            self.raw_state_history.clear()
            initial_state = {
                'depth_pond1': 0.0, 'depth_pond2': 0.0, 'depth_pond3': 0.0, 'depth_pond4': 0.0,
                'depth_C1833_1': 0.0,
                'Pond_1_setting': 0.0, 'Pond_2_setting': 0.0, 'Pond_3_setting': 0.0, 'Pond_4_setting': 0.0,
                'precipitation': self.rainfall_df.loc[self.current_time, 'precipitation'] if self.current_time in self.rainfall_df.index else 0.0
            }
            for _ in range(self.sequence_length):
                self.raw_state_history.append(initial_state.copy())
            self.current_raw_state = initial_state

        return self._get_observation_from_raw_state(), {}

    def close(self):
        if self.mode == 'swmm' and hasattr(self, 'sim'):
            self.sim.report(); self.sim.close();
