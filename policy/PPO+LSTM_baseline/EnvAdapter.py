import sys
import os

# Path injection to ensure we find the habitat-lab source
sys.path.insert(0, "/Users/xinghebai/Desktop/Master Project/habitat-lab/habitat")

import habitat
import gym 
from gym import spaces
import numpy as np
from PIL import Image 

class HabitatEnvAdapter(gym.Env):
    def __init__(self, config_path, scorer_model=None):
        super().__init__()
        self.env = habitat.Env(config=habitat.get_config(config_path))
        self.scorer = scorer_model
        
        self.prev_score = 0.0
        self.prev_loc = None
        self.steps = 0
        
        # FIX: Map only to the actions allowed by pointnav_gibson.yaml
        # The config only supports: stop, move_forward, turn_left, turn_right
        self.action_mapping = {
            0: "move_forward", 
            1: "turn_left", 
            2: "turn_right", 
            3: "stop"
        }
        
        self.observation_space = spaces.Dict({
            "frame": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
            "score": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "delta": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "egomotion": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        })
        
        # FIX: Reduce action space to 4 (was 9)
        self.action_space = spaces.Discrete(4)

    def reset(self):
        obs = self.env.reset()
        frame = obs['rgb'] 
        current_score = self._get_score(frame)
        
        self.prev_score = current_score
        self.prev_loc = self.env.sim.get_agent_state().position
        self.steps = 0
        
        return self._format_obs(frame, current_score, 0.0, np.zeros(5))

    def step(self, action_idx):
            # Default to "stop" if index is out of bounds
            action_name = self.action_mapping.get(action_idx, "stop")
            
            obs = self.env.step(action_name)
            done = self.env.episode_over
            
            self.steps += 1
            
            frame = obs['rgb']
            current_score = self._get_score(frame)
            score_delta = current_score - self.prev_score
            
            curr_loc = self.env.sim.get_agent_state().position
            motion = curr_loc - self.prev_loc
            egomotion = np.concatenate([np.array(motion), np.array([0.0, 0.0])])
            
            # Was * 10.0, now * 50.0 to make improvement very obvious
            # Lower the step penalty to -0.01 so it doesn't discourage movement
            reward = (score_delta * 50.0) - 0.01 
            
            if action_name == "stop":
                # Lower threshold
                # Was 0.7, now 0.5 so the agent finds success earlier
                if current_score > 0.5:
                    reward += 5.0 # Big Jackpot to encourage stopping on good views
                    print(f"  *** SUCCESS! Found score {current_score:.2f} ***")
                else:
                    # Remove harsh penalty for stopping too early
                    # Was -1.0, now -0.1. A gentle slap instead of a punch.
                    reward -= 0.1
                    
                done = True 
                
            self.prev_score = current_score
            self.prev_loc = curr_loc
            
            return self._format_obs(frame, current_score, score_delta, egomotion), reward, done, {}

    def _get_score(self, frame):
        if self.scorer is None:
            return np.random.random()
        return self.scorer.get_score(frame)

    def _format_obs(self, frame, score, delta, egomotion):
        # Resize using Pillow
        img = Image.fromarray(frame)
        img_resized = img.resize((84, 84))
        frame_resized = np.array(img_resized)
        
        return {
            'frame': frame_resized, 
            'score': np.array([score], dtype=np.float32), 
            'delta': np.array([delta], dtype=np.float32), 
            'egomotion': np.array(egomotion, dtype=np.float32)
        }
    

