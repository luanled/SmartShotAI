import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SmartShotPolicy(nn.Module):
    def __init__(self, action_dim=4, hidden_dim=256): 
        super(SmartShotPolicy, self).__init__()
        
        # 1. Visual Encoder (Process downscaled RGB)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            self.cnn_out_dim = self.cnn(torch.zeros(1, 3, 84, 84)).shape[1]

        # 2. Scalar Input Encoder
        self.scalar_dim = 1 + 1 + 5 + action_dim 
        self.fc_scalar = nn.Linear(self.scalar_dim, 64)

        # 3. LSTM Memory
        self.lstm = nn.LSTM(self.cnn_out_dim + 64, hidden_dim, batch_first=True)

        # 4. Heads (Actor-Critic)
        self.actor = nn.Linear(hidden_dim, action_dim) 
        self.critic = nn.Linear(hidden_dim, 1)         

    def forward(self, visual_obs, scalar_obs, hidden_state=None):
        visual_feat = self.cnn(visual_obs)
        scalar_feat = F.relu(self.fc_scalar(scalar_obs))
        combined = torch.cat([visual_feat, scalar_feat], dim=1).unsqueeze(1) 
        lstm_out, new_hidden = self.lstm(combined, hidden_state)
        x = lstm_out.squeeze(1)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value, new_hidden

class PolicyAgent:
    def __init__(self, model_path=None):
        self.action_dim = 4 
        self.policy = SmartShotPolicy(action_dim=self.action_dim)
        self.hidden_state = None 
        self.last_action = 0     
        
        if model_path:
            self.load_model(model_path)

    def reset(self):
        self.hidden_state = None
        self.last_action = 0

    def act(self, frame, score, delta_s, egomotion):
        visual_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        action_one_hot = [0] * self.action_dim
        if self.last_action < self.action_dim:
            action_one_hot[self.last_action] = 1
        
        if isinstance(egomotion, (np.ndarray, torch.Tensor)):
            egomotion = egomotion.tolist()
            
        scalar_input = [float(score), float(delta_s)] + egomotion + action_one_hot
        scalar_tensor = torch.tensor([scalar_input], dtype=torch.float32)

        with torch.no_grad():
            logits, _, self.hidden_state = self.policy(visual_tensor, scalar_tensor, self.hidden_state)
        
        probs = F.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().item()
        confidence = probs[0][action].item()

        self.last_action = action
        return action, confidence

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))