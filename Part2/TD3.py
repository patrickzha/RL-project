import sys, os
sys.path.insert(0, os.path.abspath(".."))
import copy
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from common import helper as h
from common.buffer import ReplayBuffer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Actor-critic agent
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, action_dim)
        )

    def forward(self, state):
        return self.max_action * torch.tanh(self.actor(state))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim+action_dim, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.value(x)  # output shape [batch, 1]


class DDPG(object):
    def __init__(
            self,
            state_shape,
            action_dim,
            max_action,
            actor_lr,
            critic_lr,
            gamma,
            tau,
            batch_size,
            use_ou=False,
            normalize=False,
            buffer_size=1e6
    ):
        state_dim = state_shape[0]
        self.action_dim = action_dim
        self.max_action = max_action
        self.pi = Policy(state_dim, action_dim, max_action).to(device)
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=actor_lr)

        self.q1 = Critic(state_dim, action_dim).to(device)
        self.q_target1 = copy.deepcopy(self.q1)
        self.q_optim1 = torch.optim.Adam(self.q1.parameters(), lr=critic_lr)

        self.q2 = Critic(state_dim, action_dim).to(device)
        self.q_target2 = copy.deepcopy(self.q2)
        self.q_optim2 = torch.optim.Adam(self.q2.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(state_shape, action_dim, max_size=int(buffer_size))
        if normalize:
            self.state_scaler = h.StandardScaler(n_dim=state_dim)
        else:
            self.state_scaler = None

        if use_ou:
            self.noise = h.OUActionNoise(mu=np.zeros((action_dim,)))
        else:
            self.noise = None
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        
        # used to count number of transitions in a trajectory
        self.buffer_ptr = 0
        self.buffer_head = 0 
        self.random_transition = 5000  # collect 5k random data for better exploration

    # def update(self):
    #     """ After collecting one trajectory, update the pi and q for #transition times: """
    #     info = {}
    #     update_iter = self.buffer_ptr - self.buffer_head  # update the network once per transition
    #
    #     if self.buffer_ptr > self.random_transition:  # update once have enough data
    #         for _ in range(update_iter):
    #             info = self._update()
    #
    #     # update the buffer_head:
    #     self.buffer_head = self.buffer_ptr
    #     return info

    @property
    def buffer_ready(self):
        return self.buffer_ptr > self.random_transition

    def update(self):
        batch = self.buffer.sample(self.batch_size, device=device)

        # TODO: Task 2
        ########## Your code starts here. ##########
        # Hints: 1. compute the Q target with the q_target and pi_target networks
        #        2. compute the critic loss and update the q's parameters
        #        3. compute actor loss and update the pi's parameters
        #        4. update the target q and pi using h.soft_update_params() (See the DQN code)
        
        # pass
        if self.state_scaler is not None:
            self.state_scaler.fit(batch.state)
            states = self.state_scaler.transform(batch.state)
            next_states = self.state_scaler.transform(batch.next_state)
        else:
            states = batch.state
            next_states = batch.next_state

        # compute current q
        # q_cur = self.q(states, batch.action)
        
        # compute target q
        with torch.no_grad():
            next_action = (self.pi_target(next_states)).clamp(-self.max_action, self.max_action)
            q_tar1 = self.q_target1(next_states, next_action)
            td_target1 = batch.reward + self.gamma * batch.not_done * q_tar1
        
            q_tar2 = self.q_target2(next_states, next_action)
            td_target2 = batch.reward + self.gamma * batch.not_done * q_tar2
        
            td_target = torch.min(td_target1, td_target2)

        # compute critic loss
        q_cur1 = self.q1(states, batch.action)
        q_cur2 = self.q2(states, batch.action)

        critic_loss = F.mse_loss(q_cur1, td_target) + F.mse_loss(q_cur2, td_target)


        # optimize the critic
        self.q_optim1.zero_grad()
        self.q_optim2.zero_grad()
        critic_loss.backward()
        self.q_optim1.step()
        self.q_optim2.step()

        
        # critic_loss.backward()
        

        # compute actor loss
        actor_loss = -self.q1(states, self.pi(states)).mean() -self.q2(states, self.pi(states)).mean()

        # optimize the actor
        self.pi_optim.zero_grad()
        actor_loss.backward()
        self.pi_optim.step()

        # update the target q and target pi
        h.soft_update_params(self.q1, self.q_target1, self.tau)
        h.soft_update_params(self.q2, self.q_target2, self.tau)
        h.soft_update_params(self.pi, self.pi_target, self.tau)
        ########## Your code ends here. ##########

        # if you want to log something in wandb, you can put them inside the {}, otherwise, just leave it empty.
        # return {'q': q_cur1.mean().item()}
        return {}
    
    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        if observation.ndim == 1:
            observation = observation[None]  # add the batch dimension
        x = torch.from_numpy(observation).float().to(device)

        if self.state_scaler is not None:
            x = self.state_scaler.transform(x)

        if self.buffer_ptr < self.random_transition:  # collect random trajectories for better exploration.
            action = torch.rand(self.action_dim)
        else:
            expl_noise = 0.1  # the stddev of the expl_noise if not evaluation
            ########## Your code starts here. ##########
            # Use the policy to calculate the action to execute
            # if evaluation equals False, add normal noise to the action, where the std of the noise is expl_noise
            # Hint: Make sure the returned action shape is correct.
            # pass

            action = self.pi(x)
            
            if not evaluation:
                if self.noise is not None:
                    action = action + torch.from_numpy(self.noise()).float().to(device)
                else:
                    action = action + expl_noise * torch.rand_like(action)

            ########## Your code ends here. ##########

        return action, {}  # just return a positional value

    def record(self, state, action, next_state, reward, done):
        """ Save transitions to the buffer. """
        self.buffer_ptr += 1
        self.buffer.add(state, action, next_state, reward, done)

    # You can implement these if needed, following the previous exercises.
    def load(self, filepath):
        path = filepath/'ddpg.pt'
        d = torch.load(path)
        self.pi.load_state_dict(d['actor'])
        self.q1.load_state_dict(d['critic1'])
        self.q2.load_state_dict(d['critic2'])
    
    def save(self, filepath):
        path = filepath/'ddpg.pt'
        torch.save({'actor':self.pi.state_dict(),
                    'critic1':self.q1.state_dict(),
                    'critic2':self.q2.state_dict()
                    },path)
