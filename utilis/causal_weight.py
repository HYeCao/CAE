import os
import pickle
import numpy as np
import pandas as pd
import time
from causallearnmain.causallearn.search.FCMBased import lingam
import sys
from utilis.config import ARGConfig
import torch
import torch.nn.functional as F
import ipdb


def get_sa2r_weight(env, memory, agent, sample_size=5000, causal_method='DirectLiNGAM'):
    states, actions, rewards, next_states, dones = memory.sample(sample_size)
    rewards = np.squeeze(rewards[:sample_size])
    rewards = np.reshape(rewards, (sample_size, 1))
    # AS_ori = np.hstack((states[:sample_size, :], actions[:sample_size, :]))
    AS_ori = np.hstack((actions[:sample_size, :], states[:sample_size, :]))
    AS = pd.DataFrame(AS_ori, columns=list(range(np.shape(AS_ori)[1])))

    if causal_method == 'DirectLiNGAM':
        # learn causal matrix state to action
        state_dim = np.shape(states)[1]
        action_dim = np.shape(actions)[1]
        model2 = lingam.DirectLiNGAM()
        model2.fit(AS)
        # weight_action_to_state = model2.adjacency_matrix_[:state_dim, state_dim:(state_dim + action_dim)]
        weight_action_to_state = model2.adjacency_matrix_[action_dim:, :action_dim]

        adjusted_states = states[:sample_size] + np.dot(actions[:sample_size], weight_action_to_state.T)
        print('weight_action_to_state', weight_action_to_state)
        print('states', states)
        print('adjusted_states', adjusted_states)

        X_adjusted = np.hstack((adjusted_states, actions[:sample_size, :], rewards))
        X_adjusted_df = pd.DataFrame(X_adjusted, columns=list(range(np.shape(X_adjusted)[1])))

        start_time = time.time()
        model3 = lingam.DirectLiNGAM()
        model3.fit(X_adjusted_df)
        end_time = time.time()
        model3._running_time = end_time - start_time
        weight_r_adjust = model3.adjacency_matrix_[-1, state_dim:(state_dim + action_dim)]

    weight_adjust = F.softmax(torch.Tensor(weight_r_adjust), 0)
    weight_adjust = weight_adjust.numpy()
    weight_adjust = weight_adjust * weight_adjust.shape[0]

    print('weight_adjust.shape', weight_adjust.shape)
    print('weight_adjust', weight_adjust)
    # * multiply by action size
    return weight_adjust, model3._running_time

