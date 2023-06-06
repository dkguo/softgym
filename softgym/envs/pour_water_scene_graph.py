import json

import numpy as np
from gym.spaces import Box

import pyflex
import copy

from softgym.envs.pour_water import PourWaterPosControlEnv
from softgym.utils.misc import quatFromAxisAngle
import pickle


def load_scene_graph(scene_graph_path='data/pour_milk.json'):
    with open(scene_graph_path, 'rb') as f:
        scene_graph = json.load(f)
    return scene_graph


class PourWaterSceneGraphEnv(PourWaterPosControlEnv):
    def __init__(self, observation_mode, action_mode, 
                config=None, cached_states_path='pour_water_scene_graph_init_states.pkl', **kwargs):
        self.inner_step = 0
        self.prev_state = None

        super().__init__(observation_mode, action_mode, config, cached_states_path, **kwargs)

        # self.scene_graph = load_scene_graph()

    def get_state(self):
        state_dic = super().get_state()
        state_dic['step'] = self.inner_step
        state_dic['prev_state'] = self.prev_state
        return state_dic

    def set_state(self, state_dic):
        self.inner_step = state_dic['step']
        self.prev_state = state_dic['prev_state']
        super().set_state(state_dic)

    def _step(self, action):
        if self.inner_step % 8 == 0:
            self.prev_state = self.get_state()
        super()._step(action)

    # discrete reward binary
    def compute_reward(self, obs=None, action=None, set_prev_reward=False):
        state_dic = self.get_state()
        water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
        water_num = len(water_state)

        in_poured_glass = self.in_glass(water_state, self.poured_glass_states, self.poured_border, self.poured_height)
        in_control_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
        control_water = in_control_glass * (1 - in_poured_glass)
        control_water_num = np.sum(control_water)
        good_water = in_poured_glass * (1 - in_control_glass)
        good_water_num = np.sum(good_water)

        prev_water_state = self.prev_state['particle_pos'].reshape((-1, self.dim_position))
        prev_water_num = len(prev_water_state)
        prev_glass_states = self.prev_state['glass_states']
        prev_poured_glass_states = self.prev_state['poured_glass_states']

        prev_in_poured_glass = self.in_glass(prev_water_state, prev_poured_glass_states, self.poured_border, self.poured_height)
        prev_in_control_glass = self.in_glass(prev_water_state, prev_glass_states, self.border, self.height)
        prev_control_water = prev_in_control_glass * (1 - prev_in_poured_glass)
        prev_control_water_num = np.sum(prev_control_water)
        prev_good_water = prev_in_poured_glass * (1 - prev_in_control_glass)
        prev_good_water_num = np.sum(prev_good_water)

        good_diff = float(good_water_num - prev_good_water_num) / water_num
        control_diff = float(prev_control_water_num - control_water_num) / water_num

        # binary 2 + panalty action 10 + trajectory (bpat)
        reward = 0
        if self.inner_step < 17 * 8:
            # at the beginning, control water shouldn't reduce
            reward += -1 if control_diff > 0 else 0

            dx, dy, dtheta = self.action
            panalty = 10000. * (-dx ** 2 - dy ** 2 + dtheta ** 2)
            reward -= panalty

        else:
            reward += 1 if control_diff > 0 else 0
            reward += 1 if good_diff > 0 else 0

            dx, dy, dtheta = self.action
            panalty = 10000. * (dx ** 2 + dy ** 2 - dtheta ** 2)
            reward -= panalty

        # binary 2 + panalty action 10
        # reward = 1 if diff > 0 else 0
        # dx, dy, dtheta = self.action
        # panalty = 10000. * (dx ** 2 + dy ** 2 - dtheta ** 2)
        # reward -= panalty

        # binary 2 + panalty action 9 + large rotation space
        # reward = 1 if diff > 0 else 0
        # dx, dy, dtheta = self.action
        # panalty = 10000. * (dx ** 2 + dy ** 2)
        # reward -= panalty

        # binary 2 + panalty action 8
        # reward = 1 if diff > 0 else 0
        # dx, dy, dtheta = self.action
        # panalty = 10000. * (dx ** 2 + dy ** 2)
        # reward -= panalty

        # binary 2 + panalty action 7 + large action space
        # reward = 1 if diff > 0 else 0
        # dx, dy, dtheta = self.action
        # panalty = 100. * (dx ** 2 + dy ** 2)
        # reward -= panalty

        # binary 2 + panalty action 6 + large action space
        # reward = 1 if diff > 0 else 0
        # dx, dy, dtheta = self.action
        # panalty = 10. * (dx ** 2 + dy ** 2)
        # reward -= panalty
        # print("action panalty", -panalty)

        # binary 2 + panalty action 5
        # reward = 1 if diff > 0 else 0
        # dx, dy, dtheta = self.action
        # panalty = 10. * np.sqrt(dx ** 2 + dy ** 2)
        # reward -= panalty
        # print("action panalty", -panalty)

        # binary 2 + panalty action 4
        # reward = 1 if diff > 0 else 0
        # dx, dy, dtheta = self.action
        # if self.inner_step > 240:
        #     panalty = 10. * np.sqrt(dx ** 2 + dy ** 2)
        #     reward -= panalty
        #     print("action panalty", -panalty)

        # binary 2 + panalty action 3
        # reward = 1 if diff > 0 else 0
        # dx, dy, dtheta = self.action
        # if self.inner_step > 240:
        #     panalty = 1 * np.sqrt(dx ** 2 + dy ** 2)
        #     reward -= panalty
        #     print("action panalty", -panalty)

        # binary 2 + panalty action 2
        # reward = 1 if diff > 0 else 0
        # dx, dy, dtheta = self.action
        # if self.inner_step > 240:
        #     reward -= 0.1 * np.sqrt(dx ** 2 + dy ** 2)
        #     print("action panalty", -0.1 * np.sqrt(dx ** 2 + dy ** 2))

        # binary 2 + panalty action
        # reward = 1 if diff > 0 else 0
        # dx, dy, dtheta = self.action
        # reward -= 0.1 * np.sqrt(dx ** 2 + dy ** 2)
        # print("action panalty", -0.1 * np.sqrt(dx ** 2 + dy ** 2))

        # binary 2
        # reward = 1 if diff > 0 else 0

        # binary
        # if -1e-6 < diff < 1e-6:
        #     reward = 0
        # elif diff > 0:
        #     reward = 1
        # else:
        #     reward = -1

        # print(self.inner_step, self.action, reward)
        return reward


    # discrete reward
    # def compute_reward(self, obs=None, action=None, set_prev_reward=False):
    #     state_dic = self.get_state()
    #     water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
    #     water_num = len(water_state)
    #
    #     in_poured_glass = self.in_glass(water_state, self.poured_glass_states, self.poured_border, self.poured_height)
    #     in_control_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
    #     good_water = in_poured_glass * (1 - in_control_glass)
    #     good_water_num = np.sum(good_water)
    #
    #     prev_water_state = self.prev_state['particle_pos'].reshape((-1, self.dim_position))
    #     prev_water_num = len(prev_water_state)
    #     prev_glass_states = self.prev_state['glass_states']
    #     prev_poured_glass_states = self.prev_state['poured_glass_states']
    #
    #     prev_in_poured_glass = self.in_glass(prev_water_state, prev_poured_glass_states, self.poured_border, self.poured_height)
    #     prev_in_control_glass = self.in_glass(prev_water_state, prev_glass_states, self.border, self.height)
    #     prev_good_water = prev_in_poured_glass * (1 - prev_in_control_glass)
    #     prev_good_water_num = np.sum(prev_good_water)
    #
    #     reward = float(good_water_num - prev_good_water_num) / water_num
    #     print(self.inner_step, reward)
    #     return reward

    # def _reset(self):
    #     self.prev_num_in_poured_glass = None
    #     self.prev_num_in_control_glass = None
    #     self.reward = 0
    #     return super()._reset()
    #
    # def compute_reward(self, obs=None, action=None, **kwargs):
    #     state_dic = self.get_state()
    #     water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
    #
    #     num_in_poured_glass = np.sum(self.in_glass(water_state, self.poured_glass_states, self.poured_border, self.poured_height))
    #     num_in_control_glass = np.sum(self.in_glass(water_state, self.glass_states, self.border, self.height))
    #
    #     if self.prev_num_in_poured_glass is None:
    #         self.prev_num_in_poured_glass = num_in_poured_glass
    #         self.prev_num_in_control_glass = num_in_control_glass
    #         return 0
    #
    #     # self.reward += 1 if num_in_control_glass < self.prev_num_in_control_glass else 0
    #     self.reward += 1 if num_in_poured_glass > self.prev_num_in_poured_glass else 0
    #
    #     self.prev_num_in_poured_glass = num_in_poured_glass
    #     self.prev_num_in_control_glass = num_in_control_glass
    #
    #     return self.reward

    # def compute_reward(self, obs=None, action=None, **kwargs):
    #     """
    #     The reward is computed as the fraction of water in the poured glass.
    #     NOTE: the obs and action params are made here to be compatiable with the MultiTask env wrapper.
    #     """
    #     # print('compute_reward', self.inner_step)
    #     steps = str(int(self.inner_step // 8))
    #     # print('steps', steps)
    #     if steps not in self.scene_graph:
    #         return 0
    #     desire_carton_pos = np.array(self.scene_graph[steps]["nodes"]["milk carton"]['pose'])
    #     carton_pos = self.glass_states[0, :3]
    #
    #     desire_cup_pos = np.array(self.scene_graph[steps]["nodes"]["cup"]['pose'])
    #     cup_pos = self.poured_glass_states[0, :3]
    #
    #     # print('carton_pos', carton_pos)
    #     # print('cup_pos', cup_pos)
    #
    #     carton_cup_pos_reward = -np.linalg.norm(desire_carton_pos - desire_cup_pos - (carton_pos - cup_pos), ord=2)
    #
    #     state_dic = self.get_state()
    #     water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
    #
    #     num_in_poured_glass = np.sum(self.in_glass(water_state, self.poured_glass_states, self.poured_border, self.poured_height))
    #     num_in_control_glass = np.sum(self.in_glass(water_state, self.glass_states, self.border, self.height))
    #
    #     if self.prev_num_in_poured_glass is None:
    #         self.prev_num_in_poured_glass = num_in_poured_glass
    #         self.prev_num_in_control_glass = num_in_control_glass
    #         return 0
    #
    #     carton_milk_edge = self.scene_graph[steps]["edges"]["milk carton,milk"] if \
    #         "milk carton,milk" in self.scene_graph[steps]["edges"] else 0
    #
    #     milk_cup_edge = self.scene_graph[steps]["edges"]["cup,milk"] if \
    #         "cup,milk" in self.scene_graph[steps]["edges"] else 0
    #
    #     carton_milk_edge_reward = 1 if num_in_control_glass < self.prev_num_in_control_glass == carton_milk_edge else 0
    #     milk_cup_edge_reward = 1 if num_in_poured_glass > self.prev_num_in_poured_glass == milk_cup_edge else 0
    #
    #     self.prev_num_in_poured_glass = num_in_poured_glass
    #     self.prev_num_in_control_glass = num_in_control_glass
    #
    #     # print('carton_cup_pos_reward', carton_cup_pos_reward)
    #     # print('carton_milk_edge_reward', carton_milk_edge_reward)
    #     # print('milk_cup_edge_reward', milk_cup_edge_reward)
    #
    #     reward = carton_cup_pos_reward + carton_milk_edge_reward + milk_cup_edge_reward
    #
    #     return reward

    def get_default_config(self):
        config = {
            'fluid': {
                'radius': 0.033,
                'rest_dis_coef': 0.55,
                'cohesion': 0.1,  # not actually used, instead, is computed as viscosity * 0.01
                'viscosity': 2,
                'surfaceTension': 0,
                'adhesion': 0.0,  # not actually used, instead, is computed as viscosity * 0.001
                'vorticityConfinement': 40,
                'solidpressure': 0.,
                'dim_x': 8,
                'dim_y': 40,
                'dim_z': 8,
            },
            'glass': {
                'border': 0.02,
                'height': 0.4,
                'glass_distance': 0.4,
                'poured_border': 0.02,
                'poured_height': 0.2,
            },
            'camera_name': 'default_camera',
        }
        return config

    # def generate_env_variation(self, num_variations=5, config=None, **kwargs):
    #     self.cached_configs = []
    #     self.cached_init_states = []
    #     config = self.get_default_config()
    #     config_variations = [copy.deepcopy(config) for _ in range(num_variations)]
    #     for idx in range(num_variations):
    #         self.set_scene(config_variations[idx])
    #         init_state = copy.deepcopy(self.get_state())
    #
    #         self.cached_configs.append(config_variations[idx])
    #         self.cached_init_states.append(init_state)
    #
    #     return self.cached_configs, self.cached_init_states

    def initialize_camera(self):
        self.camera_params = {
            'default_camera': {'pos': np.array([1., 2.5, 0.1]),
                               'angle': np.array([0.45 * np.pi, -60 / 180. * np.pi, 0]),
                               'width': self.camera_width,
                               'height': self.camera_height},

            'cam_2d': {'pos': np.array([0.5, .7, 4.]),
                       'angle': np.array([0, 0, 0.]),
                       'width': self.camera_width,
                       'height': self.camera_height}
        }