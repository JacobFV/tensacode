from __future__ import annotations

from typing import List
from tensacode._utils import use_cached
from tensacode.base.base_engine import R
from tensacode.nn.nn_engine import NNEngine


class Observation:
    pass


class Action:
    pass


class Step:
    observation: Observation
    action: Action
    reward: float


class Episode:
    agent: Agent
    environment: WorldModel
    steps: List[Step]


class Thought:
    description: str


ThoughtRelationship = tuple[Thought, Thought, str]


class Object:
    description: str


class WorldModel:
    objects: List[Object]
    focused_objects: List[Object]
    self: Agent


class Agent(Object):
    s_ego: R
    s_world: R
    world_model: WorldModel
    past_episodes: List[Episode]
    current_episode: Episode

    def __init__(self):
        self.llm_engine = NNEngine()

    def step(self, obs: Observation, *, idx: int):
        self.current_episode.steps.append(Step(obs, None, 0))

        # perception
        obs_enc = self.sensorimotor_fusion(self.world_model)
        immediate_relevant_trajs = use_cached(
            idx,
            self.f_obs_traj_retrieval,
            lambda: self.retrieve_relevant_past_trajectories(obs_enc),
        )
        immediate_relevant_trajs = use_cached(
            idx,
            self.f_world_traj_retrieval,
            lambda: self.retrieve_relevant_past_trajectories(self.s_world),
        )
        long_term_relevant_trajs = use_cached(
            idx,
            self.f_ego_traj_retrieval,
            lambda: self.retrieve_relevant_past_trajectories(self.s_ego),
        )

        # action

        # Deliberate on possible actions
        action_candidates = self.generate_action_candidates(self.active_thoughts)

        # Evaluate and select best action
        selected_action = self.evaluate_action_utility(action_candidates)

        # Execute action and observe results
        observation = self.execute_action_in_environment(selected_action)

        # Update episodic memory
        self.update_episodic_memory(selected_action, observation)

        # Reflect on outcomes
        self.metacognitive_reflection(self.current_episode)

        return selected_action
