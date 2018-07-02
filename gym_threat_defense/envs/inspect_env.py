"""
POMDP environment with discrete action and state space.

Authors:
Johan Backman - johback@student.chalmers.se
Hampus Ramstrom - hampusr@student.chalmers.se
"""

import numpy as np

from gym.spaces import Discrete
from enum import Enum

from base_env import BaseEnv
from inspect_env_parameters import STATES, REWARDS, TRANSITIONS, OBSERVATIONS


class Action(Enum):
    """
    The Action class, containing an action, i.e. a counter-measure.

    Defined as a numeric value [0, 4].
    """

    NONE = 0
    INSPECT = 1
    BLOCK = 2
    DISCONNECT = 3
    BOTH = 4

    def __str__(self):
        """Action to string representation."""
        if self._value_ == Action.NONE:
            return 'None'
        elif self._value_ == Action.INSPECT:
            return 'Inspect state of network'
        elif self._value_ == Action.BLOCK:
            return 'Block WebDAV service'
        elif self._value_ == Action.DISCONNECT:
            return 'Disconnect machine 2'
        else:
            return 'Block WebDAV service and Disconnect machine 2'


class State():
    """
    The State class, containing a state, i.e. the nodes enabled of the network.

    Defined as a binary vector of length 12
    """

    def __init__(self, index):
        """Initiate a state in the MDP."""
        self.index = index

    def as_list(self):
        """Return state as a vector."""
        return STATES[self.index]


class Observation():
    """
    The Observation class, containing an observation, i.e. an uncertain status.

    Denotes the status of the nodes in the network, as a binary vector of
    length 12.
    """

    def __init__(self, index):
        """Create observation from an index."""
        self.index = index

    def as_list(self):
        """Return the observation as a vector."""
        return STATES[self.index]


class ThreatDefenseInspectEnv(BaseEnv):
    """
    The class of the Defense environment.

    An OpenAI environment of the toy example given in Optimal Defense
    Policies for Partially Observable Spreading Processes on Bayesian Attack
    Graphs by Miehling, E., Rasouli, M., & Teneketzis, D. (2015). It
    constitutes a 29-state/observation, 5-action POMDP defense problem.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        """POMDP environment."""
        self.action_space = Discrete(len(Action))
        self.state_space = Discrete(STATES.shape[0])
        self.observation_space = Discrete(OBSERVATIONS.shape[0])
        self.all_states = STATES

    def step(self, action):
        """
        Progress the simulation one step.

        Arguments:
        action -- an Action containing a numeric value [0, 3].

        Returns:
        A tuple (o, r, d, i), where

        o -- the observation as a binary vector.
        r -- the reward gained as an integer.
        d -- boolean indicating if the simulation is done or not.
        i -- simulation meta-data as a dictionary.
        """
        assert self.action_space.contains(action)
        assert self.done is not True

        if not type(action) is Action:
            action = Action(action)

        self.t += 1
        self.last_action = action

        reward = REWARDS[action.value][self.state.index]

        self.done = self._is_terminal(action, self.t)

        old_state = self.state
        self.state = self._next_state(action, old_state)
        self.info = self._update_info(action, old_state, self.state)

        if action == Action.INSPECT:
            return (self._sample_observation(self.state).as_list(),
                    self.state.as_list()), reward, self.done, self.info
        else:
            return (self._sample_observation(self.state).as_list(),), \
                reward, self.done, self.info

    def reset(self):
        """
        Reset the simulation.

        Returns:
        The beginner observation as a binary vector.
        """
        self.done = False
        self.state = State(0)
        self.last_action = Action.NONE
        self.t = 0
        self.info = {}
        return (self.state.as_list(),)

    def _next_state(self, action, state):
        """
        Generate the next state from an action.

        Arguments:
        action -- an Action containing numeric value [0, 3].
        state -- a State containing a binary vector of length 12.

        Returns:
        The next State containing a binary vector of length 12.
        """
        probs = TRANSITIONS[action.value][state.index]
        return State(np.random.choice(probs.shape[0], p=probs))

    def _is_terminal(self, action, time_step):
        """
        Check if the current action terminates the simulation.

        Arguments:
        action -- an Action containing the numeric value [0, 3].
        time_step -- the current time_step as a numeric value
            0 <= time_step < max_time_step.

        Returns:
        A boolean depending on if the simulation is over or not.
        """
        return self.last_action.value is Action.BOTH.value or \
            time_step >= 100

    def _sample_observation(self, state):
        """
        Generate a new observation from the current state and action taken.

        Arguments:
        action -- an Action containing the numeric value [0, 3].
        state -- a State containing a binary vector of length 12.

        Returns:
        An Observation containing a binary vector of length 12.
        """
        probs = OBSERVATIONS[state.index]
        self.last_obs = Observation(np.random.choice(probs.shape[0], p=probs))
        return self.last_obs

    def _update_info(self, action, old_state, new_state):
        """
        Update the info meta-data Dict.

        Update the info with respect to the current and past state as well as
        the action taken.

        Arguments:
        action -- an Action containing the numeric value [0, 3].
        old_state -- a State containing a binary vector of length 12,
            representing the state at the past time step.
        new_state -- a State containing a binary vector of length 12,
            representing the state at the current time step.

        Returns:
        A dictionary containing the updated info with respect to the current
            and past state as well as the action taken.

        """
        return {'last_transition': TRANSITIONS[action.value][old_state.index],
                'last_observation': OBSERVATIONS[new_state.index],  # noqa
                'state': new_state.as_list()}
