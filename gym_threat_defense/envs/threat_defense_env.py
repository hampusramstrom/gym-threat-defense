"""
POMDP environment with discrete action and state space.

Authors:
Johan Backman - johback@student.chalmers.se
Hampus Ramstrom - hampusr@student.chalmers.se
"""

import gym
import pyglet
import numpy as np

# Graphical rendering resources
import networkx as nx

from gym.spaces import Discrete
from enum import Enum

from defense_env_parameters import STATES, REWARDS, TRANSITIONS, OBSERVATIONS


class Action(Enum):
    """
    The Action class, containing an action, i.e. a counter-measure.

    Defined as a numeric value [0, 3].
    """

    NONE = 0
    BLOCK = 1
    DISCONNECT = 2
    BOTH = 3

    def __str__(self):
        """Action to string representation."""
        if self._value_ == Action.NONE:
            return 'None'
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


class ThreatDefenseEnv(gym.Env):
    """
    The class of the Defense environment.

    An OpenAI environment of the toy example given in Optimal Defense
    Policies for Partially Observable Spreading Processes on Bayesian Attack
    Graphs by Miehling, E., Rasouli, M., & Teneketzis, D. (2015). It
    constitutes a 29-state/observation, 4-action POMDP defense problem.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        """POMDP environment."""
        self.action_space = Discrete(len(Action))
        self.state_space = Discrete(STATES.shape[0])
        self.observation_space = Discrete(OBSERVATIONS.shape[1])
        self.all_states = STATES
        self.last_obs = None
        self.viewer = None

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

        return self._sample_observation(action, old_state).as_list(), \
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
        return self.state.as_list()

    def render(self, mode='human', close=False):
        """
        Render the simulation depending on rendering mode.

        Arguments:
        mode -- select render mode ['human', 'rgb_array'].
        close -- true or false if the rendering should be closed or be kept
            after the simulation finishes.
        """
        from gym.envs.classic_control import rendering

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(-1.0, 9.0, -1.0, 9.0 *
                                   screen_height / screen_width)

        # TODO(Johan): Adjust this to use adjacency matrix to build the
        #              graph structure.
        # Generate polygons for each node in the graph

        if not self.last_obs:
            return

        s = self.last_obs.as_list()

        graph = nx.DiGraph()
        graph.add_edges_from(
            [(1, 2), (2, 3), (3, 4), (4, 9), (8, 9), (9, 10), (10, 11),
             (8, 11), (11, 12), (5, 6), (6, 7), (7, 8)])
        pos = {1: (0.0, 0.0), 2: (1.0, 0.5), 3: (2.0, 2.0), 4: (3.0, 1.0),
               5: (6.0, 0.0), 6: (5.0, 1.0), 7: (6.0, 2.0), 8: (5.0, 2.0),
               9: (4.0, 2.0), 10: (4.0, 3.0), 11: (4.0, 4.0), 12: (5.0, 5.0)}

        for edge in graph.edges():
            pos_1 = pos[edge[0]]
            pos_2 = pos[edge[1]]

            link = self.viewer.draw_line(start=pos_1, end=pos_2)
            link.set_color(0, 0, 0)

            # Draw arrow
            v = [pos_2[0] - pos_1[0], pos_2[1] - pos_1[1]]
            u = v / np.linalg.norm(v)
            c_pos = [pos_2[0], pos_2[1]] - np.multiply(u, 0.45)

            c_trans = rendering.Transform(translation=(c_pos[0], c_pos[1]))
            c = self.viewer.draw_circle(0.07)
            c.set_color(0.0, 0.0, 0.0)
            c.add_attr(c_trans)

        for node, p in pos.iteritems():
            c_trans = rendering.Transform(translation=p)

            # Draw border
            c = self.viewer.draw_circle(0.45)
            c.set_color(0.0, 0.0, 0.0)
            c.add_attr(c_trans)

            # Draw inner circle
            c = self.viewer.draw_circle(0.4)

            # Color depends on if its been taken over or not
            if s[node - 1]:
                # Red color
                c.set_color(1.0, 0.1, 0.1)
            else:
                # Default gray color
                c.set_color(.8, .8, .8)

            c.add_attr(c_trans)

        # TODO(johan): fix this
        # Draw text
        def draw_text(x, y, txt):
            node_txt = pyglet.text.Label(
                str(txt), font_size=14,
                x=20, y=20, anchor_x='left', anchor_y='center',
                color=(255, 255, 255, 255))
            node_txt.draw()

        map(lambda x: draw_text(x[1][0], x[1][1], x[0]), pos.iteritems())

        if mode == 'human':
            first_row = '(%s) --> [%s] --> [%s] --> [%s]\n'
            second_row = '\t\t      \\--> [%s] <-- [%s] <-- [%s] <-- [%s] ' \
                         '<-- (%s)\n'
            third_row = '\t\t\t   \\--> [%s] <---/\n'
            fourth_row = '\t\t\t\t  \\--> [%s] --> [[%s]]\n'

            fmt_1 = first_row % (s[0], s[1], s[2], s[3])
            fmt_2 = second_row % (s[8], s[7], s[6], s[5], s[4])
            fmt_3 = third_row % (s[9])
            fmt_4 = fourth_row % (s[10], s[11])

            print 'Action: %s' % str(self.last_action)
            print fmt_1 + fmt_2 + fmt_3 + fmt_4
            return

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

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

    def _sample_observation(self, action, state):
        """
        Generate a new observation from the current state and action taken.

        Arguments:
        action -- an Action containing the numeric value [0, 3].
        state -- a State containing a binary vector of length 12.

        Returns:
        An Observation containing a binary vector of length 12.
        """
        probs = OBSERVATIONS[action.value][state.index]
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
                'last_observation': OBSERVATIONS[action.value][old_state.index],  # noqa
                'state': new_state.as_list()}
