"""
Base class for each POMDP threat defense environment.

Authors:
Johan Backman - johback@student.chalmers.se
Hampus Ramstrom - hampusr@student.chalmers.se
"""

import gym
import pyglet
import numpy as np

# Graphical rendering resources
import networkx as nx


class BaseEnv(gym.Env):
    """Base class for each POMDP threat defense environment."""

    def __init__(self):
        self.action_space = None
        self.state_space = None
        self.observation_space = None
        self.all_states = None
        self.last_obs = None
        self.viewer = None

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

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
