import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pygame


class Sail(object):

    config = {
        "into_wind_time": 1,
        "up_wind_time": 0.5,
        "cross_wind_time": 0.1,
        "down_wind_time": 0.05,
        "away_wind_time": 0,
        "delay": 0.3,
        "wind_dynamics": [
            [0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
            [0.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.4, 0.3, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.4, 0.2, 0.4, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.4, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.4],
            [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3]
        ],
        "video_frames_per_step": 10,
        "screen_width": 800,
        "screen_height": 600,
    }

    def __init__(self, n):
        self.n = n
        self.x_a, self.y_a = 1, 1
        self.x_b, self.y_b = n-1, n-1

        self.x, self.y = self.x_a, self.y_a
        self.heading, self.w, self.t = 1, 1, 2

        self.trajectory = []
        self.best_heading = None
        self.fig = None
        self.viewer = None
        self.automatic_rendering_callback = None
        self.frames_to_render = 0

    def reset(self):
        self.x, self.y = self.x_a, self.y_a
        self.heading, self.w, self.t = 1, 1, 1
        self.trajectory = []

    def step(self, action):
        self.heading = action
        # Reward
        time = self.calc_delay(self.t, self.tack_type(self.heading, self.w)) \
            + self.time(self.heading, self.w)
        reward = -time
        # Transition
        self.t = self.tack_type(self.heading, self.w)
        self.x += self.dx(self.heading)
        self.y += self.dy(self.heading)
        self.w = self.wind(self.w)
        state = (self.x, self.y, self.heading, self.w)
        self.trajectory.append(state)
        # Terminal
        done = (self.x, self.y) == (self.x_b, self.y_b)
        return state, reward, done, {}

    def render(self, mode='human'):
        """
            Render the environment.

            Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        if self.viewer is None:
            self.viewer = Viewer(self)

        if mode == 'rgb_array':
            image = self.viewer.get_image()
            self.frames_to_render = self.config["video_frames_per_step"]
            return image

    def _automatic_rendering(self):
        if self.automatic_rendering_callback and self.viewer is not None:
            for _ in range(self.frames_to_render):
                self.automatic_rendering_callback()
            self.frames_to_render = 0

    def calculate(self):
        v, self.best_heading = self.dynamic_prog()
        print("expected best time = ", v[self.x_a][self.y_a][1][1])

    def simulate(self):
        done = False
        total_time = 0
        while not done:
            # Get best heading
            heading = self.best_heading[self.x, self.y, self.t, self.w]
            _, reward, _, done = self.step(heading)
            total_time += -reward
            self.render()

        print("total time", total_time)

    @staticmethod
    def tack(h, w):
        return w-h if (w - h) >= 0 else w-h+8

    def tack_type(self, h, w):
        t = self.tack(h, w)
        if 1 <= t <= 3:  # Right wind
            return 1
        elif 5 <= t <= 7:  # Left wind
            return 0
        else:  # Front or rear wind
            return 2

    def calc_delay(self, tack_type, next_tack_type):
        return 0 if (tack_type == next_tack_type or tack_type == 2 or next_tack_type == 2) else self.config["delay"]

    def time(self, h, w):
        time_factor = np.sqrt(2) if (h % 2 == 1) else 1
        t = self.tack(h, w)
        if t == 0:
            return self.config["into_wind_time"] * time_factor
        elif t in [1, 7]:
            return self.config["up_wind_time"] * time_factor
        elif t in [2, 6]:
            return self.config["cross_wind_time"] * time_factor
        elif t in [3, 5]:
            return self.config["down_wind_time"] * time_factor
        elif t == 4:
            return self.config["away_wind_time"] * time_factor
        else:
            return 0

    @staticmethod
    def dx(h):
        if h in [0, 4]:
            return 0
        elif h in [1, 2, 3]:
            return 1
        elif h in [5, 6, 7]:
            return -1
        else:
            return 0

    @staticmethod
    def dy(h):
        if h in [2, 6]:
            return 0
        elif h in [0, 1, 7]:
            return 1
        elif h in [3, 4, 5]:
            return -1
        else:
            return 0

    def wind(self, old_w):
        return np.random.choice(range(8), p=self.config["wind_dynamics"][old_w])

    def dynamic_prog(self):
        tolerance, error = 1.0e-3, np.inf
        v = np.ones((self.n + 1, self.n + 1, 3, 8)) * np.inf
        v[self.x_b, self.y_b, :, :] = 0
        best_heading = np.zeros((self.n + 1, self.n + 1, 3, 8))

        while error > tolerance:
            error = 0
            for x in range(1, self.n):
                for y in range(1, self.n):
                    if x != self.x_b or y != self.y_b:
                        for t in range(3):
                            for w in range(8):
                                new_value = np.inf
                                h_min = 0
                                for h in range(8):
                                    f = 0
                                    for w_p in range(8):
                                        f = self.calc_delay(t, self.tack_type(h, w)) \
                                            + self.time(h, w) \
                                            + sum(self.config["wind_dynamics"][w][w_p] *
                                                  v[x+self.dx(h), y+self.dy(h), self.tack_type(h, w), w_p]
                                                  for w_p in range(8))
                                    if f < new_value:
                                        new_value = f
                                        h_min = h
                                if abs(new_value - v[x][y][t][w]) > error:
                                    error = abs(new_value - v[x][y][t][w])
                                best_heading[x][y][t][w] = h_min
                                v[x][y][t][w] = new_value
            print("residual = ", error)
        return v, best_heading


class Viewer(object):
    def __init__(self, env):
        self.env = env

        pygame.init()
        pygame.display.set_caption("Sailing-env")
        self.screen = pygame.display.set_mode([self.env.config["screen_width"], self.env.config["screen_height"]])

    def render(self):
        surf_size = self.screen.get_size()
        img_str, size = self.plot(self.env.history,
                                  figsize=(surf_size[0]/100, surf_size[1]/100))
        surf = pygame.image.fromstring(img_str, size, "RGB")
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

    def get_image(self):
        """
        :return: the rendered image as a rbg array
        """
        data = pygame.surfarray.array3d(self.screen)
        return np.moveaxis(data, 0, 1)

    def plot(self, trajectory, figsize):
        fig = plt.figure(figsize=figsize, tight_layout=True)
        x, y, h, w = trajectory[-1]
        # Wind
        plt.arrow(0.5, self.env.n - 0.5, -0.5*self.env.dx(w), -0.5*self.env.dy(w), color="k")
        plt.plot([0, 0, 1, 1, 0], self.env.n - 1 + np.array([0, 1, 1, 0, 0]), color="k")
        # Boat
        direction = np.array([self.env.dx(h), self.env.dy(h)])
        direction_n = direction / np.linalg.norm(direction)
        side = np.array([[0,  1], [-1, 0]]) @ direction_n
        points = np.array([0.4*direction_n, 0.1*side, -0.1*side, 0.4*direction_n])
        plt.plot(x + 0.3*direction[0] + points[:, 0], y + 0.3*direction[1] + points[:, 1], 'm')
        # Path
        for x, y, h, w in trajectory:
            t = self.env.tack_type(h, w)
            if t == 0:
                color = "r"
            elif t == 1:
                color = "g"
            else:
                color = "k"
            plt.arrow(x, y, self.env.dx(h), self.env.dy(h), color=color)

        plt.axis('equal')
        plt.xlim((0, self.env.n))
        plt.ylim((0, self.env.n))

        # Figure export
        fig.canvas.draw()
        data_str = fig.canvas.tostring_rgb()
        plt.close()
        return data_str, fig.canvas.get_width_height()


if __name__ == '__main__':
    sail = Sail(5)
    sail.calculate()
    sail.simulate()
