import numpy as np
import matplotlib.pyplot as plt

vu = 5
m = 1
Cf = 1
Cr = 1
theta = [Cf, Cr]
a = 2.5
b = 2.5
width = 2
Iz = 1/12 * m * ((a+b)**2 + 3 * width**2)


def dynamics():
    # x = [y, psi, v, r] and psi~= 0
    A0 = [
        [0, vu, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, -vu],
        [0, 0, 0, 0]
    ]
    phi = [
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -2 / (m*vu), -2*a / (m*vu)],
            [0, 0, -2*a / (Iz*vu), -2*a**2 / (Iz*vu)]
        ], [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -2 / (m*vu), 2*b / (m*vu)],
            [0, 0, 2*b / (Iz*vu), -2*b**2 / (Iz*vu)]
        ],
    ]
    A = A0 + np.tensordot(theta, phi, axes=[0, 0])
    B = [[0],
         [0],
         [2*Cf / m],
         [Cf*a / Iz]]
    A = np.array(A)
    B = np.array(B)
    x = np.zeros((4, 1))
    return x, A, B


def step(x, A, B, u, noise, dt):
    dx = A@x + B@u + B@noise
    return x + dx * dt


def simulate(dt=0.1, sigma=0):
    time = np.arange(0, 15, dt)
    x, A, B = dynamics()
    Bp = np.array([[1], [0], [0], [0]])
    xx, uu = [], []
    K = np.array([[1e-1, 2, 0, 1]])
    for t in time:
        u = - K @ x
        noise = np.array([[sigma * (2*np.random.random()-1)]])
        x = step(x, A, B, u, noise, dt)
        omega = 2*np.pi/40
        # x += 10*omega*np.cos(omega*t) * Bp * dt
        x += np.isclose(t, 1.9, atol=dt/2)*5 * Bp
        xx.append(x.copy())
        uu.append(u.copy())
    xx, uu = np.array(xx), np.array(uu)
    return time, xx, uu


def main():
    time, xx, uu = simulate()
    pos_x = vu * time
    pos_y = xx[:, 0, 0]
    psi_x = np.cos(xx[:, 1, 0])
    psi_y = np.sin(xx[:, 1, 0])
    dir_x = np.cos(xx[:, 1, 0] + uu[:, 0, 0])
    dir_y = np.sin(xx[:, 1, 0] + uu[:, 0, 0])
    fig, ax = plt.subplots(1, 1)
    ax.plot(pos_x, pos_y)
    dir_scale = 1/5
    ax.quiver(pos_x[::20]-1/dir_scale*psi_x[::20],
              pos_y[::20]-1/dir_scale*psi_y[::20],
              psi_x[::20], psi_y[::20],
              angles='xy', scale_units='xy', scale=dir_scale, width=0.005, headwidth=1)
    ax.quiver(pos_x[::20], pos_y[::20], dir_x[::20], dir_y[::20],
              angles='xy', scale_units='xy', scale=0.25, width=0.005, color='r')
    ax.axis("equal")
    ax.grid()
    # ax1.plot(pos_x, xx[:, 3, 0])
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()