import numpy as np
import matplotlib.pyplot as plt

u = 5
m = 1
Cf = 5
Cr = 5
a = 2.5
b = 2.5
width = 2
Iz = 1/12 * m * ((a+b)**2 + 3 * width**2)


def dynamics():
    # x = [y, psi, v, r] and psi~= 0
    A = [[0, u, 1, 0],
         [0, 0, 0, 1],
         [0, 0, -2*(Cf + Cr) / (m*u),         2*(Cr*b - Cf*a) / (m*u) - u],
         [0, 0, 2*(Cr*b - Cf*a) / (Iz*u), -2*(Cf*a**2 + Cr*b**2) / (Iz*u)]]
    B = [[0],
         [0],
         [2*Cf / m],
         [Cf*a / Iz]]
    A = np.array(A)
    B = np.array(B)
    x = np.zeros((4, 1))
    return x,A,B


def step(x, A, B, u, dt, sigma=2):
    noise = np.array([[sigma * (2*np.random.random()-1)]])
    dx = A@x + B@u + B@noise
    return x + dx * dt


def simulate(dt=0.1):
    time = np.arange(0, 30, dt)
    x, A, B = dynamics()
    xx = []
    K = 1e-2
    for t in time:
        u = np.array([[- K*x[0, 0]]])
        x = step(x, A, B, u, dt)
        xx.append(x.copy())
    xx = np.array(xx)
    return time, xx


def main():
    time, xx = simulate()
    pos_x = u * time
    pos_y = xx[:, 0, 0]
    psi = xx[:, 1, 0]
    dir_x = np.cos(psi)
    dir_y = np.sin(psi)
    plt.plot(pos_x, pos_y)
    plt.quiver(pos_x[::20] - dir_x[::20]/2, pos_y[::20] - dir_y[::20]/2, dir_x[::20], dir_y[::20])
    plt.axis("equal")
    plt.grid()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()