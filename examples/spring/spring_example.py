import numpy as np
from scipy.integrate import solve_ivp


def get_spring_data(n_ics):
    t, x, dx, z = generate_spring_data(n_ics)
    data = {}
    data['t'] = t
    data['x'] = x.reshape((n_ics * t.size, -1))
    data['dx'] = dx.reshape((n_ics * t.size, -1))
    data['z'] = z.reshape((n_ics * t.size, -1))[:, 0:1]
    data['dz'] = z.reshape((n_ics * t.size, -1))[:, 1:2]

    return data


def generate_spring_data(n_ics, b=0.2, m=0.5, k=1.):

    # state space representation of the spring
    # z_dot = [z1_dot, z2_dot]
    # z1_dot = z2
    # z2_dot = -b/m * z2 - (k / m) * z1
    f = lambda t, z: [z[1], -b/m * z[1] - (k / m) * z[0]]
    wn = np.sqrt(k / m)
    zeta = b/2/np.sqrt(m*k)
    Ts = 4/zeta/wn
    freq = wn / (2 * np.pi)
    t_step = 1 / (400 * freq) # ~1ms step for default parameters
    #t_step = 0.01
    t = np.arange(0, Ts, t_step)

    z = np.zeros((n_ics, t.size, 2))
    dz = np.zeros(z.shape)

    z1range = np.array([-1., 1.])
    z2range = np.array([-0.1, 0.1])
    i = 0
    while i < n_ics:
        # Random initial condition
        z0 = np.array([(z1range[1] - z1range[0]) * np.random.rand() + z1range[0],
                       (z2range[1] - z2range[0]) * np.random.rand() + z2range[0]])
        z0 = z0.reshape(1, -1)
        
        z[i] = solve_ivp(f, t_span=(t[0], t[-1]), y0=z0[0, :], t_eval=t, method='DOP853').y.T
        dz[i] = np.array([f(t[j], z[i, j]) for j in range(len(t))])

        i += 1

    x, dx = spring_to_movie(z, dz)


    return t, x, dx, z


def spring_to_movie(z, dz, n = 30):
    n_ics = z.shape[0]
    n_samples = z.shape[1]

    y1, y2 = np.meshgrid(np.linspace(-2.5, 2.5, n), np.linspace(2.5, -2.5, n))

    # defining a guassian over the image centered where our point mass is
    create_image = lambda x: np.exp(-((y1 - x) ** 2 + (y2 - 0) ** 2) / .05)
    d_create_image = lambda x, dx: -1 / .05 * create_image(x) * 2 * (y1 - x) * dx

    x = np.zeros((n_ics, n_samples, n, n))
    dx = np.zeros((n_ics, n_samples, n, n))
    for i in range(n_ics):
        for j in range(n_samples):
            x[i, j] = create_image(z[i, j, 0])
            dx[i, j] = d_create_image(z[i, j, 0], dz[i, j, 0])

    return x, dx

