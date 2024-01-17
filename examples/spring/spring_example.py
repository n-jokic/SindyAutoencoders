import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import legendre


def get_spring_data(n_ics, spread=1):
    t, x, dx, z, dz = generate_spring_data(n_ics, spread)
    data = {}
    data['t'] = t
    data['x'] = x.reshape((n_ics * t.size, -1))
    data['dx'] = dx.reshape((n_ics * t.size, -1))
    data['z'] = z.reshape((n_ics * t.size, -1))
    data['dz'] = dz.reshape((n_ics * t.size, -1))

    return data

def simulate_spring(n_ics, f, t, spread):
    z = np.zeros((n_ics, t.size, 2))
    dz = np.zeros(z.shape)

    z1range = np.array([-1., 1.])*spread
    z2range = np.array([-0.1, 0.1])*spread
    i = 0
    while i < n_ics:
        # Random initial condition
        z0 = np.array([(z1range[1] - z1range[0]) * np.random.rand() + z1range[0],
                       (z2range[1] - z2range[0]) * np.random.rand() + z2range[0]])
        z0 = z0.reshape(1, -1)
        
        z[i] = solve_ivp(f, t_span=(t[0], t[-1]), y0=z0[0, :], t_eval=t, method='DOP853').y.T
        dz[i] = np.array([f(t[j], z[i, j]) for j in range(len(t))])

        i += 1
    return z, dz



def generate_spring_data(n_ics, spread=1, linear=False, b=0.2, m=0.5, k=1.):

    # state space representation of the spring
    # z_dot = [z1_dot, z2_dot]
    # z1_dot = z2
    # z2_dot = -b/m * z2 - (k / m) * z1
    f = lambda t, z: [z[1], -b/m * z[1] - (k / m) * z[0]]

    wn = np.sqrt(k / m)
    zeta = b/2/np.sqrt(m*k)
    Ts = 0.8*4/zeta/wn
    freq = wn / (2 * np.pi)
    t_step = 1 / (80 * freq) # ~50ms step for default parameters
    #t_step = 0.01
    t = np.arange(0, Ts, t_step)
    z, dz =  simulate_spring(n_ics, f, t, spread)
    x, dx = spring_to_higher_dimensions(z, dz, t, linear)


    return t, x, dx, z, dz


def spring_to_higher_dimensions(z, dz, t, linear, n_dimension = 60):
    n_ics = z.shape[0]
    n_steps = t.size
    d = z.shape[2]

    L = 1
    y_spatial = np.linspace(-L,L, n_dimension)

    modes = np.zeros((2*d, n_dimension))
    for i in range(2*d):
        modes[i] = legendre(i)(y_spatial)

    x1 = np.zeros((n_ics,n_steps,n_dimension))
    x2 = np.zeros((n_ics,n_steps,n_dimension))
    x3 = np.zeros((n_ics,n_steps,n_dimension))
    x4 = np.zeros((n_ics,n_steps,n_dimension))

    x = np.zeros((n_ics,n_steps, n_dimension))
    dx = np.zeros(x.shape)
    for i in range(n_ics):
        for j in range(n_steps):
            x1[i,j] = modes[0]*z[i,j,0]
            x2[i,j] = modes[1]*z[i,j,1]
            x3[i,j] = modes[2]*z[i,j,0]**3
            x4[i,j] = modes[3]*z[i,j,1]**3
            
            x[i,j] = x1[i,j] + x2[i,j]
            if not linear:
                x[i,j] += x3[i,j] + x4[i,j]

            dx[i,j] = modes[0]*dz[i,j,0] + modes[1]*dz[i,j,1] 
            if not linear:
                dx[i,j] += modes[2]*3*(z[i,j,0]**2)*dz[i,j,0] + modes[3]*3*(z[i,j,1]**2)*dz[i,j,1]

    return x, dx

