"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from numpy import array, eye, zeros
from scipy.linalg import block_diag


def order_by_derivative(Q, dim, block_size):
    """Given a matrix Q, ordered assuming state space.

        [x y z x' y' z' x'' y'' z''...]

    return a reordered matrix assuming an ordering of
       [ x x' x'' y y' y'' z z' y'']

    This works for any covariance matrix or state transition function

    Parameters
    ----------
    Q : np.array, square
        The matrix to reorder

    dim : int >= 1

       number of independent state variables. 3 for x, y, z

    block_size : int >= 0
        Size of derivatives. Second derivative would be a block size of 3
        (x, x', x'')
    """

    N = dim * block_size

    D = zeros((N, N))

    Q = array(Q)
    for i, x in enumerate(Q.ravel()):
        f = eye(block_size) * x

        ix, iy = (i // dim) * block_size, (i % dim) * block_size
        D[ix:ix + block_size, iy:iy + block_size] = f

    return D


def Q_discrete_white_noise(dim,
                           dt=1.,
                           var=1.,
                           block_size=1,
                           order_by_dim=True):
    """Returns the Q matrix for the Discrete Constant White Noise Model. dim
    may be either 2, 3, or 4 dt is the time step, and sigma is the variance in
    the noise.

    Q is computed as the G * G^T * variance, where G is the process noise per
    time step. In other words, G = [[.5dt^2][dt]]^T for the constant velocity
    model.

    Parameters
    -----------

    dim : int (2, 3, or 4)
        dimension for Q, where the final dimension is (dim x dim)

    dt : float, default=1.0
        time step in whatever units your filter is using for time. i.e. the
        amount of time between innovations

    var : float, default=1.0
        variance in the noise

    block_size : int >= 1
        If your state variable contains more than one dimension, such as
        a 3d constant velocity model [x x' y y' z z']^T, then Q must be
        a block diagonal matrix.

    order_by_dim : bool, default=True
        Defines ordering of variables in the state vector. `True` orders
        by keeping all derivatives of each dimensions)

        [x x' x'' y y' y'']

        whereas `False` interleaves the dimensions

        [x y z x' y' z' x'' y'' z'']


    Examples
    --------
    >>> # constant velocity model in a 3D world with a 10 Hz update rate
    >>> Q_discrete_white_noise(2, dt=0.1, var=1., block_size=3)
    array([[0.000025, 0.0005  , 0.      , 0.      , 0.      , 0.      ],
           [0.0005  , 0.01    , 0.      , 0.      , 0.      , 0.      ],
           [0.      , 0.      , 0.000025, 0.0005  , 0.      , 0.      ],
           [0.      , 0.      , 0.0005  , 0.01    , 0.      , 0.      ],
           [0.      , 0.      , 0.      , 0.      , 0.000025, 0.0005  ],
           [0.      , 0.      , 0.      , 0.      , 0.0005  , 0.01    ]])

    References
    ----------

    Bar-Shalom. "Estimation with Applications To Tracking and Navigation".
    John Wiley & Sons, 2001. Page 274.
    """

    if dim not in [2, 3, 4]:
        raise ValueError('dim must be between 2 and 4')

    if dim == 2:
        Q = [[.25 * dt**4, .5 * dt**3], [.5 * dt**3, dt**2]]
    elif dim == 3:
        Q = [[.25 * dt**4, .5 * dt**3, .5 * dt**2], [.5 * dt**3, dt**2, dt],
             [.5 * dt**2, dt, 1]]
    else:
        Q = [[(dt**6) / 36, (dt**5) / 12, (dt**4) / 6, (dt**3) / 6],
             [(dt**5) / 12, (dt**4) / 4, (dt**3) / 2, (dt**2) / 2],
             [(dt**4) / 6, (dt**3) / 2, dt**2, dt],
             [(dt**3) / 6, (dt**2) / 2, dt, 1.]]

    if order_by_dim:
        return block_diag(*[Q] * block_size) * var
    return order_by_derivative(array(Q), dim, block_size) * var


def Q_continuous_white_noise(dim,
                             dt=1.,
                             spectral_density=1.,
                             block_size=1,
                             order_by_dim=True):
    """Returns the Q matrix for the Discretized Continuous White Noise Model.
    dim may be either 2, 3, 4, dt is the time step, and sigma is the variance
    in the noise.

    Parameters
    ----------

    dim : int (2 or 3 or 4)
        dimension for Q, where the final dimension is (dim x dim)
        2 is constant velocity, 3 is constant acceleration, 4 is
        constant jerk

    dt : float, default=1.0
        time step in whatever units your filter is using for time. i.e. the
        amount of time between innovations

    spectral_density : float, default=1.0
        spectral density for the continuous process

    block_size : int >= 1
        If your state variable contains more than one dimension, such as
        a 3d constant velocity model [x x' y y' z z']^T, then Q must be
        a block diagonal matrix.

    order_by_dim : bool, default=True
        Defines ordering of variables in the state vector. `True` orders
        by keeping all derivatives of each dimensions)

        [x x' x'' y y' y'']

        whereas `False` interleaves the dimensions

        [x y z x' y' z' x'' y'' z'']

    Examples
    --------

    >>> # constant velocity model in a 3D world with a 10 Hz update rate
    >>> Q_continuous_white_noise(2, dt=0.1, block_size=3)
    array([[0.00033333, 0.005     , 0.        , 0.        , 0.        , 0.        ],
           [0.005     , 0.1       , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.00033333, 0.005     , 0.        , 0.        ],
           [0.        , 0.        , 0.005     , 0.1       , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.00033333, 0.005     ],
           [0.        , 0.        , 0.        , 0.        , 0.005     , 0.1       ]])
    """

    if dim not in [2, 3, 4]:
        raise ValueError('dim must be between 2 and 4')

    if dim == 2:
        Q = [[(dt**3) / 3., (dt**2) / 2.], [(dt**2) / 2., dt]]
    elif dim == 3:
        Q = [[(dt**5) / 20., (dt**4) / 8., (dt**3) / 6.],
             [(dt**4) / 8., (dt**3) / 3., (dt**2) / 2.],
             [(dt**3) / 6., (dt**2) / 2., dt]]

    else:
        Q = [[(dt**7) / 252., (dt**6) / 72., (dt**5) / 30., (dt**4) / 24.],
             [(dt**6) / 72., (dt**5) / 20., (dt**4) / 8., (dt**3) / 6.],
             [(dt**5) / 30., (dt**4) / 8., (dt**3) / 3., (dt**2) / 2.],
             [(dt**4) / 24., (dt**3) / 6., (dt**2 / 2.), dt]]

    if order_by_dim:
        return block_diag(*[Q] * block_size) * spectral_density

    return order_by_derivative(array(Q), dim, block_size) * spectral_density
