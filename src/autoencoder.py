import tensorflow as tf
tf.compat.v1.disable_eager_execution()


def full_network(params):
    """
    Define the full network architecture.

    Arguments:
        params - Dictionary object containing the parameters that specify the training.
        See README file for a description of the parameters.

    Returns:
        network - Dictionary containing the tensorflow objects that make up the network.
    """
    input_dim = params['input_dim']
    latent_dim = params['latent_dim']
    activation = params['activation']
    poly_order = params['poly_order']
    include_sine=params.get('include_sine',False)
    library_dim = params['library_dim']

    network = {}

    x = tf.compat.v1.placeholder(tf.float32, shape=[None, input_dim], name='x')
    dx = tf.compat.v1.placeholder(tf.float32, shape=[None, input_dim], name='dx')

    if activation == 'linear':
        z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases = linear_autoencoder(x, input_dim, latent_dim)
    else:
        z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases = nonlinear_autoencoder(x, input_dim, latent_dim, params['widths'], activation=activation)
    
    dz = z_derivative(x, dx, encoder_weights, encoder_biases, activation=activation)
    Theta = sindy_library_tf(z, latent_dim, poly_order, include_sine)
    

    if params['coefficient_initialization'] == 'xavier':
        sindy_coefficients = tf.compat.v1.get_variable('sindy_coefficients', shape=[library_dim,latent_dim], initializer=tf.contrib.layers.xavier_initializer())
    elif params['coefficient_initialization'] == 'specified':
        sindy_coefficients = tf.compat.v1.get_variable('sindy_coefficients', initializer=params['init_coefficients'])
    elif params['coefficient_initialization'] == 'constant':
        sindy_coefficients = tf.compat.v1.get_variable('sindy_coefficients', shape=[library_dim,latent_dim], initializer=tf.constant_initializer(1.0))
    elif params['coefficient_initialization'] == 'normal':
        sindy_coefficients = tf.compat.v1.get_variable('sindy_coefficients', shape=[library_dim,latent_dim], initializer=tf.initializers.random_normal())
    
    if params['sequential_thresholding']:
        coefficient_mask = tf.compat.v1.placeholder(tf.float32, shape=[library_dim,latent_dim], name='coefficient_mask')
        sindy_predict = tf.linalg.matmul(Theta, coefficient_mask*sindy_coefficients)
        network['coefficient_mask'] = coefficient_mask
    else:
        sindy_predict = tf.linalg.matmul(Theta, sindy_coefficients)

    
    dx_decode = z_derivative(z, sindy_predict, decoder_weights, decoder_biases, activation=activation)

    network['x'] = x
    network['dx'] = dx
    network['z'] = z
    network['dz'] = dz
    network['x_decode'] = x_decode
    network['dx_decode'] = dx_decode
    network['encoder_weights'] = encoder_weights
    network['encoder_biases'] = encoder_biases
    network['decoder_weights'] = decoder_weights
    network['decoder_biases'] = decoder_biases
    network['Theta'] = Theta
    network['sindy_coefficients'] = sindy_coefficients


    network['dz_predict'] = sindy_predict

    return network


def define_loss(network, params):
    """
    Create the loss functions.

    Arguments:
        network - Dictionary object containing the elements of the network architecture.
        This will be the output of the full_network() function.
    """
    x = network['x']
    x_decode = network['x_decode']
    if params['model_order'] == 1:
        dz = network['dz']
        dz_predict = network['dz_predict']
        dx = network['dx']
        dx_decode = network['dx_decode']
    else:
        ddz = network['ddz']
        ddz_predict = network['ddz_predict']
        ddx = network['ddx']
        ddx_decode = network['ddx_decode']
    sindy_coefficients = params['coefficient_mask']*network['sindy_coefficients']

    losses = {}
    losses['decoder'] = tf.reduce_mean((x - x_decode)**2)
    if params['model_order'] == 1:
        losses['sindy_z'] = tf.reduce_mean((dz - dz_predict)**2)
        losses['sindy_x'] = tf.reduce_mean((dx - dx_decode)**2)
    else:
        losses['sindy_z'] = tf.reduce_mean((ddz - ddz_predict)**2)
        losses['sindy_x'] = tf.reduce_mean((ddx - ddx_decode)**2)
    losses['sindy_regularization'] = tf.reduce_mean(tf.abs(sindy_coefficients))
    loss = params['loss_weight_decoder'] * losses['decoder'] \
           + params['loss_weight_sindy_z'] * losses['sindy_z'] \
           + params['loss_weight_sindy_x'] * losses['sindy_x'] \
           + params['loss_weight_sindy_regularization'] * losses['sindy_regularization']

    loss_refinement = params['loss_weight_decoder'] * losses['decoder'] \
                      + params['loss_weight_sindy_z'] * losses['sindy_z'] \
                      + params['loss_weight_sindy_x'] * losses['sindy_x']

    return loss, losses, loss_refinement


def linear_autoencoder(x, input_dim, latent_dim):
    z,encoder_weights,encoder_biases = build_network_layers(x, input_dim, latent_dim, [], None, 'encoder')
    x_decode,decoder_weights,decoder_biases = build_network_layers(z, latent_dim, input_dim, [], None, 'decoder')

    return z, x_decode, encoder_weights, encoder_biases,decoder_weights,decoder_biases


def nonlinear_autoencoder(x, input_dim, latent_dim, widths, activation='elu'):
    """
    Construct a nonlinear autoencoder.

    Arguments:

    Returns:
        z -
        x_decode -
        encoder_weights - List of tensorflow arrays containing the encoder weights
        encoder_biases - List of tensorflow arrays containing the encoder biases
        decoder_weights - List of tensorflow arrays containing the decoder weights
        decoder_biases - List of tensorflow arrays containing the decoder biases
    """
    if activation == 'relu':
        activation_function = tf.nn.relu
    elif activation == 'elu':
        activation_function = tf.nn.elu
    elif activation == 'sigmoid':
        activation_function = tf.sigmoid
    else:
        raise ValueError('invalid activation function')
    
    z,encoder_weights,encoder_biases = build_network_layers(x, input_dim, latent_dim, widths, activation_function, 'encoder')
    x_decode,decoder_weights,decoder_biases = build_network_layers(z, latent_dim, input_dim, widths[::-1], activation_function, 'decoder')

    return z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases


def build_network_layers(input, input_dim, output_dim, widths, activation, name):
    """
    Construct one portion of the network (either encoder or decoder).

    Arguments:
        input - 2D tensorflow array, input to the network (shape is [?,input_dim])
        input_dim - Integer, number of state variables in the input to the first layer
        output_dim - Integer, number of state variables to output from the final layer
        widths - List of integers representing how many units are in each network layer
        activation - Tensorflow function to be used as the activation function at each layer
        name - String, prefix to be used in naming the tensorflow variables

    Returns:
        input - Tensorflow array, output of the network layers (shape is [?,output_dim])
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
    """
    weights = []
    biases = []
    last_width=input_dim
    for i,n_units in enumerate(widths):
        W = tf.get_variable(name+'_W'+str(i), shape=[last_width,n_units],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name+'_b'+str(i), shape=[n_units],
            initializer=tf.constant_initializer(0.0))
        input = tf.matmul(input, W) + b
        if activation is not None:
            input = activation(input)
        last_width = n_units
        weights.append(W)
        biases.append(b)
    W = tf.get_variable(name+'_W'+str(len(widths)), shape=[last_width,output_dim],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name+'_b'+str(len(widths)), shape=[output_dim],
        initializer=tf.constant_initializer(0.0))
    input = tf.matmul(input,W) + b
    weights.append(W)
    biases.append(b)
    return input, weights, biases

def sindy_library_tf(z, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library.

    Arguments:
        z - 2D tensorflow array of the snapshots on which to build the library. Shape is number of
        time points by the number of state variables.
        latent_dim - Integer, number of state variable in z.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.

    Returns:
        2D tensorflow array containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """
    library = [tf.ones(tf.shape(z)[0])]

    for i in range(latent_dim):
        library.append(z[:,i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(tf.multiply(z[:,i], z[:,j]))

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i]*z[:,j]*z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q])

    if include_sine:
        for i in range(latent_dim):
            library.append(tf.sin(z[:,i]))

    return tf.stack(library, axis=1)



def z_derivative(input, dx, weights, biases, activation='elu'):
    """
    Compute the first order time derivatives by propagating through the network.

    Arguments:
        input - 2D tensorflow array, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.

    Returns:
        dz - Tensorflow array, first order time derivatives of the network output.
    """
    dz = dx
    if activation == 'elu':
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            dz = tf.multiply(tf.minimum(tf.exp(input),1.0),
                                  tf.matmul(dz, weights[i]))
            input = tf.nn.elu(input)
        dz = tf.matmul(dz, weights[-1])
    elif activation == 'relu':
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            dz = tf.multiply(tf.to_float(input>0), tf.matmul(dz, weights[i]))
            input = tf.nn.relu(input)
        dz = tf.matmul(dz, weights[-1])
    elif activation == 'sigmoid':
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            input = tf.sigmoid(input)
            dz = tf.multiply(tf.multiply(input, 1-input), tf.matmul(dz, weights[i]))
        dz = tf.matmul(dz, weights[-1])
    else:
        for i in range(len(weights)-1):
            dz = tf.matmul(dz, weights[i])
        dz = tf.matmul(dz, weights[-1])
    return dz

