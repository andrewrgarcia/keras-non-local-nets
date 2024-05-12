from tensorflow.keras.layers import Layer, Conv1D, Conv2D, Conv3D, Reshape, Dot, Activation, Lambda, MaxPool1D, Add
from tensorflow.keras import backend as K

class NonLocalBlock(Layer):
    def __init__(self, intermediate_dim=None, compression=2, mode='embedded', add_residual=True, **kwargs):
        """
        Initializes a NonLocalBlock with configurable parameters and operational mode.

        Parameters
        ----------
        intermediate_dim : int, optional
            The dimension of the intermediate representation in the convolution layers. If None, it defaults to half of the channels in the input shape.
        compression : float, optional
            The factor by which to compress feature dimensions during pooling operations. Defaults to 2, halving the feature dimensions.
        mode : str, optional
            Operational mode of the block. Supported modes are 'gaussian', 'dot', 'embedded', and 'concatenate'. Defaults to 'embedded'.
        add_residual : bool, optional
            Whether to include a residual connection that adds the input to the output of the block. Defaults to True.
        kwargs : dict
            Additional keyword arguments inherited from tf.keras.layers.Layer.
        """
        super(NonLocalBlock, self).__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.compression = compression if compression is not None else 1
        self.mode = mode
        self.add_residual = add_residual
        self.conv_layers = {}

    def build(self, input_shape):
        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
        channels = input_shape[channel_dim]
        self.intermediate_dim = channels // 2 if self.intermediate_dim is None else int(self.intermediate_dim)
        if self.intermediate_dim < 1:
            self.intermediate_dim = 1

        # Instantiate convolution layers here to be used in the call method
        rank = len(input_shape)
        self.conv_layers['theta'] = self._create_conv_layer(rank, self.intermediate_dim)
        self.conv_layers['phi'] = self._create_conv_layer(rank, self.intermediate_dim)
        self.conv_layers['g'] = self._create_conv_layer(rank, self.intermediate_dim)
        self.conv_layers['final'] = self._create_conv_layer(rank, channels)

    def call(self, inputs):
        # Use the convolution layers created in build
        theta = self.conv_layers['theta'](inputs)
        phi = self.conv_layers['phi'](inputs)
        g = self.conv_layers['g'](inputs)

        channels = self._initialize_dimensions(inputs)

        f = self._instantiate_f(channels, theta, phi, inputs)
        g = self._handle_g(g)
        y = self._non_local_operation_neural(f, g, inputs)
        z = self._non_local_block(y, inputs)

        return z

    def _initialize_dimensions(self, inputs):
        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
        channels = K.int_shape(inputs)[channel_dim]

        if self.mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot`, or `concatenate`')
        return channels

    def _instantiate_f(self, channels, theta, phi, inputs):
        if self.mode == 'gaussian':
            x = Reshape((-1, channels))(inputs)
            f = Dot(axes=2)([x, x])
            f = Activation('softmax')(f)
        elif self.mode == 'dot':
            theta = Reshape((-1, self.intermediate_dim))(theta)
            phi = Reshape((-1, self.intermediate_dim))(phi)
            f = Dot(axes=2)([theta, phi])
            f = Lambda(lambda z: (1. / float(K.int_shape(f)[-1])) * z)(f)  # reintroduced scaling
            f = Activation('softmax')(f)
        elif self.mode == 'embedded':
            # Embedded Gaussian instantiation
            theta = Reshape((-1, self.intermediate_dim))(theta)
            phi = Reshape((-1, self.intermediate_dim))(phi)

            if self.compression > 1:
                phi = MaxPool1D(self.compression)(phi)  # Apply compression as max pooling

            f = Dot(axes=2)([theta, phi])
            f = Activation('softmax')(f)
        else:
            # If concatenate mode or any other mode, handle accordingly
            raise NotImplementedError('Concatenate model has not been implemented yet')
        
        return f

    
    def _handle_g(self, g):
        # g has already been instantiated with a simple linear embedding at the beginning of call 
        g = Reshape((-1, self.intermediate_dim))(g)
        if self.compression > 1 and self.mode == 'embedded':
            g = MaxPool1D(self.compression)(g)
        return g
    
    def _non_local_operation_neural(self, f, g, inputs):
        # Final output path 
        y = Dot(axes=[2, 1])([f, g])
        y = Reshape(K.int_shape(inputs)[1:-1] + (self.intermediate_dim,))(y)
        return y 

    def _non_local_block(self, y, inputs):
        # Final combination and residual connection
        y = self.conv_layers['final'](y)
        if self.add_residual:
            y = Add()([inputs, y])
        return y 

    def _create_conv_layer(self, rank, channels):
        if rank == 3:
            return Conv1D(channels, 1, padding='same', use_bias=False, kernel_initializer='he_normal')
        elif rank == 4:
            return Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')
        elif rank == 5:
            return Conv3D(channels, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(NonLocalBlock, self).get_config()
        config.update({
            "intermediate_dim": self.intermediate_dim,
            "compression": self.compression,
            "mode": self.mode,
            "add_residual": self.add_residual
        })
        return config
