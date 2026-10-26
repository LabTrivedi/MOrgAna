import tqdm, os
import numpy as np
from skimage import transform, morphology
from sklearn import preprocessing, linear_model, neural_network

from morgana.ImageTools import processfeatures

def make_labels( y_in, edge_size=5 ):
    '''
    Convert a ground truth mask into 3 classes: 0 background, 1 inside, 2 edge.
    Also returns the edge mask, used to give more weight to the border pixels.

    '''
    Y = 1.*(y_in>np.min(y_in))
    edge = Y - morphology.binary_dilation(Y, morphology.disk(1))
    edge = morphology.binary_dilation(edge, morphology.disk(edge_size))
    Y = 1 * np.logical_or(Y, edge) + edge

    return Y, edge

def generate_training_set( _input, gt,
                        sigmas=[.1,.5,1,2.5,5,7.5,10],
                        down_shape=-1,
                        edge_size=5,
                        fraction=0.1,
                        bias=-1,
                        edge_weight=5,
                        feature_mode='ilastik' ):

    '''
    Note: _input and gt should have shape (n_images,x,y)

    '''

    # ### if only one image, make sure it's a 3D image still
    # if len(_input.shape)==2:
    #     _input = np.expand_dims(_input, axis=0)
    #     gt = np.expand_dims(gt, axis=0)

    ### read in kwargs
    if down_shape != -1:
        ### reshape input and ground truth
        _input = [i.astype(float) for i in _input]
        _input = [ transform.resize(i, (int(i.shape[0]*down_shape),int(i.shape[1]*down_shape)) , preserve_range=True) for i in _input ]
        gt = [i.astype(float) for i in gt]
        gt = [ transform.resize(i, (int(i.shape[0]*down_shape),int(i.shape[1]*down_shape)), order=0, preserve_range=False) for i in gt ]
    shapes = [i.shape for i in _input]

    ### generate empty vector for trainings
    n_coords_per_image = [(fraction*np.prod(i.shape)).astype(int) for i in _input]
    n_coords = int( np.sum(n_coords_per_image))
    print('Number of images: %d'%len(_input))
    print('Number of pixels extracted per image (%d%%):'%(100*fraction),n_coords_per_image)
    if feature_mode=='ilastik':
        print('Number of features per image: %d'%(len(sigmas)*4+1))
        X_train = np.zeros((n_coords,len(sigmas)*4+1))
    elif feature_mode=='daisy':
        print('Number of features per image:%d'%((5*8+1)*8+len(sigmas)*4+1))
        X_train = np.zeros((n_coords,(5*8+1)*8+len(sigmas)*4+1))
    Y_train = np.zeros(n_coords)
    weight_train = np.zeros(n_coords)

    print('Extracting features...')
    start = 0
    for i in tqdm.tqdm(range(len(_input))):
        stop = start + n_coords_per_image[i]
        shape = shapes[i]
        x_in, y_in = _input[i], gt[i]

        # compute all features
        X = processfeatures.get_features(x_in, sigmas, feature_mode=feature_mode)

        # find the edges of the mask and give them more weight
        Y, edge = make_labels(y_in, edge_size)

        # flatten the images and normalize to -std:+std
        X = np.transpose(np.reshape(X, (X.shape[0], np.prod(shape)))) # flatten the image feature

        Y = np.reshape(Y, np.prod(shape)) # flatten the ground truth
        edge = np.reshape(edge, np.prod(shape)) # flatten the edge

        # extract coordinates with the right probability distribution
        if (bias>0) and (bias<=1):
            prob = (Y>0).astype(float)
            Nw = np.sum(prob)
            Nd = np.prod(prob.shape)-Nw
            probW = bias*prob/Nw
            probD = (1-bias)*(prob==0)/Nd
            prob = probW+probD
        else:
            prob = np.ones(Y.shape)/np.prod(Y.shape)
        coords = np.random.choice(np.arange(X.shape[0]), n_coords_per_image[i], p=prob)

        # populate training dataset, ground truth and weight
        X_train[start:stop,:] = X[coords,:]
        Y_train[start:stop] = Y[coords]
        weight = edge_weight * edge + 1
        weight_train[start:stop] = weight[coords]
        start = n_coords_per_image[i]

    scaler = preprocessing.RobustScaler(quantile_range=(1.0, 99.0))
    scaler.fit(X_train) # normalize
    X_train = scaler.transform(X_train)

    # shuffle the training set    
    p = np.random.permutation(X_train.shape[0])
    X_train = X_train[p,:]
    Y_train = Y_train[p]

    return X_train, Y_train, weight_train, scaler

def train_classifier( X, Y, w, deep=False, epochs=50, n_classes = 3, hidden=(350,50) ):

    # train the classifier
    if not deep:
        print('Training of Logistic Regression classifier...')
        classifier = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
        classifier.fit(X, Y, sample_weight=w)
    else:
        print('Training of MLP classifier...')
        from tensorflow.keras import layers
        from tensorflow import keras
        Y = keras.utils.to_categorical(Y, num_classes=n_classes)

        # Create the model
        # Define Sequential model with 3 layers
        model_layers = [layers.Dense(hidden[i], activation="relu", name="layer%d"%i) for i in range(len(hidden))]
        model_layers.append(layers.Dense(n_classes, activation='softmax', name="layer%d"%len(hidden)))

        classifier = keras.Sequential( model_layers )

        # Configure the model and start training
        classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        classifier.fit(X, Y, epochs=epochs, batch_size=1024, verbose=1, validation_split=0.1, shuffle=True)

    return classifier

def normalize_image( _input, percs=(1,99) ):
    '''
    Rescale an image between 0 and 1 using its own percentiles.
    Used by the U-Net instead of the scaler, so it should also be applied before prediction.

    '''
    _input = _input.astype(float)
    low, high = np.percentile(_input, percs)
    _input = np.clip((_input-low)/max(high-low,1e-8), 0, 1)

    return _input

def build_unet( n_classes=3, depth=4, filters=16 ):
    '''
    Fully convolutional U-Net: it accepts images of any size whose sides are multiples of 2**depth.

    '''
    from tensorflow.keras import layers
    from tensorflow import keras

    def conv_block(x, n):
        for _ in range(2):
            x = layers.Conv2D(n, 3, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
        return x

    inputs = keras.Input(shape=(None, None, 1))
    x = inputs
    skips = []
    for i in range(depth):
        x = conv_block(x, filters*2**i)
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)
    x = conv_block(x, filters*2**depth)
    for i in reversed(range(depth)):
        x = layers.Conv2DTranspose(filters*2**i, 2, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, skips[i]])
        x = conv_block(x, filters*2**i)
    outputs = layers.Conv2D(n_classes, 1, activation='softmax')(x)

    return keras.Model(inputs, outputs, name='unet')

def sample_patches( _input, labels, weights, n_patches, patch_size ):
    '''
    Draw random patches with random flips and 90 degree rotations.

    '''
    X = np.zeros((n_patches, patch_size, patch_size, 1), dtype=np.float32)
    Y = np.zeros((n_patches, patch_size, patch_size), dtype=np.uint8)
    W = np.zeros((n_patches, patch_size, patch_size), dtype=np.float32)
    for p in range(n_patches):
        i = np.random.randint(len(_input))
        r = np.random.randint(_input[i].shape[0]-patch_size+1)
        c = np.random.randint(_input[i].shape[1]-patch_size+1)
        x = _input[i][r:r+patch_size, c:c+patch_size]
        y = labels[i][r:r+patch_size, c:c+patch_size]
        w = weights[i][r:r+patch_size, c:c+patch_size]
        k = np.random.randint(4)
        x, y, w = np.rot90(x,k), np.rot90(y,k), np.rot90(w,k)
        if np.random.rand()<0.5:
            x, y, w = x[:,::-1], y[:,::-1], w[:,::-1]
        X[p,...,0], Y[p], W[p] = x, y, w

    return X, Y, W

def train_unet( _input, gt,
                down_shape=-1,
                edge_size=5,
                edge_weight=5,
                epochs=50,
                patches_per_epoch=256,
                patch_size=256,
                batch_size=8,
                depth=4,
                filters=16,
                n_classes=3 ):
    '''
    Train a U-Net on whole images, with the same 3 classes as the pixel classifier
    (background, inside, edge) and the same extra weight on edge pixels.
    Note: _input and gt are lists of 2D images (they can have different shapes).
    Input images are normalized with normalize_image, apply it before predicting as well.

    '''
    from tensorflow import keras

    if patch_size % 2**depth != 0:
        raise ValueError('patch_size must be a multiple of 2**depth')

    _input = [normalize_image(i) for i in _input]
    if down_shape != -1:
        _input = [ transform.resize(i, (int(i.shape[0]*down_shape),int(i.shape[1]*down_shape)), preserve_range=True) for i in _input ]
        gt = [ transform.resize(i.astype(float), (int(i.shape[0]*down_shape),int(i.shape[1]*down_shape)), order=0, preserve_range=False) for i in gt ]

    labels, weights = [], []
    for i in range(len(_input)):
        Y, edge = make_labels(gt[i], edge_size)
        labels.append(Y)
        weights.append(edge_weight * edge + 1.)

    # pad images smaller than a patch
    for i in range(len(_input)):
        pad = [(0, max(0, patch_size-s)) for s in _input[i].shape]
        _input[i] = np.pad(_input[i], pad, mode='reflect')
        labels[i] = np.pad(labels[i], pad, mode='reflect')
        weights[i] = np.pad(weights[i], pad, mode='reflect')

    print('Number of images: %d'%len(_input))
    print('Training of U-Net...')
    model = build_unet(n_classes=n_classes, depth=depth, filters=filters)
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

    # new random patches at every epoch, so that few images still give varied training data
    for epoch in range(epochs):
        X, Y, W = sample_patches(_input, labels, weights, patches_per_epoch, patch_size)
        model.fit(X, Y, sample_weight=W, batch_size=batch_size,
                  initial_epoch=epoch, epochs=epoch+1, verbose=1, shuffle=True)

    return model
