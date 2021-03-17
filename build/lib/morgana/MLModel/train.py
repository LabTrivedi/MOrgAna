import tqdm, os
import numpy as np
from skimage import transform, morphology
from sklearn import preprocessing, linear_model, neural_network

from morgana.ImageTools import processfeatures

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
        _input = [i.astype(np.float) for i in _input]
        _input = [ transform.resize(i, (int(i.shape[0]*down_shape),int(i.shape[1]*down_shape)) , preserve_range=True) for i in _input ]
        gt = [i.astype(np.float) for i in gt]
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
        Y = 1.*(y_in>np.min(y_in))
        edge = Y - morphology.binary_dilation(Y, morphology.disk(1))
        edge = morphology.binary_dilation(edge, morphology.disk(edge_size))
        Y = 1 * np.logical_or(Y, edge) + edge

        # flatten the images and normalize to -std:+std
        X = np.transpose(np.reshape(X, (X.shape[0], np.prod(shape)))) # flatten the image feature

        Y = np.reshape(Y, np.prod(shape)) # flatten the ground truth
        edge = np.reshape(edge, np.prod(shape)) # flatten the edge

        # extract coordinates with the right probability distribution
        if (bias>0) and (bias<=1):
            prob = (Y>0).astype(np.float64)
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
        Y = keras.utils.to_categorical(Y, num_classes=n_classes, dtype='int')

        # Create the model
        # Define Sequential model with 3 layers
        model_layers = [layers.Dense(hidden[i], activation="relu", name="layer%d"%i) for i in range(len(hidden))]
        model_layers.append(layers.Dense(n_classes, activation='softmax', name="layer%d"%len(hidden)))

        classifier = keras.Sequential( model_layers )

        # Configure the model and start training
        classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        classifier.fit(X, Y, epochs=epochs, batch_size=1024, verbose=1, validation_split=0.1, shuffle=True)

    return classifier
