import joblib, os, json

def save_model( model_folder, classifier, scaler,
                sigmas=[.1,.5,1,2.5,5,7.5,10],
                down_shape=-1,
                edge_size=5,
                fraction=0.1,
                bias=-1,
                feature_mode='ilastik',
                deep=False,
                unet=False ):
    r'''
    save a previously generated machine learning model in the "model_folder" input path:
    * model_folder\classifier.pkl: logistic classifier model
    * model_folder\classifier.keras: MLP model (deep=True)
    * model_folder\unet.keras: U-Net model (unet=True)
    * model_folder\scaler.pkl: scaler used to normalize the trainingset
    * model_folder\params.json: parameters used for training

    '''

    # Make it work for Python 2+3 and with Unicode
    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str

    if unet:
        classifier.save(os.path.join(model_folder, "unet.keras"))
    elif not deep:
        joblib.dump(classifier, os.path.join(model_folder,'classifier.pkl'))
    else:
        classifier.save(os.path.join(model_folder, "classifier.keras"))
    
    # the U-Net has no scaler (scaler=None), the file is still saved to mark the model as trained
    joblib.dump(scaler, os.path.join(model_folder,'scaler.pkl'))
    
    params = {  'sigmas': sigmas,
                'down_shape': down_shape,
                'edge_size': edge_size,
                'fraction': fraction,
                'bias': bias,
                'feature_mode': feature_mode  }
    with open(os.path.join(model_folder,'params.json'), 'w', encoding='utf8') as f:
        str_ = json.dumps(params,
                    indent=4, sort_keys=True,
                    separators=(',', ': '), ensure_ascii=False)
        f.write(to_unicode(str_))

def load_model( model_folder, deep=False, unet=False ):
    r'''
    load a previously saved machine learning model from the "model_folder" input path:
    * model_folder\classifier.pkl: logistic classifier model
    * model_folder\classifier.keras: MLP model (deep=True)
    * model_folder\unet.keras: U-Net model (unet=True)
    * model_folder\scaler.pkl: scaler used to normalize the trainingset (None for the U-Net)
    * model_folder\params.json: parameters used for training

    '''
    if unet:
        from tensorflow import keras
        try:
            classifier = keras.models.load_model(os.path.join(model_folder, "unet.keras"))
        except:
            return None, None, None
    elif not deep:
        try:
            classifier = joblib.load(os.path.join(model_folder,'classifier.pkl'))
        except:
            return None, None, None
    else:
        from tensorflow import keras
        try:
            classifier = keras.models.load_model(os.path.join(model_folder, "classifier.keras"))
        except:
            return None, None, None

    scaler = joblib.load(os.path.join(model_folder,'scaler.pkl'))
    with open(os.path.join(model_folder,'params.json'), 'r') as f:
        params = json.load(f)
    
    ### patch to take into account the old definition of down_shape
    if params['down_shape']==500:
        params['down_shape'] = 500./2160.
    return classifier, scaler, params

