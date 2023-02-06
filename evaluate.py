import numpy as np
from ilotto import ILotto

from helpers import beam_search_decoder

def evaluate(checkpoint_path, X_test, y_test, beam_width=10):
    model = ILotto()
    model.load_weights(checkpoint_path.format(epoch=0))
    pred = model.predict(X_test)
    pred = np.argmax(pred, axis = 2)
    
    for i in range(y_test.shape[0]):
        print('Prediction:\t', pred[i] + 1)
        print('GoundTruth:\t', y_test[i] + 1)
        print('-' * 40)
    
    X_latest = X_test[0][1:]
    X_latest = np.concatenate([X_latest, y_test[0].reshape(1, 7)], axis = 0)
    X_latest = X_latest.reshape(1, X_latest.shape[0], X_latest.shape[1])
    print(X_latest + 1)
    
    pred_latest = model.predict(X_latest)
    pred_latest = np.squeeze(pred_latest)
    pred_latest_greedy = np.argmax(pred_latest, axis = 1)
    print(pred_latest_greedy + 1)
    
    replace = True

    result = beam_search_decoder(pred_latest, beam_width, replace)
    print('Beam Width:\t', beam_width)
    print('Replace:\t', replace)
    print('-' * 85)
    for seq in result:
        print('Prediction: ', np.array(seq[0]) + 1, '\tLog Likelihood: ', seq[1])
        
    replace = False

    result = beam_search_decoder(pred_latest, beam_width, replace)
    print('Beam Width:\t', beam_width)
    print('Replace:\t', replace)
    print('-' * 85)
    for seq in result:
        print('Prediction: ', np.array(seq[0]) + 1, '\tLog Likelihood: ', seq[1])