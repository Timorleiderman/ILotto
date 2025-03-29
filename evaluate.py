import logging
import numpy as np
from mdutils.mdutils import MdUtils

from train import MODEL_CP_PATH
from logger import setup_logger
from tensorflow.keras.models import load_model
from train import SAVED_MODEL_PATH
from helpers import beam_search_decoder
from helpers import fetch_dataset, train_test_split

setup_logger()
logger = logging.getLogger(__name__)

def evaluate(weights_path, X_test, y_test, beam_width=10):

    logger.info("Loading model from %s", weights_path)
    
    mdFile = MdUtils(file_name="README", title="ILotto")
    mdFile.new_paragraph("Israel Lotto predictor")
    mdFile.new_paragraph()
    mdFile.new_header(level=1, title="Prediction")
    
    model = load_model(SAVED_MODEL_PATH)
    # model.load_weights(weights_path)
    
    logger.info("Model loaded successfully")
    
    pred = model.predict(X_test)
    logger.info("Prediction completed")
    pred = np.argmax(pred, axis=2)


    X_latest = X_test[0][1:]
    X_latest = np.concatenate([X_latest, y_test[0].reshape(1, 7)], axis = 0)
    X_latest = X_latest.reshape(1, X_latest.shape[0], X_latest.shape[1])
    
    pred_latest = model.predict(X_latest)
    
    logger.info("Latest prediction completed")
    pred_latest = np.squeeze(pred_latest)
    mdFile.new_paragraph()
    replace = False
    result = beam_search_decoder(pred_latest, beam_width, replace)
    mdFile.new_line("Beam Width: {} replace {}".format(beam_width, replace))
    mdFile.new_list(
        items=[
            "Prediction: "
            + str(np.array(seq[0]) + 1)
            + "\tLog Likelihood: "
            + str(seq[1])
            for seq in result
        ]
    )

    mdFile.create_md_file()
    logger.info("Markdown file created successfully")

if __name__ == "__main__":

    lotto_ds = fetch_dataset()
    X_train, y_train, X_test, y_test = train_test_split(lotto_ds)

    evaluate(MODEL_CP_PATH, X_test, y_test)
