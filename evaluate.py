import numpy as np
import tensorflow as tf
from mdutils.mdutils import MdUtils

from helpers import beam_search_decoder
from helpers import fetch_dataset, train_test_split


def evaluate(model_path, X_test, y_test, beam_width=10):

    mdFile = MdUtils(file_name="README", title="ILotto")
    mdFile.new_paragraph("Israel Lotto predictor")
    mdFile.new_paragraph()
    mdFile.new_header(level=1, title="Prediction")

    model = tf.keras.models.load_model(model_path)
    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=2)

    X_latest = X_test[0][1:]
    X_latest = np.concatenate([X_latest, y_test[0].reshape(1, 7)], axis=0)
    X_latest = X_latest.reshape(1, X_latest.shape[0], X_latest.shape[1])
    pred_latest = model.predict(X_latest)
    pred_latest = np.squeeze(pred_latest)
    pred_latest_greedy = np.argmax(pred_latest, axis=1)
    mdFile.new_line(str(pred_latest_greedy + 1), bold_italics_code="bic")
    replace = True
    mdFile.new_paragraph()
    mdFile.new_header(level=2, title="Beam Serarch")
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

    mdFile.new_paragraph()
    mdFile.new_header(level=2, title="Test set validation")
    mdFile.new_paragraph()
    list_of_strings = ["Prediction", "GoundTruth"]
    for p, y in zip(pred, y_test):
        list_of_strings.extend([p, y])

    mdFile.new_table(
        columns=2, rows=len(pred) + 1, text=list_of_strings, text_align="center"
    )
    mdFile.create_md_file()


if __name__ == "__main__":

    lotto_ds = fetch_dataset()
    X_train, y_train, X_test, y_test = train_test_split(lotto_ds)

    model_path = "model/Ilotto"
    evaluate(model_path, X_test, y_test)
