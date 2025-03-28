import os
import math
import requests
import pandas as pd
import numpy as np


def get_attr(obj, string):
    return getattr(obj, string)


class ILottoCSV(object):

    def __init__(
        self, input_csv="Lotto.csv", output_csv="input/lottoIL_filt.csv"
    ) -> None:
        self.file_name = input_csv
        self.out_file = output_csv
        self.header = [
            "Date",
            "Ball_1",
            "Ball_2",
            "Ball_3",
            "Ball_4",
            "Ball_5",
            "Ball_6",
            "Ball_Bonus",
        ]
        self.ball_numbers = 37
        self.strong_numbers = 7
        self.filter_csv()

    def filter_csv(self):
        names = self.header
        lotto = pd.read_csv(self.file_name, encoding="latin-1")
        curr_names = lotto.columns
        lotto.drop(curr_names[0], axis=1, inplace=True)

        cnt_idx = 0
        for column_headers in lotto.columns:
            if cnt_idx > len(names) - 1:
                lotto.drop(column_headers, axis=1, inplace=True)
            else:
                lotto.rename(columns={column_headers: names[cnt_idx]}, inplace=True)
                cnt_idx += 1

        lotto = pd.DataFrame(lotto).set_index(names[0])

        lotto.drop(
            lotto[
                (get_attr(lotto, names[1]) > self.ball_numbers)
                | (get_attr(lotto, names[2]) > self.ball_numbers)
                | (get_attr(lotto, names[3]) > self.ball_numbers)
                | (get_attr(lotto, names[4]) > self.ball_numbers)
                | (get_attr(lotto, names[5]) > self.ball_numbers)
                | (get_attr(lotto, names[6]) > self.ball_numbers)
                | (get_attr(lotto, names[7]) > self.strong_numbers)
            ].index,
            inplace=True,
        )

        lotto.to_csv(self.out_file)


# beam search
def beam_search_decoder(data, k, replace=True):
    sequences = [[list(), 0.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            best_k = np.argsort(row)[-k:]
            for j in best_k:
                candidate = [seq + [j], score + math.log(row[j])]
                if replace:
                    all_candidates.append(candidate)
                elif (replace == False) and (
                    len(set(candidate[0])) == len(candidate[0])
                ):
                    all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        # select k best
        sequences = ordered[:k]
    return sequences


def fetch_dataset(
    orig_lotto_csv="input/Orig_IL_lotto.csv",
    lotto_csv_file="input/lotto_IL_filtered.csv",
):

    csv_url = "https://pais.co.il/Lotto/lotto_resultsDownload.aspx"
    base_dir = os.path.dirname(orig_lotto_csv)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    r = requests.get(csv_url)
    with open(orig_lotto_csv, "wb") as f:
        f.write(r.content)
    ILottoCSV(orig_lotto_csv, lotto_csv_file)
    lotto_ds = pd.read_csv(lotto_csv_file, index_col="Date")
    return lotto_ds


def train_test_split(lotto_ds, test_size=50, w=10):
    data = lotto_ds.values - 1
    train = data[test_size:]
    test = data[:test_size]

    X_train = []
    y_train = []
    for i in range(w, len(train)):
        X_train.append(train[i - w : i, :])
        y_train.append(train[i])

    X_train, y_train = np.array(X_train), np.array(y_train)
    inputs = data[data.shape[0] - test.shape[0] - w :]
    X_test = []
    for i in range(w, inputs.shape[0]):
        X_test.append(inputs[i - w : i, :])
    X_test = np.array(X_test)
    y_test = test

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    test = ILottoCSV()
