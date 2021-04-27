import argparse
import os
import pandas as pd
import math
from typing import List


def construct_dataset(num_classes:int, train_test_val_split:List, data_src, save_dir):
    """
    Creates three JSON Files, one for train, test and eval.
    :param num_classes: the number of classes (5, 9 or 13)
    :param train_test_val_split: list of float indicating the split for train, test and evaluation. like [0.7, 0.2, 0.1]
    :param data_src: path to the NER Dataframe from preprocessing (results_df_ner)
    :param save_dir: path to output directory
    :return:
    """
    assert num_classes in [5, 9, 13]
    assert len(train_test_val_split) == 3
    assert sum(train_test_val_split) == 1
    df = pd.read_json(data_src)
    for i, row in df.iterrows():
        df.at[i, 'tokens'] = [str(y) if str(type(y)) != "<class 'str'>" else y for y in row.tokens]
        if num_classes == 5:
            name = '5_classes'
            df.at[i, 'ner_tags'] = [ 'I-ORG' if (y ==  "B-ORG") | (y == 'L-ORG') else y for y in row.ner_tags]
            df.at[i, 'ner_tags'] = [ 'I-MONEY' if (y ==  "B-MONEY") | (y == 'L-MONEY') else y for y in row.ner_tags]
            df.at[i, 'ner_tags'] = [ 'I-GPE' if (y ==  "B-GPE") | (y == 'L-GPE') else y for y in row.ner_tags]
            df.at[i, 'ner_tags'] = [ 'I-DATE' if (y ==  "B-DATE") | (y == 'L-DATE') else y for y in row.ner_tags]
        elif num_classes == 9:
            name = '9_classes'
            df.at[i, 'ner_tags'] = ['I-ORG' if y == 'L-ORG' else y for y in row.ner_tags]
            df.at[i, 'ner_tags'] = ['I-MONEY' if y == 'L-MONEY' else y for y in row.ner_tags]
            df.at[i, 'ner_tags'] = ['I-GPE' if y == 'L-GPE' else y for y in row.ner_tags]
            df.at[i, 'ner_tags'] = ['I-DATE' if y == 'L-DATE' else y for y in row.ner_tags]
        else:
            name = '13_classes'
    df = df.sample(frac=1)
    part = math.floor(df.shape[0] / 100)
    train = df[0: train_test_val_split[0] * 100 * part]
    test = df[train_test_val_split[0] * 100 * part:(train_test_val_split[0] + train_test_val_split[1]) * 100 * part]
    validate = df[(train_test_val_split[0] + train_test_val_split[1]) * 100 * part: df.shape[0]]

    train.to_json(os.path.join(save_dir, f"../data/SROIE/Baseline/train_{name}.json"), orient='table')
    test.to_json(os.path.join(save_dir, f"../data/SROIE/Baseline/test_{name}.json"), orient='table')
    validate.to_json(os.path.join(save_dir, f"../data/SROIE/Baseline/validate_{name}.json"), orient='table')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script zum Erstellen des BERT Datensatzes für SROIE.')

    parser.add_argument('--data_src', type=str, help='Pfad zu den vorverarbeiteten Daten. (results_df_ner.json) ',
                        default=os.path.join(os.path.dirname(__file__), f"../../data/SROIE/results_df_ner.json"))
    parser.add_argument('--classes', choices=[5,9,13], help='Anzahl der Klassen . 5, 9 oder 13. '
                                                            'Unterscheiden sich nach Anzahl der BILUO Präfixe'
                        , default=5)
    parser.add_argument('--output_dir', type=str, help="Pfad zum Zielordner für die fertigen Daten",
                        default=os.path.join(os.path.dirname(__file__), "../data/SROIE/Baseline"))
    parser.add_argument('--train_test_eval_split', type=List,
                        help="Aufteilung des Datensatzes für Trainings, Test und Evaluierung. "
                             "Liste mit 3 Float Einträgen, die in Summe 1 ergeben.",
                        default=[0.7, 0.2, 0.1])
    args = parser.parse_args()

    construct_dataset()

