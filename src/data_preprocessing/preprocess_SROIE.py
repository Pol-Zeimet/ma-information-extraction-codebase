import os
import argparse
import glob
import pandas as pd
import math
import json
from tqdm import tqdm


def match_token_to_label(tokens_src, ground_truth_src, output_dir):
    results_df_ner = pd.DataFrame()
    results_df = pd.DataFrame()
    doc_names = get_doc_names(tokens_src)

    for doc_name in tqdm(doc_names):
        with open(tokens_src + doc_name + '.txt', 'r') as token_file, \
                open(ground_truth_src + doc_name + '.txt', 'br') as truth:
            truth_data = json.loads(truth.read())
            labels = []
            tokens = []
            positions = []

            for token_line in token_file:
                label, pos_vals, token_val = process_token_line(token_line, truth_data)
                labels.append(label)
                tokens.append(token_val.replace('\n', ''))
                positions.append(pos_vals)
            file_results_df = pd.DataFrame({'doc_name': doc_name, 'position': positions, 'tokens': tokens, 'labels': labels})
            results_df, results_df_ner = attach_ner_prefix_to_labels(file_results_df, results_df, results_df_ner, doc_name)

    results_df_ner.to_json(os.path.join(output_dir, 'results_df_ner.json'))
    results_df.to_json(os.path.join(output_dir, 'results_df.json'))


def process_token_line(token_line, truth_data):
    all_values = token_line.split(',')
    pos_vals = all_values[:8]
    token_val = ','.join(all_values[8:])
    eval_token_val = ''.join(token_val.split(' ')).strip()
    label = determine_label_for_token(eval_token_val, truth_data)
    return label, pos_vals, token_val


def match_token_to_label_v2(tokens_src, ground_truth_src, output_dir):
    results_df_ner = pd.DataFrame()
    results_df = pd.DataFrame()
    doc_names = get_doc_names(tokens_src)

    for doc_name in tqdm(doc_names):
        with open(tokens_src + doc_name + '.txt', 'r') as token_file, \
                open(ground_truth_src + doc_name + '.txt', 'br') as truth:
            truth_data = json.loads(truth.read())
            labels = []
            tokens = []
            positions = []

            for token_line in token_file:
                current_x_pos, label, pos_vals, token_array, xticks_per_char = process_token_line_v2(token_line,
                                                                                                     truth_data)
                for token in token_array:
                    if len(token) > 0:
                        labels.append(label)
                        tokens.append(token.replace('\n', ''))
                        next_x_pos = current_x_pos + len(token) * xticks_per_char
                        position = [current_x_pos, pos_vals[1], next_x_pos, pos_vals[3],
                                    next_x_pos, pos_vals[5], current_x_pos, pos_vals[7]]
                        positions.append(position)
                        # compensating for whitespaces between words
                        current_x_pos = next_x_pos + xticks_per_char
            file_results_df = pd.DataFrame({'doc_name': doc_name, 'position': positions,
                                            'tokens': tokens, 'labels': labels})
            results_df, results_df_ner = attach_ner_prefix_to_labels(file_results_df, results_df,
                                                                     results_df_ner, doc_name)

    results_df_ner.to_json(os.path.join(output_dir, 'results_df_ner_v2.json'))
    results_df.to_json(os.path.join(output_dir, 'results_df_v2.json'))


def process_token_line_v2(token_line, truth_data):
    all_values = token_line.split(',')
    pos_vals = all_values[:8]
    min_xpos = pos_vals[0]
    max_xpos = pos_vals[2]
    token_val = ','.join(all_values[8:])
    xticks_per_char = math.ceil((int(max_xpos) - int(min_xpos)) / len(token_val))
    current_xpos = int(min_xpos)
    token_array = token_val.split(' ')
    eval_token_val = ''.join(token_array).strip()
    label = determine_label_for_token(eval_token_val, truth_data)
    return current_xpos, label, pos_vals, token_array, xticks_per_char


def match_token_to_label_v3(tokens_src, ground_truth_src, output_dir):
    results_df_ner = pd.DataFrame()
    results_df = pd.DataFrame()
    doc_names = get_doc_names(tokens_src)

    for doc_name in tqdm(doc_names):
        with open(tokens_src + doc_name + '.txt', 'r') as token_file, \
                open(ground_truth_src + doc_name + '.txt', 'br') as truth:
            truth_data = json.loads(truth.read())
            labels = []
            tokens = []
            positions = []

            for token_line in token_file:
                process_token_line_v3(labels, positions, token_line, tokens, truth_data)
            file_results_df = pd.DataFrame({'doc_name': doc_name, 'position': positions,
                                            'tokens': tokens, 'labels': labels})
            results_df, results_df_ner = attach_ner_prefix_to_labels(file_results_df, results_df,
                                                                     results_df_ner, doc_name)
    results_df_ner.to_json(os.path.join(output_dir, 'results_df_ner_v2.json'))
    results_df.to_json(os.path.join(output_dir, 'results_df_v2.json'))


def process_token_line_v3(labels, positions, token_line, tokens, truth_data):
    all_values = token_line.split(',')
    pos_vals = all_values[:8]
    token_val = ','.join(all_values[8:])
    eval_tokens = token_val.split(' ')
    min_xpos = pos_vals[0]
    max_xpos = pos_vals[2]
    xticks_per_char = math.ceil((int(max_xpos) - int(min_xpos)) / len(token_val))
    current_xpos = int(min_xpos)
    for token_idx, token in enumerate(eval_tokens, start=0):
        token = token.replace('\n', '').strip()
        if len(token) > 0:
            possible_labels = check_label(token, truth_data)
            if len(possible_labels) > 0:
                if token_idx + 1 < len(eval_tokens):
                    temp = token + ' ' + eval_tokens[token_idx + 1]
                    possible_labels = check_label(temp, truth_data)
                elif token_idx - 1 >= 0:
                    temp = eval_tokens[token_idx - 1] + ' ' + token
                    possible_labels = check_label(temp, truth_data)
                if len(possible_labels) > 0:
                    label = possible_labels[0]
            else:
                label = possible_labels[0]

            next_xpos = current_xpos + len(token) * xticks_per_char
            position = [
                current_xpos,
                int(pos_vals[1]),
                next_xpos,
                int(pos_vals[3]),
                next_xpos,
                int(pos_vals[5]),
                current_xpos,
                int(pos_vals[7])
            ]
            positions.append(position)
            current_xpos = next_xpos
            labels.append(label)
            tokens.append(token)


def get_doc_names(tokens_src):
    filenames = glob.glob(tokens_src + '*.txt')
    file_count = len(filenames)
    doc_names = [filename.split('/')[-1].split('.')[0].split(' ')[0] for filename in filenames]
    doc_names = set(doc_names)
    doc_name_count = len(doc_names)
    print('found ' + str(file_count - doc_name_count) + ' duplicates')
    return doc_names


def attach_ner_prefix_to_labels(file_results_df, results_df, results_df_ner, doc_name):
    for idx, row in file_results_df.iterrows():
        token_array = row.tokens.split(' ')
        row.tokens = token_array

        ner_tag = str(row.labels)[:]
        if idx > 0:
            previous_tag = ner_tag
        else:
            previous_tag = ''

        row.labels = [ner_tag for item in token_array]
        if ner_tag != 'O':
            if previous_tag != ner_tag:
                row.labels[0] = 'B-' + row.labels[0].split('-')[1]
            if idx < file_results_df.shape[0] - 1 and file_results_df.labels[idx + 1] != ner_tag:
                row.labels[-1] = 'L-' + row.labels[-1].split('-')[1]
    aggregated_tokens = [token for tokens in file_results_df.tokens for token in tokens]
    aggregated_labels = [label for labels in file_results_df.labels for label in labels]
    aggregated_file_dict = {
        'file_name': doc_name,
        'tokens': aggregated_tokens,
        'ner_tags': aggregated_labels
    }
    results_df = pd.concat([results_df, file_results_df], ignore_index=True)
    results_df_ner = results_df_ner.append(aggregated_file_dict, ignore_index=True)
    return results_df, results_df_ner


def determine_label_for_token(eval_token_val, truth_data):
    for key in truth_data.keys():
        eval_truth_str = ''.join(truth_data[key].split(' ')).strip()
        if len(eval_token_val) > 1 and eval_token_val in eval_truth_str or eval_truth_str in eval_token_val:
            if key == 'total':
                label = 'I-MONEY'
                break
            elif key == 'company':
                label = 'I-ORG'
                break
            elif key == 'date':
                label = 'I-DATE'
                break
            elif key == 'address':
                label = 'I-GPE'
                break
        else:
            label = 'O'
    return label


def check_label(token, truth_data):
    possible_labels = []
    found = False
    if len(token) > 1:
        if 'total' in truth_data.keys() and token in truth_data['total']:
            possible_labels.append('I-MONEY')
            found = True
        if 'company' in truth_data.keys() and token in truth_data['company']:
            possible_labels.append('I-ORG')
            found = True
        if 'date' in truth_data.keys() and token in truth_data['date']:
            possible_labels.append('I-DATE')
            found = True
        if 'address' in truth_data.keys() and token in truth_data['address']:
            possible_labels.append('I-GPE')
            found = True
    if found is False:
        possible_labels.append('O')
    return possible_labels



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing Script für SROIE Datensatz. Matching zwischen gegebenen'
                                                 'und Truth. Erstellt 2 JSON Dateien: Einmal für Graphen (results_df) '
                                                 'und einmal für LayoutLM (results_def_ner) '
                                                 'Token')
    parser.add_argument('--method', choices=['v1', 'v2', 'v3'],
                        help="Wahl aus v1, v2 und v3. v1 matched ganze zeilen zu einem Label und gibt jedem token "
                             "der Zeile die gleiche bounding box."
                             "v2 matched ganze Zeilen zu einem Label und teilt gegebene Boundingbox auf einzelne"
                             " Wörter der Zeile auf"
                             "v3 versucht, jedes Token einzeln einem Label zuzuordnen und gibt jedem Token seine"
                             " eigene Bounding Box"
                             "v2 funktioniert am Besten und ist Default", default='v2')
    parser.add_argument('--token_src', type=str, help='Pfad zu den extrahierten Token und deren Koordinaten.'
                                                      ' (task1train(626p))',
                        required=True)
    parser.add_argument('--ground_truth_src', type=str, help='Pfad zu der Grountruth für die dokumente. '
                                                             '(task2train(626p))', required=True)
    parser.add_argument('--output_dir', type=str, help="Speicherpfad für die fertigen Daten")
    args = parser.parse_args()

    if args.method == 'v1':
        match_token_to_label(args.token_src, args.ground_truth_scr, args.output_dir)
    elif args.method == 'v2':
        match_token_to_label_v2(args.token_src, args.ground_truth_scr, args.output_dir)
    elif args.method == 'v3':
        match_token_to_label_v3(args.token_src, args.ground_truth_scr, args.output_dir)
