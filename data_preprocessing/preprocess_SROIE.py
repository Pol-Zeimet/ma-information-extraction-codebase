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
    filenames = glob.glob(tokens_src + '*.txt')
    file_count = len(filenames)
    slugs = [filename.split('/')[-1].split('.')[0].split(' ')[0] for filename in filenames]
    slugs = set(slugs)
    slug_count = len(slugs)

    print('found ' + str(file_count - slug_count) + ' duplicates')
    for slug in tqdm(slugs):
        with  open(tokens_src + slug + '.txt', 'r') as token_file, open(ground_truth_src + slug + '.txt',
                                                                        'br') as truth:
            truth_data = json.loads(truth.read())
            labels = []
            tokens = []
            positions = []

            for token_line in token_file:
                all_values = token_line.split(',')
                pos_vals = all_values[:8]
                token_val = ','.join(all_values[8:])
                eval_token_val = ''.join(token_val.split(' ')).strip()

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

                labels.append(label)
                tokens.append(token_val.replace('\n', ''))
                positions.append(pos_vals)
            file_results_df = pd.DataFrame({'slug': slug, 'position': positions, 'tokens': tokens, 'labels': labels})

            for idx, row in file_results_df.iterrows():
                token_array = row.tokens.split(' ')
                row.tokens = token_array

                if idx > 0:
                    previous_tag = ner_tag
                else:
                    previous_tag = ''

                ner_tag = str(row.labels)[:]
                row.labels = [ner_tag for item in token_array]
                if ner_tag != 'O':
                    if previous_tag != ner_tag:
                        row.labels[0] = 'B-' + row.labels[0].split('-')[1]
                    if idx < file_results_df.shape[0] - 1 and file_results_df.labels[idx + 1] != ner_tag:
                        row.labels[-1] = 'L-' + row.labels[-1].split('-')[1]

            aggregated_tokens = [token for tokens in file_results_df.tokens for token in tokens]
            aggregated_labels = [label for labels in file_results_df.labels for label in labels]

            aggregated_file_dict = {
                'file_name': slug,
                'tokens': aggregated_tokens,
                'ner_tags': aggregated_labels
            }
            results_df = pd.concat([results_df, file_results_df], ignore_index=True)
            results_df_ner = results_df_ner.append(aggregated_file_dict, ignore_index=True)

    results_df_ner.to_json(os.path.join(output_dir, 'results_df_ner.json'))
    results_df.to_json(os.path.join(output_dir, 'results_df.json'))


def match_token_to_label_v2(tokens_src, ground_truth_src, output_dir):
    results_df_ner = pd.DataFrame()
    results_df = pd.DataFrame()
    filenames = glob.glob(tokens_src + '*.txt')
    file_count = len(filenames)
    slugs = [filename.split('/')[-1].split('.')[0].split(' ')[0] for filename in filenames]
    slugs = set(slugs)
    slug_count = len(slugs)

    print('found ' + str(file_count - slug_count) + ' duplicates')
    for slug in tqdm(slugs):
        with  open(tokens_src + slug + '.txt', 'r') as token_file, open(ground_truth_src + slug + '.txt',
                                                                        'br') as truth:
            truth_data = json.loads(truth.read())
            labels = []
            tokens = []
            positions = []

            for token_line in token_file:
                all_values = token_line.split(',')

                pos_vals = all_values[:8]
                min_xpos = pos_vals[0]
                max_xpos = pos_vals[2]
                token_val = ','.join(all_values[8:])
                xticks_per_char = math.ceil((int(max_xpos) - int(min_xpos)) / len(token_val))
                current_xpos = int(min_xpos)
                token_array = token_val.split(' ')
                eval_token_val = ''.join(token_array).strip()

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

                for token in token_array:
                    if len(token) > 0:
                        labels.append(label)
                        tokens.append(token.replace('\n', ''))
                        next_xpos = current_xpos + len(token) * xticks_per_char
                        position = [
                            current_xpos,
                            pos_vals[1],
                            next_xpos,
                            pos_vals[3],
                            next_xpos,
                            pos_vals[5],
                            current_xpos,
                            pos_vals[7]
                        ]
                        positions.append(position)
                        # compensating for whitespaces between words
                        current_xpos = next_xpos + xticks_per_char
            file_results_df = pd.DataFrame({'slug': slug, 'position': positions, 'tokens': tokens, 'labels': labels})

            for idx, row in file_results_df.iterrows():
                token_array = row.tokens.split(' ')
                row.tokens = token_array

                if idx > 0:
                    previous_tag = ner_tag
                else:
                    previous_tag = ''

                ner_tag = str(row.labels)[:]
                row.labels = [ner_tag for item in token_array]
                if ner_tag != 'O':
                    if previous_tag != ner_tag:
                        row.labels[0] = 'B-' + row.labels[0].split('-')[1]
                    if idx < file_results_df.shape[0] - 1 and file_results_df.labels[idx + 1] != ner_tag:
                        row.labels[-1] = 'L-' + row.labels[-1].split('-')[1]

            aggregated_tokens = [token for tokens in file_results_df.tokens for token in tokens]
            aggregated_labels = [label for labels in file_results_df.labels for label in labels]

            aggregated_file_dict = {
                'file_name': slug,
                'tokens': aggregated_tokens,
                'ner_tags': aggregated_labels
            }
            results_df = pd.concat([results_df, file_results_df], ignore_index=True)
            results_df_ner = results_df_ner.append(aggregated_file_dict, ignore_index=True)
    results_df_ner.to_json(os.path.join(output_dir, 'results_df_ner_v2.json'))
    results_df.to_json(os.path.join(output_dir, 'results_df_v2.json'))


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


def match_token_to_label_v3(tokens_src, ground_truth_src, output_dir):
    results_df_ner = pd.DataFrame()
    results_df = pd.DataFrame()

    filenames = glob.glob(tokens_src + '*.txt')
    file_count = len(filenames)
    slugs = [filename.split('/')[-1].split('.')[0].split(' ')[0] for filename in filenames]
    slugs = set(slugs)
    slug_count = len(slugs)

    print('found ' + str(file_count - slug_count) + ' duplicates')
    for slug in tqdm(slugs):
        with  open(tokens_src + slug + '.txt', 'r') as token_file, open(ground_truth_src + slug + '.txt',
                                                                        'br') as truth:
            truth_data = json.loads(truth.read())
            labels = []
            tokens = []
            positions = []

            for token_line in token_file:
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
                            pos_vals[1],
                            next_xpos,
                            pos_vals[3],
                            next_xpos,
                            pos_vals[5],
                            current_xpos,
                            pos_vals[7]
                        ]
                        positions.append(position)
                        current_xpos = next_xpos

                        labels.append(label)
                        tokens.append(token)

            file_results_df = pd.DataFrame({'slug': slug, 'position': positions, 'tokens': tokens, 'labels': labels})

            for idx, row in file_results_df.iterrows():
                token_array = row.tokens.split(' ')
                row.tokens = token_array

                if idx > 0:
                    previous_tag = ner_tag
                else:
                    previous_tag = ''

                ner_tag = str(row.labels)[:]
                row.labels = [ner_tag for item in token_array]
                if ner_tag != 'O':
                    if previous_tag != ner_tag:
                        row.labels[0] = 'B-' + row.labels[0].split('-')[1]
                    if idx < file_results_df.shape[0] - 1 and file_results_df.labels[idx + 1] != ner_tag:
                        row.labels[-1] = 'L-' + row.labels[-1].split('-')[1]

            aggregated_tokens = [token for tokens in file_results_df.tokens for token in tokens]
            aggregated_labels = [label for labels in file_results_df.labels for label in labels]

            aggregated_file_dict = {
                'file_name': slug,
                'tokens': aggregated_tokens,
                'ner_tags': aggregated_labels
            }
            results_df = pd.concat([results_df, file_results_df], ignore_index=True)
            results_df_ner = results_df_ner.append(aggregated_file_dict, ignore_index=True)
    results_df_ner.to_json(os.path.join(output_dir, 'results_df_ner_v2.json'))
    results_df.to_json(os.path.join(output_dir, 'results_df_v2.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing Script für SROIE Datensatz. MAtching zwischen gegebenen'
                                                 'und Truth. '
                                                 'Token')
    parser.add_argument('--method', choices=['v1', 'v2', 'v3'],
                        help="Wahl aus v1, v2 und v3. v1 matched ganze zeilen zu einem Label und gibt jedem token der Zeile die gleiche bounding box."
                             "v2 matched ganze Zeilen zu einem Label und teilt gegebene Boundingbox auf einzelne Wörter der Zeile auf"
                             "v3 versucht, jedes Token einzeln einem Label zuzuordnen und gibt jedem Token seine eigene Bounding Box"
                             "v2 funktioniert am Besten und ist Default", default='v2')
    parser.add_argument('--token_src', type=str, help='Pfad zu den extrahierten Token und deren Koordinaten. (task1train(626p))',
                        required=True)
    parser.add_argument('--ground_truth_src', type=str, help='Pfad zu der Grountruth für die dokumente. (task2train(626p))', required=True)
    parser.add_argument('--output_dir', type=str, help="Speicherpfad für die fertigen Daten")
    args = parser.parse_args()

    if args.method == 'v1':
        match_token_to_label(args.token_src, args.ground_truth_scr, args.output_dir)
    elif args.method == 'v2':
        match_token_to_label_v2(args.token_src, args.ground_truth_scr, args.output_dir)
    elif args.method == 'v3':
        match_token_to_label_v3(args.token_src, args.ground_truth_scr, args.output_dir)
