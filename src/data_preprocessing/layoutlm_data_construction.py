import pathlib
import numpy as np
from tqdm import tqdm
import pandas as pd
import math
import json
import os
from PIL import Image
from transformers import AutoTokenizer



def format_data(train_test_split:float, per_word_src: str, per_line_src: str, doc_names_src: str, output_dir: str):
    train_path = os.path.join(output_dir, "testing_data/annotations")
    test_path = os.path.join(output_dir, "training_data/annotations")
    pathlib.Path(train_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(test_path).mkdir(parents=True, exist_ok=True)
    per_word_df = pd.read_json(per_word_src)
    per_line_df=pd.read_json(per_line_src)
    all_doc_names = np.load(doc_names_src, allow_pickle=True)
    train_doc_names = all_doc_names[math.floor(len(all_doc_names) * train_test_split):]
    test_doc_names = all_doc_names[0:math.floor(len(all_doc_names) * train_test_split)]
    for doc_list, save_path in zip([train_doc_names, test_doc_names], [train_path, test_path]):
        for doc_name in tqdm(doc_list):
            words_df = per_word_df[per_word_df.doc_name == doc_name]
            lines_df = per_line_df[per_line_df.doc_name == doc_name]
            words_start = 0
            form = []
            for id, (actual_bbox, tokens, labels) in enumerate(zip(lines_df.position, lines_df.tokens, lines_df.labels)):
                words_end = process_line(actual_bbox, form, id, labels, tokens, words_df, words_start)
                words_start = words_end
            with open(os.path.join(save_path, doc_name + '.json'), 'w') as f:
                json.dump({'form': form}, f)


def process_line(actual_bbox, form, id, labels, tokens, words_df, words_start):
    words_end = words_start + len(tokens)
    words = []
    if labels[0] == 'O':
        label = 'other'
    else:
        label = labels[0].split('-')[-1]
    for token_bbox, token in zip(words_df.position[words_start:words_end], words_df.tokens[words_start:words_end]):
        words.append({'box': [int(token_bbox[0]), int(token_bbox[1]), int(token_bbox[4]), int(token_bbox[5])],
                      'text': token[0]})
    form.append({
        'box': [int(actual_bbox[0]), int(actual_bbox[1]), int(actual_bbox[4]), int(actual_bbox[5])],
        'text': ' '.join(tokens),
        'label': label,
        'words': words,
        'linking': [],
        'id': id})
    return words_end


def get_bbox_string(box, width, length):
    return (
            str(int(1000 * (int(box[0]) / width)))
            + " "
            + str(int(1000 * (int(box[1]) / length)))
            + " "
            + str(int(1000 * (int(box[2]) / width)))
            + " "
            + str(int(1000 * (int(box[3]) / length)))
    )


def get_actual_bbox_string(box, width, length):
    return (
            str(box[0])
            + " "
            + str(box[1])
            + " "
            + str(box[2])
            + " "
            + str(box[3])
            + "\t"
            + str(width)
            + " "
            + str(length)
    )


def convert(image_dir, data_dir, data_split, output_dir, num_classes):
    with open(
            os.path.join(output_dir, data_split + ".txt.tmp"),
            "w",
            encoding="utf8",
    ) as fw, open(
        os.path.join(output_dir, data_split + "_box.txt.tmp"),
        "w",
        encoding="utf8",
    ) as fbw, open(
        os.path.join(output_dir, data_split + "_image.txt.tmp"),
        "w",
        encoding="utf8",
    ) as fiw:
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(image_dir, file)
            image_path = image_path.replace("json", "jpg")
            file_name = os.path.basename(image_path)
            image = Image.open(image_path)
            width, length = image.size
            for item in data["form"]:
                convert_item(fbw, file_name, fiw, fw, item, length, num_classes, width)

            fw.write("\n")
            fbw.write("\n")
            fiw.write("\n")


def convert_item(fbw, file_name, fiw, fw, item, length, num_classes, width):
    words, label = item["words"], item["label"]
    words = [w for w in words if w["text"].strip() != ""]
    if len(words) != 0:
        if label == "other":
            label_with_prefix = 'O'
            for w in words:
                write_word_data_to_files(fbw, file_name, fiw, fw, label_with_prefix, length, w, width)
        else:
            if int(num_classes) == 5:
                label_with_prefix = 'I-' + label.upper()
                for w in words:
                    write_word_data_to_files(fbw, file_name, fiw, fw, label_with_prefix, length, w, width)

            elif int(num_classes) == 9:
                label_with_prefix = 'B-' + label.upper()
                w = words[0]
                write_word_data_to_files(fbw, file_name, fiw, fw, label_with_prefix, length, w, width)
                label_with_prefix = 'I-' + label.upper()
                for w in words[1:]:
                    write_word_data_to_files(fbw, file_name, fiw, fw, label_with_prefix, length, w, width)
            elif int(num_classes) == 13:
                label_with_prefix = 'B-' + label.upper()
                w = words[0]
                write_word_data_to_files(fbw, file_name, fiw, fw, label_with_prefix, length, w, width)

                if len(words) > 1:
                    label_with_prefix = 'I-' + label.upper()
                    for w in words[1:-1]:
                        write_word_data_to_files(fbw, file_name, fiw, fw, label_with_prefix, length, w, width)
                    label_with_prefix = 'L-' + label.upper()
                    w = words[-1]
                    write_word_data_to_files(fbw, file_name, fiw, fw, label_with_prefix, length, w, width)


def write_word_data_to_files(fbw, file_name, fiw, fw, l, length, w, width):
    fw.write(w["text"] + f"\t{l}\n")
    fbw.write(
        w["text"]
        + "\t"
        + get_bbox_string(w["box"], width, length)
        + "\n"
    )
    fiw.write(
        w["text"]
        + "\t"
        + get_actual_bbox_string(w["box"], width, length)
        + "\t"
        + file_name
        + "\n"
    )


def seg_file(file_path, tokenizer, max_len):
    subword_len_counter = 0
    output_path = file_path[:-4]
    print(output_path)
    with open(file_path, "r", encoding="utf8") as f_p, open(
            output_path, "w", encoding="utf8"
    ) as fw_p:
        for line in f_p:
            line = line.rstrip()

            if not line:
                fw_p.write(line + "\n")
                subword_len_counter = 0
                continue
            token = line.split("\t")[0]

            current_subwords_len = len(tokenizer.tokenize(token))

            # Token contains strange control characters like \x96 or \x95
            # Just filter out the complete line
            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                fw_p.write("\n" + line + "\n")
                subword_len_counter = current_subwords_len
                continue

            subword_len_counter += current_subwords_len

            fw_p.write(line + "\n")


def seg(data_split, output_dir, model_name_or_path, max_len):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, do_lower_case=True
    )
    seg_file(
        os.path.join(output_dir, data_split + ".txt.tmp"),
        tokenizer,
        max_len,
    )
    seg_file(
        os.path.join(output_dir, data_split + "_box.txt.tmp"),
        tokenizer,
        max_len,
    )
    seg_file(
        os.path.join(output_dir, data_split + "_image.txt.tmp"),
        tokenizer,
        max_len,
    )


def preprocess(image_dir, data_dir, data_split, output_dir, model_name_or_path, max_len, num_classes):
    convert(image_dir, data_dir, data_split, output_dir, num_classes)
    seg( data_split, output_dir, model_name_or_path, max_len)


def format_and_preprocess(image_dir,
                          train_test_split=0.70,
                          max_len=510,
                          num_classes=5,
                          per_word_src=os.path.join(os.path.abspath(__file__), '../../data/SROIE/results_df_v2.json'),
                          per_line_src=os.path.join(os.path.abspath(__file__), '../../data/SROIE/results_df.json'),
                          doc_names_src=os.path.join(os.path.abspath(__file__), '../../data/SROIE/doc_names.npy'),
                          output_dir=os.path.join(os.path.abspath(__file__), '../../data/SROIE/LayoutLM_SROIE_data/5_classes')):

    format_data(train_test_split=train_test_split,
                per_word_src=per_word_src,
                per_line_src=per_line_src,
                doc_names_src=doc_names_src,
                output_dir=output_dir)

    preprocess(image_dir=image_dir,
               data_dir=os.path.join(output_dir, 'training_data/annotations'),
               data_split='train',
               model_name_or_path='microsoft/layoutlm-base-uncased',
               max_len=max_len,
               num_classes=num_classes)

    preprocess(image_dir=image_dir,
               data_dir=os.path.join(output_dir, 'testing_data/annotations'),
               data_split='test',
               model_name_or_path='microsoft/layoutlm-base-uncased',
               max_len=max_len,
               num_classes=num_classes)