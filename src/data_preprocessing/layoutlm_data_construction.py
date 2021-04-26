import pathlib
import numpy as np
from tqdm import tqdm
import pandas as pd
import math
import json
import os
from PIL import Image
from transformers import AutoTokenizer
from typing import List


def format_data(train_test_split: float, per_word_src: str, per_line_src: str, doc_names_src: str, output_dir: str):
    """
    creates Train-Test Split on document names and creates a json file for each document. The json contains a form object
    with an entry for each line in the document in the given output directory.

    Structure of generatesd JSON:
        Each line object looks as follows:
        {
            'box': Bounding Box for the line
            'text': Text for the entire line,
            'label': label for the entire line,
            'words': List ob objects for every word in line
            'linking': interlinking of lines for semantic linking (not used here),
            'id': id of the line
        }
        each word object has the following structure:

        {
            'box': bounding box for word,
            'text': string value for word
        }

    Args:
        :param train_test_split: float to indicate percentage of training data
        :param per_word_src: src of dataframe containin labeling and bbox for individual words (using method v2 from preprocessing)
        :param per_line_src: src of dataframe containing labeling and bbox for entire line ( using method v2 from preprocessing)
        :param doc_names_src: src of ndarray containing all document names
        :param output_dir: output directory. Will be filled with folders for training and testing data
    """
    train_path = os.path.join(output_dir, "testing_data/annotations")
    test_path = os.path.join(output_dir, "training_data/annotations")
    pathlib.Path(train_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(test_path).mkdir(parents=True, exist_ok=True)
    per_word_df = pd.read_json(per_word_src)
    per_line_df=pd.read_json(per_line_src)
    all_doc_names = np.load(doc_names_src, allow_pickle=True)
    np.random.shuffle(all_doc_names)
    train_doc_names = all_doc_names[math.floor(len(all_doc_names) * train_test_split):]
    test_doc_names = all_doc_names[0:math.floor(len(all_doc_names) * train_test_split)]
    for doc_list, save_path in zip([train_doc_names, test_doc_names], [train_path, test_path]):
        for doc_name in tqdm(doc_list):
            words_df = per_word_df[per_word_df.doc_name == doc_name]
            lines_df = per_line_df[per_line_df.doc_name == doc_name]
            word_index = 0
            form = []
            for id, (actual_bbox, tokens, labels) in enumerate(zip(lines_df.position, lines_df.tokens, lines_df.labels)):
                word_index = process_line(actual_bbox, form, id, labels, tokens, words_df, word_index)
            with open(os.path.join(save_path, doc_name + '.json'), 'w') as f:
                json.dump({'form': form}, f)


def process_line(actual_bbox: List, form: List, id: int, labels: List,
                 tokens: List, words_df: pd.DataFrame, word_index: int) -> int:
    """
    Processes a line from the OCR data of the document and appends it to the given form List.
    Structure of item in Form List:
        Each line object looks as follows:
        {
            'box': Bounding Box for the line
            'text': Text for the entire line,
            'label': label for the entire line,
            'words': List ob objects for every word in line
            'linking': interlinking of lines for semantic linking (not used here),
            'id': id of the line
        }
        each word object has the following structure:

        {
            'box': bounding box for word,
            'text': string value for word
        }

    Args:
        :param actual_bbox: BBox of line
        :param form: form List to append to
        :param id: id of the line
        :param labels: list of labels for each word of the line
        :param tokens: list of tokens in the line
        :param words_df: dataframe containing labeling and bbox for each word
        :param word_index: current word index inside the document
    Return Value:
        :return: word_index_end: new word index after processing line
    """
    word_index_end = word_index + len(tokens)
    words = []
    if labels[0] == 'O':
        label = 'other'
    else:
        label = labels[0].split('-')[-1]
    for token_bbox, token in zip(words_df.position[word_index:word_index_end], words_df.tokens[word_index:word_index_end]):
        words.append({'box': [int(token_bbox[0]), int(token_bbox[1]), int(token_bbox[4]), int(token_bbox[5])],
                      'text': token[0]})
    form.append({
        'box': [int(actual_bbox[0]), int(actual_bbox[1]), int(actual_bbox[4]), int(actual_bbox[5])],
        'text': ' '.join(tokens),
        'label': label,
        'words': words,
        'linking': [],
        'id': id})
    return word_index_end


def get_bbox_string(box: List, width: int, height: int) -> str:
    """
    return bbox as string

    :param box: box as List like [x1,y1,x2,y2]
    :param width: width of document image
    :param height: height of document images
    """
    return (
            str(int(1000 * (int(box[0]) / width)))
            + " "
            + str(int(1000 * (int(box[1]) / height)))
            + " "
            + str(int(1000 * (int(box[2]) / width)))
            + " "
            + str(int(1000 * (int(box[3]) / height)))
    )


def get_actual_bbox_string(box, width, height) -> str:
    """
        return actual bbox as string together with image width and height

        :param box: box as List like [x1,y1,x2,y2]
        :param width: width of document image
        :param height: height of document images
        """
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
            + str(height)
    )


def convert(image_dir, data_dir, data_split, output_dir, num_classes):
    """
    convert json files into txt files containing label and bbox for every token

    :param image_dir: path to directory of receipt scans from dataset
    :param data_dir: path to form json files for documents
    :param data_split: train or test
    :param output_dir: output directory for txt files
    :param num_classes: number of classes. 5, 9 or 13
    """
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


def convert_item(fbw, file_name: str, fiw, fw, item: dict, height: int, num_classes: int, width: int):
    """
    converts single json file and appends content to given txt files.

    :param fbw:  boxes file object to append to
    :param file_name: name of document
    :param fiw: image file object to append to
    :param fw: label file object to append to
    :param item: element in form json to convert
    :param height: height of the image
    :param num_classes: number of classes
    :param width: width of the image
    :return:
    """
    words, label = item["words"], item["label"]
    words = [w for w in words if w["text"].strip() != ""]
    if len(words) != 0:
        if label == "other":
            label_with_prefix = 'O'
            for w in words:
                write_word_data_to_files(fbw, file_name, fiw, fw, label_with_prefix, height, w, width)
        else:
            if int(num_classes) == 5:
                label_with_prefix = 'I-' + label.upper()
                for w in words:
                    write_word_data_to_files(fbw, file_name, fiw, fw, label_with_prefix, height, w, width)

            elif int(num_classes) == 9:
                label_with_prefix = 'B-' + label.upper()
                w = words[0]
                write_word_data_to_files(fbw, file_name, fiw, fw, label_with_prefix, height, w, width)
                label_with_prefix = 'I-' + label.upper()
                for w in words[1:]:
                    write_word_data_to_files(fbw, file_name, fiw, fw, label_with_prefix, height, w, width)
            elif int(num_classes) == 13:
                label_with_prefix = 'B-' + label.upper()
                w = words[0]
                write_word_data_to_files(fbw, file_name, fiw, fw, label_with_prefix, height, w, width)

                if len(words) > 1:
                    label_with_prefix = 'I-' + label.upper()
                    for w in words[1:-1]:
                        write_word_data_to_files(fbw, file_name, fiw, fw, label_with_prefix, height, w, width)
                    label_with_prefix = 'L-' + label.upper()
                    w = words[-1]
                    write_word_data_to_files(fbw, file_name, fiw, fw, label_with_prefix, height, w, width)


def write_word_data_to_files(fbw, file_name:str, fiw, fw, l:str, height:int, w:str, width:int):
    """
    write data of given word to files.

    :param fbw: boxes file object to append to
    :param file_name: document name
    :param fiw: image file object to append to
    :param fw: label file object to append to
    :param l: label for word
    :param height: height of image
    :param w: word
    :param width: width of image
    :return:
    """
    fw.write(w["text"] + f"\t{l}\n")
    fbw.write(
        w["text"]
        + "\t"
        + get_bbox_string(w["box"], width, height)
        + "\n"
    )
    fiw.write(
        w["text"]
        + "\t"
        + get_actual_bbox_string(w["box"], width, height)
        + "\t"
        + file_name
        + "\n"
    )


def seg_file(file_path, tokenizer, max_len):
    """
    seqment temporary file. Check every Token if it can be subtokenized
    Check if max length will be crossed

    Args:
        :param file_path: path of file to segment
        :param tokenizer: tokenizer instance
        :param max_len: max length for sequence
        :return:
    """
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
    """
    segment the txt.tmp files of current dataset
    Args:
        :param data_split: path to tmp files
        :param output_dir: output directory for segmented files
        :param model_name_or_path: path to Tokenizer for model
        :param max_len: max sequence length

    """
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


def preprocess(image_dir: str, data_dir: str, data_split: str, output_dir: str, model_name_or_path: str,
               max_len: int, num_classes: int):
    """
    Process json files to txt files and segment them
    :param image_dir: path to scanned documents of dataset
    :param data_dir: path to form json files
    :param data_split: train or test
    :param output_dir: output dir for txt files
    :param model_name_or_path: name of model for tokenizer
    :param max_len: max sequence length
    :param num_classes: number of classes, 5,9 or 13

    """
    convert(image_dir, data_dir, data_split, output_dir, num_classes)
    seg(data_split, output_dir, model_name_or_path, max_len)


def create_label_file(path: str):
    """Generate txt file with labels from train.txt containing for dataset.
        :param path: path to the train.txt file
    """
    with open(os.path.join(path, 'train.txt'), 'r', encoding="utf8") as f:
        labels = set([line.split('\t')[-1].strip() for line in f if line.split('\t')[-1].strip() != ''])
        labels = list(labels)
        labels.sort()
    with open(os.path.join(path, 'labels.txt'), 'w', encoding="utf8") as f:
        for label in labels:
            f.write(label + '\n')


def format_and_preprocess(image_dir: str,
                          train_test_split: float = 0.70,
                          max_len: int = 510,
                          num_classes: int = 5,
                          per_word_src: str = os.path.join(os.path.abspath(__file__), '../../data/SROIE/results_df_v2.json'),
                          per_line_src: str = os.path.join(os.path.abspath(__file__), '../../data/SROIE/results_df.json'),
                          doc_names_src: str = os.path.join(os.path.abspath(__file__), '../../data/SROIE/doc_names.npy'),
                          output_dir: str = os.path.join(os.path.abspath(__file__), '../../data/SROIE/LayoutLM_SROIE_data/5_classes')):
    """
    Preprocess dataset
        Bring data into form json format, convert them to txt files and segment these to use as dataset for LayoutLM

        Args:
            :param image_dir: path to scanned documents from dataset
            :param train_test_split: float >0  snd <1 indicating the percentage of data to use for training
            :param max_len: max sequence length
            :param num_classes: number of classes 5,9 or 13
            :param per_word_src: path to dataframe containing bbox and labels per word
            (results_df using v2 or v3 of preprocessing)
            :param per_line_src: path to dataframe containing bbox and labels per line
            (results_df using v1 of preprocessing)
            :param doc_names_src: path to ndarray containing the names of all documents in the dataset
            :param output_dir: output directory to save the files to
    :return:
    """

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

    create_label_file(output_dir)