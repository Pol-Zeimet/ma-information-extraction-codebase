import numpy as np
from tensorflow import keras
import torch
import pytesseract
from PIL import ImageDraw, ImageFont
import os
from src.utils.utils import find_max_positions, normalize_box_for_graphs, convert_to_padded_graph, normalize_box, \
    convert_example_to_features, iob_to_label, Embedder

try:
    from PIL import Image
except ImportError:
    import Image

from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
from transformers import LayoutLMTokenizer

from tensorflow import argmax as tf_argmax, one_hot as tf_one_hot


class Predictor:
    def __init__(self, model_name, model_path, output_dir, embeddings=None, pad_to_nodes=None, pad_to_edges=None,
                 embedding_size=None):
        self.model_name = model_name
        self.model_path = model_path
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        self.pad_to_nodes = pad_to_nodes
        self.pad_to_edges = pad_to_edges
        self.embedding_size = embedding_size
        self.args = None
        self.embedding_model = None
        self.embeddings = embeddings
        self.label_map = {0: 'I-DATE', 1: 'I-GPE', 2: 'I-MONEY', 3: 'I-ORG', 4: 'O'}
        self.label2width = {'other': 1, 'org': 2, 'gpe': 2, 'money': 2, 'date': 2}
        self.label2color = {'other': '#b97a56', 'org': '#ff0000', 'gpe': '#0800ff', 'money': 'green', 'date': '#ff9500'}
        self.label2text = {'other': 'O', 'org': 'ORG', 'gpe': 'ADDR', 'money': 'TOTAL', 'date': 'DATE'}
        self.embedder: Embedder = None

    def create(self):
        if self.model_name == 'bert':
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)

        elif self.model_name == 'graph':
            self.embedder = Embedder(self.embeddings, "/content/drive/MyDrive/Master_Arbeit/colab/SROIE_Baseline")

            if 'CRF' in self.model_path:
                raise NotImplemented
            else:
                self.model = keras.models.load_model(self.model_path)

        elif self.model_name == 'layoutlm':
            args = {'local_rank': -1,
                    'overwrite_cache': True,
                    'max_seq_length': 512,
                    'model_type': 'layoutlm', }

            class AttrDict(dict):
                def __init__(self, *args, **kwargs):
                    super(AttrDict, self).__init__(*args, **kwargs)
                    self.__dict__ = self

            self.args = AttrDict(args)
            self.tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
            self.model = torch.load(self.model_path)
            self.model.eval()

    def predict_image(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB")

            actual_boxes, words = self._extract_token_and_bb_from_image(image)

            word_level_predictions = []
            final_boxes = []

            if self.model_name == 'bert':
                tokens = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(words,
                                                                                             padding="max_length",
                                                                                             truncation=True,
                                                                                             is_split_into_words=True)))
                tokenized_input = self.tokenizer(words,
                                                 padding="max_length",
                                                 truncation=True,
                                                 is_split_into_words=True,
                                                 return_offsets_mapping=True)

                offset_mapping = tokenized_input.pop("offset_mapping")
                input_ids = self.tokenizer.encode(words,
                                                  padding="max_length",
                                                  truncation=True,
                                                  is_split_into_words=True,
                                                  return_tensors="pt")
                box_index = 0
                offset_boxes = []
                for offset in offset_mapping:
                    current_box = actual_boxes[box_index]
                    if offset[0] == 0 and offset[1] != 0:
                        box_index += 1
                        offset_boxes.append(current_box)
                    elif offset[0] == 0 and offset[1] == 0:
                        continue
                    else:
                        offset_boxes.append(current_box)

                outputs = self.model(input_ids)[0]
                predictions = torch.argmax(outputs, dim=2).numpy()
                for token, token_pred, box in zip(tokens, predictions[0], offset_boxes):
                    if (token.startswith("##")) or (
                            self.tokenizer.encode([token], is_split_into_words=True) in [self.tokenizer.cls_token_id,
                                                                                         self.tokenizer.sep_token_id,
                                                                                         self.tokenizer.pad_token_id]):
                        continue
                    else:
                        word_level_predictions.append(self.label_map[token_pred])
                        final_boxes.append(box)

            elif self.model_name == 'graph':
                normalized_boxes = []
                max_x_pos, max_y_pos = find_max_positions(actual_boxes)
                for box in actual_boxes:
                    normalized_boxes.append(normalize_box_for_graphs(box, max_x_pos, max_y_pos))

                labels = ['O', 'I-MONEY', 'I-ORG', 'I-DATE', 'I-GPE']
                x = convert_to_padded_graph(words, normalized_boxes, self.embedder, self.pad_to_nodes, self.pad_to_edges,
                                            self.embedding_size)
                assert len(normalized_boxes) == len(words)
                pred = self.model.predict(x)
                predictions = tf_one_hot(tf_argmax(pred[0], axis=2), depth=5)
                predictions = [np.where(labels == 1)[0][0] for prediction in predictions for labels in prediction]
                predictions = np.reshape(predictions, (1, self.pad_to_nodes))
                word_level_predictions = [labels[pred_val] for pred_val in predictions[0][:len(words)]]
                print(word_level_predictions)
                final_boxes = actual_boxes

            elif self.model_name == 'layoutlm':
                boxes = []
                width, height = image.size
                for box in actual_boxes:
                    boxes.append(normalize_box(box, width, height))

                input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes = convert_example_to_features(
                    image=image,
                    words=words,
                    boxes=boxes,
                    actual_boxes=actual_boxes,
                    tokenizer=self.tokenizer,
                    args=self.args)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
                attention_mask = torch.tensor(input_mask, device=device).unsqueeze(0)
                token_type_ids = torch.tensor(segment_ids, device=device).unsqueeze(0)
                bbox = torch.tensor(token_boxes, device=device).unsqueeze(0)

                outputs = self.model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                                     token_type_ids=token_type_ids)
                token_predictions = outputs.logits.argmax(-1).squeeze().tolist()
                for id, token_pred, box in zip(input_ids.squeeze().tolist(), token_predictions, token_actual_boxes):
                    if (self.tokenizer.decode([id]).startswith("##")) or (id in [self.tokenizer.cls_token_id,
                                                                                 self.tokenizer.sep_token_id,
                                                                                 self.tokenizer.pad_token_id]):
                        continue
                    else:
                        word_level_predictions.append(self.label_map[token_pred])
                        final_boxes.append(box)

            image = self._draw_results(image, word_level_predictions, final_boxes)
            if self.output_dir != "":
                image.save(os.path.join(self.output_dir, self.model_path.split('/')[-1]))

    @staticmethod
    def _extract_token_and_bb_from_image(image):
        ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
        width, height = image.size
        w_scale = 1000 / width
        h_scale = 1000 / height
        ocr_df.dropna().assign(left_scaled=ocr_df.left * w_scale,
                               width_scaled=ocr_df.width * w_scale,
                               top_scaled=ocr_df.top * h_scale,
                               height_scaled=ocr_df.height * h_scale,
                               right_scaled=lambda x: x.left_scaled + x.width_scaled,
                               bottom_scaled=lambda x: x.top_scaled + x.height_scaled)
        float_cols = ocr_df.select_dtypes('float').columns
        ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
        ocr_df = ocr_df.dropna().reset_index(drop=True)
        words = list(ocr_df.text)

        coordinates = ocr_df[['left', 'top', 'width', 'height']]
        actual_boxes = []

        for idx, row in coordinates.iterrows():
            x, y, w, h = tuple(row)  # the row comes in (left, top, width, height) format
            actual_box = [x, y, x + w,
                          y + h]  # we turn it into (left, top, left+width, top+height) to get the actual box
            actual_boxes.append(actual_box)

        return actual_boxes, words

    def _draw_results(self, image, predictions, boxes):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for key in self.label2color.keys():
            for prediction, box in zip(predictions, boxes):
                predicted_label = iob_to_label(prediction).lower()
                if predicted_label == key:
                    draw.rectangle(box, outline=self.label2color[predicted_label],
                                   width=self.label2width[predicted_label])
                    draw.text((box[0] + 10, box[1] - 10), text=self.label2text[predicted_label],
                              fill=self.label2color[predicted_label], font=font)

        return image
