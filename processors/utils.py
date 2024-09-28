
import csv
import os
import torch
import logging
import numpy as np
from PIL import Image
from bottom_up_attention.utils.extraction_bbox import get_boxes
import cv2
from tools.progressbar import ProgressBar

logger = logging.getLogger(__name__)


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """
        Gets an example from a dict with tensorflow tensors.

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """

        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """
        Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are. This method converts
        examples to the correct format.
        """
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

label_to_id = {'other': 0, 'neutral': 1, 'positive': 2, 'negative': 3, 'conflict': 4}
id_to_label = {0: 'other', 1: 'neutral', 2: 'positive', 3: 'negative', 4: 'conflict'}

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label_aspect=None, image_path=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label_aspect = label_aspect
        self.image_path = image_path


class MMBotoomSeqInputFeatures(object):
    """A single set of features of data for the ABSA task"""
    def __init__(self, input_ids, input_mask, subword_mask, img_feat, class_ids, box_dis_position, image_mask, segment_ids, label_ids, evaluate_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.subword_mask = subword_mask
        self.img_feat = img_feat
        self.class_ids = class_ids
        self.box_dis_position = box_dis_position
        self.image_mask = image_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # mapping between word index and head token index
        self.evaluate_label_ids = evaluate_label_ids



def image_process(image_path, transform):
    try:
        image = Image.open(image_path).convert("RGB")
    except(OSError, NameError):
        split_list = image_path.split('/')[:-1]
        head = ''
        for path in split_list:
            head = os.path.join(head, path)
        image_path = os.path.join(head, "17_06_4705.jpg")
        image = Image.open(image_path).convert("RGB")

    image = transform(image)
    return image


def convert_mm_examples_to_features_bert_bottom_vit_8(examples, label_list, tokenizer,
                                     cls_token_at_end=False, pad_on_left=False, cls_token='[CLS]',
                                     sep_token='[SEP]', pad_token=0, sequence_a_segment_id=0,
                                     sequence_b_segment_id=1, cls_token_segment_id=0, pad_token_segment_id=0,
                                     vit_extractor=None, vit_model=None, bottom_model=None,mask_padding_with_zero=True):

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    max_seq_length = -1
    examples_tokenized = []
    for (example_index, example) in enumerate(examples):
        tokens_a = []
        labels_a = []
        evaluate_label_ids = []
        words = example.text_a.split(' ')
        wid, tid = 0, 0
        for word, label in zip(words, example.label_aspect):
            subwords = tokenizer.tokenize(word)
            if subwords == []:
                subwords = ["[UNK]"]
            tokens_a.extend(subwords)
            if label == 'O':
                labels_a.extend(['O'] * len(subwords))
            else:
                cur_pos, cur_class = label.split('-')
                if cur_pos == 'B':
                    labels_a.extend([label] + ['I-'+str(cur_class)] * (len(subwords) - 1))
                else:
                    labels_a.extend([label] * len(subwords))
            evaluate_label_ids.append(tid)
            wid += 1
            # move the token pointer
            tid += len(subwords)
        #print(evaluate_label_ids)
        assert tid == len(tokens_a)
        evaluate_label_ids = np.array(evaluate_label_ids, dtype=np.int32)
        # if len(tokens_a) > max_seq_length:
        #     max_seq_length = len(tokens_a)
        # add [CLS] and [SEP]
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        labels = labels_a + ['O']
        if cls_token_at_end:
            # evaluate label ids not change
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
            labels = labels + ['O']
        else:
            # right shift 1 for evaluate label ids
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
            labels = ['O'] + labels
            evaluate_label_ids += 1


        ## Add image path
        image_path = example.image_path
        if not os.path.exists(image_path):
            print(image_path)

        examples_tokenized.append((tokens, segment_ids, labels, evaluate_label_ids, image_path))
        if len(tokens) > max_seq_length:
            max_seq_length = len(tokens)
            
    pbar = ProgressBar(n_total=len(examples_tokenized),desc = "Creating Feature")
    for ex_index, (tokens, segment_ids, labels, evaluate_label_ids, image_path) in enumerate(examples_tokenized):

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        
        subword_mask = [0] * len(input_ids)
        for i in evaluate_label_ids.tolist():
            subword_mask[i] = 1

        padding_length = max_seq_length - len(input_ids)
        label_ids = [label_map[label] for label in labels]

        # pad the input sequence and the mask sequence
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            subword_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + subword_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            # pad sequence tag 'O'
            label_ids = ([-100] * padding_length) + label_ids
            # right shift padding_length for evaluate_label_ids
            evaluate_label_ids += padding_length
        else:
            # evaluate ids not change
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            subword_mask = subword_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            # pad sequence tag 'O'
            label_ids = label_ids + ([-100] * padding_length)

        try:
            image = Image.open(image_path).convert("RGB")
        except(OSError, NameError):
            split_list = image_path.split('/')[:-1]
            head = ''
            # 重新拼接路径
            for path in split_list:
                head = os.path.join(head, path)
            image_path = os.path.join(head, "17_06_4705.jpg")
            image = Image.open(image_path).convert("RGB")
            
        im = cv2.imread(image_path)
        info = get_boxes(image_path, bottom_model)
        bbox_es = info["bbox"]
        ## 处理标签类别
        class_es = info["class"]
        class_text = " , ".join(class_es)
        class_token = tokenizer.tokenize(class_text)
        class_ids = tokenizer.convert_tokens_to_ids(class_token)
        padding_length = 100 - len(class_ids) #TODO: 100图片实体对应的文本标签的最大长度，写成参数传进来
        class_ids = class_ids + ([pad_token] * padding_length)
        
        #  position
        center_dis = info["distance"]
        box_dis_list = np.argsort(center_dis).tolist() # 按照box中心离左上角（0，0）举例排序
        padding_length = 8 - len(box_dis_list) #TODO: 8是图片中提取到的实体最大数量，写成参数传进来
        box_dis_index = box_dis_list + ([-100] * padding_length)
        
        ## 处理image mask
        image_mask = [1 if mask_padding_with_zero else 0] * len(box_dis_list)
        image_mask = image_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        
        img_feats = None
        for index in range(len(bbox_es)):
            bbox = bbox_es[index]
            object_image = im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
            object_image = cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB)
            vit_object_image = vit_extractor(images=object_image, return_tensors="pt")["pixel_values"][0]
            vit_object_image = vit_object_image.unsqueeze(0).to('cuda')
            vit_model.eval()
            with torch.no_grad():
                image_output = vit_model(vit_object_image)
            # torch.cuda.empty_cache()
            image_pooler = image_output.pooler_output
            if img_feats == None:
                img_feats = image_pooler
            else:
                img_feats = torch.cat((img_feats, image_pooler), dim=0)
        img_zero = image_pooler
        img_zero[:, :] = 0.0
        for index in range(padding_length):
            img_feats = torch.cat((img_feats, img_zero), dim=0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(subword_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("subword_mask: %s" % " ".join([str(x) for x in subword_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("labels: %s " % ' '.join([str(x) for x in label_ids]))
            logger.info("evaluate label ids: %s" % evaluate_label_ids)

        features.append(
            MMBotoomSeqInputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             subword_mask=subword_mask,
                             img_feat=img_feats,
                             class_ids=class_ids,
                             box_dis_position=box_dis_index,
                             image_mask=image_mask,
                             segment_ids=segment_ids,
                             label_ids=label_ids,
                             evaluate_label_ids=evaluate_label_ids))
        pbar(ex_index)
    print("maximal sequence length is", max_seq_length)
    return features


