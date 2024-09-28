import os
import logging
import torch
from .utils import DataProcessor

logger = logging.getLogger(__name__)

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label_aspect=None, image_path=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label_aspect = label_aspect
        self.image_path = image_path
        

class EntityImageProcessor(DataProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, text_data_dir, image_data_dir, tagging_schema):
        """Gets a collection of :class:`InputExample` for the train set."""
        return self._create_examples(text_path=os.path.join(text_data_dir, "train.txt"), 
                                    image_path=image_data_dir, set_type="train", tagging_schema=tagging_schema)

    def get_dev_examples(self, text_data_dir, image_data_dir, tagging_schema):
        """Gets a collection of :class:`InputExample` for the dev set."""
        return self._create_examples(text_path=os.path.join(text_data_dir, "dev.txt"), 
                                    image_path=image_data_dir, set_type="dev", tagging_schema=tagging_schema)

    def get_test_examples(self, text_data_dir, image_data_dir, tagging_schema):
        """Gets a collection of :class:`InputExample` for the test set."""
        return self._create_examples(text_path=os.path.join(text_data_dir, "test.txt"), 
                                    image_path=image_data_dir, set_type="test", tagging_schema=tagging_schema)

    def get_labels(self, tagging_schema):
        """Gets the list of labels for this data set."""
        # tag sequence for opinion target extraction
        if tagging_schema == 'MATE':
            return ['O', 'B-ASPECT', 'I-ASPECT']
        # tag sequence for targeted sentiment
        elif tagging_schema == 'TS':
            return ['O', 'EQ', 'B-POS', 'I-POS', 'B-NEG', 'I-NEG', 'B-NEU', 'I-NEU']
        elif tagging_schema == 'MNER':
            return ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O']
        else:
            raise Exception("Invalid tagging schema %s..." % tagging_schema)

    def get_cl_labels(self, tagging_schema):
        """Gets the list of labels for this data set."""
        # tag sequence for opinion target extraction
        if tagging_schema == 'MATE':
            return ['O', 'EQ', 'B', 'I']
        # tag sequence for targeted sentiment
        elif tagging_schema == 'TS':
            return ['O', 'EQ', 'B-POS', 'I-POS', 'B-NEG', 'I-NEG', 'B-NEU', 'I-NEU']
        elif tagging_schema == 'MNER':
            return ['O', 'PER', 'LOC', 'ORG', 'MISC']
        else:
            raise Exception("Invalid tagging schema %s..." % tagging_schema)

    def _create_examples(self, text_path, image_path, set_type, tagging_schema):
        load_file = text_path
        examples = []
        count = 0
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            raw_words, raw_targets = [], []
            raw_word, raw_target = [], []
            imgs = []
            for line in lines:
                if line.startswith("IMGID:"):
                    img_id = line.strip().split('IMGID:')[1] + '.jpg'
                    imgs.append(img_id)
                    continue
                if line != "\n":
                    raw_word.append(line.split('\t')[0])
                    label = line.split('\t')[1][:-1]
                    if 'OTHER' in label:
                        label = label[:2] + 'MISC'
                    raw_target.append(label)
                else:
                    assert len(raw_word) == len(raw_target)
                    raw_words.append(raw_word)
                    raw_targets.append(raw_target)
                    guid = "%s-%s" % (set_type, count)
                    text_a = ' '.join(raw_word)
                    tags = raw_target
                    image_path_single = image_path + str(img_id)
                    count += 1
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=None, label_aspect=tags, image_path=image_path_single)
                    )
                    raw_word, raw_target = [], []
            return examples



glue_processors = {
    "twitter_15_bert_bottom-vit_mner": EntityImageProcessor,
    "twitter_17_bert_bottom-vit_mner": EntityImageProcessor,
}

glue_output_modes = {
    "twitter_15_bert_bottom-vit_mner": "classification",
    "twitter_17_bert_bottom-vit_mner": "classification",
}
