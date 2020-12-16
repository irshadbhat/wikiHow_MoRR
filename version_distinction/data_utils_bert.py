from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics import classification_report as cr

logger = logging.getLogger(__name__)

csv.field_size_limit(sys.maxsize)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_idsA, input_maskA, segment_idsA, input_idsB, input_maskB, segment_idsB, label_id):
        self.input_idsA = input_idsA
        self.input_maskA = input_maskA
        self.segment_idsA = segment_idsA
        self.input_idsB = input_idsB
        self.input_maskB = input_maskB
        self.segment_idsB = segment_idsB
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[2]
            text_b = line[3]
            label = "0"#line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        tokensA = tokens_a + [sep_token]
        #if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
        #    tokensA += [sep_token]
        segment_idsA = [sequence_a_segment_id] * len(tokensA)

        tokensB = tokens_b + [sep_token]
        segment_idsB = [sequence_a_segment_id] * len(tokensB)
        #if tokens_b:
        #    segment_idsB += [sequence_a_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokensA = tokensA + [cls_token]
            tokensB = tokensB + [cls_token]
            segment_idsA = segment_idsA + [cls_token_segment_id]
            segment_idsB = segment_idsB + [cls_token_segment_id]
        else:
            tokensA = [cls_token] + tokensA
            segment_idsA = [cls_token_segment_id] + segment_idsA
            tokensB = [cls_token] + tokensB
            segment_idsB = [cls_token_segment_id] + segment_idsB

        input_idsA = tokenizer.convert_tokens_to_ids(tokensA)
        input_idsB = tokenizer.convert_tokens_to_ids(tokensB)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_maskA = [1 if mask_padding_with_zero else 0] * len(input_idsA)
        input_maskB = [1 if mask_padding_with_zero else 0] * len(input_idsB)

        # Zero-pad up to the sequence length.
        padding_lengthA = max_seq_length - len(input_idsA)
        padding_lengthB = max_seq_length - len(input_idsB)
        if pad_on_left:
            input_idsA = ([pad_token] * padding_lengthA) + input_idsA
            input_maskA = ([0 if mask_padding_with_zero else 1] * padding_lengthA) + input_maskA
            segment_idsA = ([pad_token_segment_id] * padding_lengthA) + segment_idsA

            input_idsB = ([pad_token] * padding_lengthB) + input_idsB
            input_maskB = ([0 if mask_padding_with_zero else 1] * padding_lengthB) + input_maskB
            segment_idsB = ([pad_token_segment_id] * padding_lengthB) + segment_idsB
        else:
            input_idsA = input_idsA + ([pad_token] * padding_lengthA)
            input_maskA = input_maskA + ([0 if mask_padding_with_zero else 1] * padding_lengthA)
            segment_idsA = segment_idsA + ([pad_token_segment_id] * padding_lengthA)

            input_idsB = input_idsB + ([pad_token] * padding_lengthB)
            input_maskB = input_maskB + ([0 if mask_padding_with_zero else 1] * padding_lengthB)
            segment_idsB = segment_idsB + ([pad_token_segment_id] * padding_lengthB)

        assert len(input_idsA) == max_seq_length
        assert len(input_maskA) == max_seq_length
        assert len(segment_idsA) == max_seq_length
        assert len(input_idsB) == max_seq_length
        assert len(input_maskB) == max_seq_length
        assert len(segment_idsB) == max_seq_length

        if output_mode == "classification":
            label_id = 0#label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokensA]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_idsA]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_maskA]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_idsA]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_idsA=input_idsA,
                              input_maskA=input_maskA,
                              segment_idsA=segment_idsA,
                              input_idsB=input_idsB,
                              input_maskB=input_maskB,
                              segment_idsB=segment_idsB,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


processors = {
    "mrpc": MrpcProcessor,
}

output_modes = {
    "mrpc": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "mrpc": 2,
}
