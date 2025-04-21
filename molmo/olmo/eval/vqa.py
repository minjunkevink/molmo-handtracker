"""Various VQA metrics used by different datasets"""
import re
import string
from collections import Counter
from typing import Optional, List

import editdistance
import numpy as np

from olmo.eval import mmmu_eval_utils, math_vista_utils

contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
    "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
    "youll": "you'll", "youre": "you're", "youve": "you've"}

manualMap = {
    'none': '0',
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10'
}

articles = ['a','an','the']

punct = [
    ';', r"/", '[', ']', '"', '{', '}',
    '(', ')', '=', '+', '\\', '_', '-',
    '>', '<', '@', '`', ',', '?', '!']

periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(\d)(\,)(\d)")


def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("",outText,re.UNICODE)
    return outText


def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def preprocess_answer(ans, cache={}):
    if ans in cache:
        return cache[ans]
    ans = ans.replace('\n', ' ')
    ans = ans.replace('\t',' ')
    ans = ans.lower().strip()
    preprocessed = processDigitArticle(processPunctuation(ans))
    cache[ans] = preprocessed
    return preprocessed


def vqa_score(target, pred):
    """
    Evaluation with VQA 2 style preprocessing
    """
    pred = preprocess_answer(pred)
    if isinstance(target, list):
        target = Counter(preprocess_answer(x) for x in target)
        return min(target[pred] / 3.0, 1)
    else:
        return float(pred == target)


def a_okvqa_score(target, pred):
    # A-OK-VQA eval scripts don't seem to do any answer pre-processing
    target = Counter([x.lower().strip() for x in target])
    return min(target[pred.lower().strip()] / 3.0, 1)


def select_mc_option(target, options):
    """
    Selects a multiple-choice option based on the model output

    The output is should exactly match one of the option, but contains
    some heuristic fallbacks in case the does not occur
    """
    target = target.lower().strip()
    n = len(options)
    options = [x.lower().strip() for x in options]
    assert len(set(options)) == n
    for ix, option in enumerate(options):
        if option == target:
            return ix

    contains = []
    for ix, option in enumerate(options):
        if target in option:
            contains.append(ix)
    if len(contains) == 1:
        return contains[0]
    distances = [editdistance.eval(opt, target) for opt in options]
    return np.argmin(distances)


# From https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py
def anls_metric(target: str, prediction: str, theta: float = 0.5):
    """Calculates ANLS for DocVQA.

    There does not seem to be an official evaluation script.
    Public implementation on which this implementation is based:
    https://github.com/herobd/layoutlmv2/blob/main/eval_docvqa.py#L92

    Original paper (see Eq 1): https://arxiv.org/pdf/1907.00490.pdf

    Args:
      target: Target string.
      prediction: Predicted string.
      theta: Filter threshold set to 0.5 for DocVQA.

    Returns:
      ANLS score.
    """
    # Lowercase is not in https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py
    # However https://rrc.cvc.uab.es/?ch=17&com=tasks says
    #  - "Answers are not case sensitive"
    #  - "Answers are space sensitive"
    edit_distance = editdistance.eval(target.lower(), prediction.lower())
    normalized_ld = edit_distance / max(len(target), len(prediction))
    return 1 - normalized_ld if normalized_ld < theta else 0


# From https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


# From https://github.com/MMMU-Benchmark/MMMU/blob/main/eval/main_parse_and_eval.py
def mmmu_score(
    target: List[str],
    response: str,
    metadata: dict,
):
    question_type = metadata["question_type"]
    if question_type == "multiple-choice":
        options = metadata["options"]
        options = [opt for opt in options if len(opt) > 0]
        all_choices = [chr for chr in string.ascii_uppercase[:len(options)]]
        index2ans = {chr: option for chr, option in zip(all_choices, options)}
        parsed_pred = mmmu_eval_utils.parse_multi_choice_response(response, all_choices, index2ans)
        correct = mmmu_eval_utils.eval_multi_choice(target, parsed_pred)
    else: # open
        parsed_pred = mmmu_eval_utils.parse_open_response(response)
        correct = mmmu_eval_utils.eval_open(target, parsed_pred)
    return float(correct)


def real_world_qa_score(
    target: str,
    prediction: str,
    metadata: dict,
):
    question_type = metadata["question_type"]
    if question_type == "multiple_choice":
        options = ["A", "B", "C", "D"]
        pred_idx = select_mc_option(prediction, options)
        gt_idx = options.index(target)
        score = pred_idx == gt_idx
    else:
        pred = preprocess_answer(prediction)
        gt = preprocess_answer(target)
        score = float(pred == gt)
    return score


def math_vista_score(
    response: str,
    metadata: dict,
    openai_api_key: str,
    use_api: bool = True,
):
    # extract answer using GPT-4.
    pid = metadata["example_id"]
    question_type = metadata["question_type"]
    answer_type = metadata["answer_type"]
    choices = metadata["choices"]
    target = metadata["answer"]
    query = metadata["query"]

    if use_api:
        extraction = math_vista_utils.extract_answer(
            pid, response, question_type, answer_type, choices, query, openai_api_key,
        )
    else:
        if question_type == "multi_choice":
            options = [chr(ord("A") + i) for i in range(len(choices))]
            pred_idx = select_mc_option(response, options)
            extraction = choices[pred_idx]
        else:
            if answer_type == "integer":
                try:
                    extraction = str(int(response))
                except:
                    extraction = response
            elif answer_type == "float":
                try:
                    extraction = str(float(response))
                except:
                    extraction = response
            else:
                extraction = response

    # calculate score
    precision = metadata["precision"]

    # normalize the extracted answer to match the answer type
    prediction = math_vista_utils.normalize_extracted_answer(
        extraction, choices, question_type, answer_type, precision,
    )

    # verify the prediction is true or false
    true_false = math_vista_utils.safe_equal(prediction, target)

    return true_false
