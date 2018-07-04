import data
import func
import re
import string
import utils
from collections import Counter


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_em_f1(feeder, pids, predict_start, predict_end, target_start, target_end):
    predict = feeder.ids_to_sent(pids[predict_start:predict_end+1])
    target = feeder.ids_to_sent(pids[target_start:target_end+1])
    em = metric_max_over_ground_truths(exact_match_score, predict, target)
    f1 = metric_max_over_ground_truths(f1_score, predict, target)
    return em, f1


def evaluate_batch(feeder, cs, y1p, y2p, y1s, y2s):
    total_em, total_f1, total = 0, 0, 0
    for pids, predict_start, predict_end, target_start, target_end in zip(cs, y1p, y2p, y1s, y2s):
        em, f1 = evaluate_em_f1(feeder, pids, predict_start, predict_end, target_start, target_end)
        total_em += em
        total_f1 += f1
        total += 1
    return total_em, total_f1, total


def evaluate_accuracy(model, dataset, batch_size=64, size=None, output_file='./output/dev.txt'):
    feeder = data.TrainFeeder(dataset, batch_size)
    feeder.prepare('dev')
    size = size or feeder.size
    feeder.sort(size)
    lines = []
    total_em, total_f1, total = 0, 0, 0
    while feeder.cursor < size:
        _, cs, qs, chs, qhs, y1s, y2s = feeder.next(batch_size)
        logits1, logits2 = model(func.tensor(cs), func.tensor(qs), func.tensor(chs), func.tensor(qhs))
        y1p, y2p = model.calc_span(logits1, logits2)
        for pids, qids, lable_start, label_end, predict_start, predict_end in zip(cs, qs, y1s, y2s, y1p, y2p):
            lines.append('--------------------------------')
            lines.append(feeder.ids_to_sent(pids))
            lines.append('question:  ' + feeder.ids_to_sent(qids))
            lines.append('reference: ' + feeder.ids_to_sent(pids[lable_start:label_end+1]))
            lines.append('predict:   ' + feeder.ids_to_sent(pids[predict_start:predict_end+1]))
        em, f1, bs = evaluate_batch(feeder, cs, y1p.tolist(), y2p.tolist(), y1s, y2s)
        total_em += em
        total_f1 += f1
        total += bs
        print('{}/{}'.format(feeder.cursor, size))

    exact_match = total_em / total * 100
    f1 = total_f1 / total * 100
    message = 'EM: {:>.4F}, F1: {:>.4F}, Total: {}'.format(exact_match, f1, total)
    lines.append(message)
    utils.write_all_lines(output_file, lines)
    print('evauation finished with ' + message)
    return exact_match, f1