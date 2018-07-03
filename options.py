import argparse

def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-char_hidden_size', type=int, default=200)
    group.add_argument('-ckpt_path', type=str, default='./checkpoint/model.pt')


def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-batch_size', type=int, default=64)
    group.add_argument('-learning_rate', type=float, default=0.001)
    group.add_argument('-dropout', type=float, default=0.3)


def evaluate_opts(parser):
    group = parser.add_argument_group('evaluate')
    group.add_argument('-beam_size', type=int, default=5)
    group.add_argument('-max_length', type=int, default=20)
    group.add_argument('-min_length', type=int, default=5)
    group.add_argument('-batch_size', type=int, default=32)
    group.add_argument('-best_k_questions', type=int, default=3)
    group.add_argument('-output_file', type=str, default='./output/evaluate.txt')
    group.add_argument('-dropout', type=float, default=0)


def preprocess_opts(parser):
    group = parser.add_argument_group('preprocess')
    group.add_argument('-squad_train_file', type=str, default='./data/squad/train-v1.1.json')
    group.add_argument('-squad_dev_file', type=str, default='./data/squad/dev-v1.1.json')
    group.add_argument('-squad_test_file', type=str, default='./data/squad/dev-v1.1.json')
    group.add_argument('-glove_word_emb_file', type=str, default='./data/glove/glove.840B.300d.txt')


def data_opts(parser):
    group = parser.add_argument_group('data')
    group.add_argument('-word_dim', type=int, default=300)
    group.add_argument('-char_dim', type=int, default=8)
    group.add_argument('-word_emb_file', type=str, default='./generate/emb.word.json')
    group.add_argument('-char_emb_file', type=str, default='./generate/emb.char.json')
    group.add_argument('-w2i_file', type=str, default='./generate/w2i.json')
    group.add_argument('-c2i_file', type=str, default='./generate/c2i.json')

    group.add_argument('-train_example_file', type=str, default='./generate/example.train.json')
    group.add_argument('-dev_example_file', type=str, default='./generate/example.dev.json')
    group.add_argument('-test_example_file', type=str, default='./generate/example.test.json')

    group.add_argument('-train_eval_file', type=str, default='./generate/eval.train.json')
    group.add_argument('-dev_eval_file', type=str, default='./generate/eval.dev.json')
    group.add_argument('-test_eval_file', type=str, default='./generate/eval.test.json')
