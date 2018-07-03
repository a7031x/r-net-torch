import utils
import random

NULL = '<NULL>'
OOV = '<OOV>'

NULL_ID = 0
OOV_ID = 1

class Dataset(object):
    def __init__(self, opt):
        self.train_set = utils.load_json(opt.train_example_file)
        self.dev_set = utils.load_json(opt.dev_example_file)
        self.test_set = utils.load_json(opt.test_example_file)
        self.word_emb = utils.load_json(opt.word_emb_file)
        self.char_emb = utils.load_json(opt.char_emb_file)
        self.w2i = utils.load_json(opt.w2i_file)
        self.c2i = utils.load_json(opt.c2i_file)


class Feeder(object):
    def __init__(self, dataset):
        self.dataset = dataset


    def word_to_id(self, word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in self.dataset.w2i:
                return self.dataset.w2i[each]
        return OOV_ID


    def char_to_id(self, char):
        return self.dataset.c2i[char] if char in self.dataset.c2i else OOV_ID


    def word_to_cids(self, word):
        return [self.char_to_id(char) for char in word]


    def sent_to_ids(self, sent):
        return [self.word_to_id(x) for x in sent]


    def sent_to_cids(self, sent):
        return[self.word_to_cids(x) for x in sent]


    def parse_example(self, example):
        context = example['context_tokens']
        question = example['question_tokens']
        c = self.sent_to_ids(context)
        q = self.sent_to_ids(question)
        ch = self.sent_to_cids(context)
        qh = self.sent_to_cids(question)
        y1, y2 = [0.0] * len(c), [0.0] * len(c)
        start, end = example['y1s'][-1], example['y2s'][-1]
        y1[start] = y2[end] = 1.0
        return example['id'], c, q, ch, qh, y1, y2


class TrainFeeder(Feeder):
    def __init__(self, dataset):
        super(TrainFeeder, self).__init__(dataset)


    def prepare(self, type):
        if type == 'train':
            self.prepare_data(self.dataset.train_set)
            self.shuffle_index()
        elif type == 'dev':
            self.prepare_data(self.dataset.dev_set)
        else:
            self.prepare_data(self.dataset.test_set)
        self.size = len(self.data)
        self.cursor = 0


    def prepare_data(self, dataset):
        self.data = dataset
        self.data_index = list(range(len(self.data)))


    def state(self):
        return self.iteration, self.cursor, self.data_index


    def load_state(self, state):
        self.iteration, self.cursor, self.data_index = state


    def shuffle_index(self):
        random.shuffle(self.data_index)


    def eof(self):
        return self.cursor == self.size


    def next(self, batch_size):
        if self.eof():
            self.iteration += 1
            self.cursor = 0
            if self.data == self.dataset.train_set:
                self.shuffle_index()

        size = min(self.size - self.cursor, batch_size)
        batch = self.data_index[self.cursor:self.cursor+size]
        batch = [self.data[idx] for idx in batch]
        ids, cs, qs, chs, qhs, y1s, y2s = [], [], [], [], [], [], []
        for example in batch:
            id, c, q, ch, qh, y1, y2 = self.parse_example(example)
            ids.append(id)
            cs.append(c)
            qs.append(q)
            chs.append(ch)
            qhs.append(qh)
            y1s.append(y1)
            y2s.append(y2)
        self.cursor += size
        return align(ids), align(cs), align(qs), align(chs), align(qhs), align(y1s), align(y2s)

                
def align1d(value, mlen, fill=0):
    return value + [fill] * (mlen - len(value))


def align2d(values, fill=0):
    mlen = max([len(row) for row in values])
    return [align1d(row, mlen, fill) for row in values]


def align3d(values, fill=0):
    lengths = [[len(x) for x in y] for y in values]
    maxlen0 = max([max(x) for x in lengths])
    maxlen1 = max([len(x) for x in lengths])
    for row in values:
        for line in row:
            line += [fill] * (maxlen0 - len(line))
        row += [([fill] * maxlen0)] * (maxlen1 - len(row))
    return values


def align(values, fill=0):
    dim = 0
    inp = values
    while isinstance(inp, list):
        dim += 1
        inp = inp[0]
    if dim == 1:
        return values
    elif dim == 2:
        return align2d(values, fill)
    elif dim == 3:
        return align3d(values, fill)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    import argparse
    import options
    parser = argparse.ArgumentParser()
    options.data_opts(parser)
    options.train_opts(parser)
    opt = parser.parse_args()
    dataset = Dataset(opt)
    
    assert len(dataset.word_emb_file) == len(dataset.w2i)
    assert len(dataset.char_emb_file) == len(dataset.c2i)

    print('examples: {}/{}/{}'.format(len(dataset.train_set), len(dataset.dev_set), len(dataset.test_set)))
    print('vocab_size: {}/{}'.format(len(dataset.word_emb_file), len(dataset.char_emb_file)))
    feeder = TrainFeeder(dataset)
    feeder.prepare('train')
    ids, c, q, ch, qh, y1, y2 = feeder.next(opt.batch_size)
    assert len(ids) == opt.batch_size