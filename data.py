import utils

NULL = '<NULL>'
OOV = '<OOV>'

NULL_ID = 0
OOV_ID = 1

class Dataset(object):
    def __init__(self, opt):
        self.train_examples = utils.load_json(opt.train_example_file)
        self.dev_examples = utils.load_json(opt.dev_example_file)
        self.test_examples = utils.load_json(opt.test_example_file)
        self.word_emb_file = utils.load_json(opt.word_emb_file)
        self.char_emb_file = utils.load_json(opt.char_emb_file)
        self.w2i = utils.load_json(opt.w2i_file)
        self.c2i = utils.load_json(opt.c2i_file)


class Feeder(object):
    def __init__(self, dataset):
        pass


if __name__ == '__main__':
    import argparse
    import options
    parser = argparse.ArgumentParser()
    options.data_opts(parser)
    opt = parser.parse_args()
    dataset = Dataset(opt)
    
    assert len(dataset.word_emb_file) == len(dataset.w2i)
    assert len(dataset.char_emb_file) == len(dataset.c2i)

    print('examples: {}/{}/{}'.format(len(dataset.train_examples), len(dataset.dev_examples), len(dataset.test_examples)))
    print('vocab_size: {}/{}'.format(len(dataset.word_emb_file), len(dataset.char_emb_file)))