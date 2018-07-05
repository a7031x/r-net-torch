import options
import argparse
import evaluate
import models
import random
import func
import utils


def make_options():
    parser = argparse.ArgumentParser(description='train.py')
    options.model_opts(parser)
    options.train_opts(parser)
    options.data_opts(parser)
    return parser.parse_args()


def run_epoch(opt, model, feeder, optimizer, batches):
    model.train()
    nbatch = 0
    criterion = models.make_loss_compute()
    while nbatch < batches:
        _, cs, qs, chs, qhs, y1s, y2s = feeder.next(opt.batch_size)
        nbatch += 1
        logits1, logits2 = model(func.tensor(cs), func.tensor(qs), func.tensor(chs), func.tensor(qhs))
        t1, t2 = func.tensor(y1s), func.tensor(y2s)
        loss = (criterion(logits1, t1) + criterion(logits2, t2)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('------ITERATION {}, {}/{}, epoch: {:>.2F}% loss: {:>.4F}'.format(feeder.iteration, feeder.cursor, feeder.size, 100.0*nbatch/batches, loss.tolist()))


class Logger(object):
    def __init__(self, opt):
        self.output_file = opt.summary_file
        self.lines = list(utils.read_all_lines(self.output_file))


    def __call__(self, message):
        print(message)
        self.lines.append(message)
        utils.write_all_lines(self.output_file, self.lines)


def train(steps=400, evaluate_size=None):
    opt = make_options()
    model, optimizer, feeder, ckpt = models.load_or_create_models(opt, True)
    log = Logger(opt)
    if ckpt is not None:
        _, last_accuracy = evaluate.evaluate_accuracy(model, feeder.dataset, batch_size=opt.batch_size, size=evaluate_size)
    else:
        last_accuracy = 0
    while True:
        run_epoch(opt, model, feeder, optimizer, steps)
        em, accuracy = evaluate.evaluate_accuracy(model, feeder.dataset, batch_size=opt.batch_size, size=evaluate_size)
        if accuracy > last_accuracy:
            models.save_models(opt, model, optimizer, feeder)
            last_accuracy = accuracy
            log('MODEL SAVED WITH ACCURACY EM:{:>.2F}, F1:{:>.2F}.'.format(em, accuracy))
        else:
            if random.randint(0, 4) == 0:
                models.restore(opt, model, optimizer, feeder)
                log('MODEL RESTORED {:>.2F}/{:>.2F}.'.format(accuracy, last_accuracy))
            else:
                log('CONTINUE TRAINING {:>.2F}/{:>.2F}.'.format(accuracy, last_accuracy))


train()