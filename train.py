import options
import argparse
import evaluate
import models
import random
import func


def make_options():
    parser = argparse.ArgumentParser(description='train.py')
    options.model_opts(parser)
    options.train_opts(parser)
    options.data_opts(parser)
    return parser.parse_args()


def run_epoch(opt, model, feeder, optimizer, batches):
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
        print('------ITERATION {}, {}/{}, loss: {:>.4F}'.format(feeder.iteration, feeder.cursor, feeder.size, loss.tolist()))


def train(steps=200, evaluate_size=None):
    opt = make_options()
    model, optimizer, feeder, ckpt = models.load_or_create_models(opt, True)
    if ckpt is not None:
        _, last_accuracy = evaluate.evaluate_accuracy(model, feeder.dataset, size=evaluate_size)
    else:
        last_accuracy = 0
    while True:
        run_epoch(opt, model, feeder, optimizer, steps)
        _, accuracy = evaluate.evaluate_accuracy(model, feeder.dataset, size=evaluate_size)
        if accuracy > last_accuracy:
            models.save_models(opt, model, optimizer, feeder)
            last_accuracy = accuracy
            print('MODEL SAVED WITH ACCURACY {:>.2F}.'.format(accuracy))
        else:
            if random.randint(0, 4) == 0:
                models.restore(opt, model, optimizer, feeder)
                print('MODEL RESTORED {:>.2F}/{:>.2F}.'.format(accuracy, last_accuracy))
            else:
                print('CONTINUE TRAINING {:>.2F}/{:>.2F}.'.format(accuracy, last_accuracy))


train()