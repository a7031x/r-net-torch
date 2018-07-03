import options
import argparse
import evaluate
import models
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
        ids, cs, qs, chs, qhs, y1s, y2s = feeder.next(opt.batch_size)
        batch_size = len(ids)
        nbatch += 1
        model(func.tensor(cs), func.tensor(qs), func.tensor(chs), func.tensor(qhs))


def train(steps=200, evaluate_size=500):
    opt = make_options()
    model, optimizer, feeder, ckpt = models.load_or_create_models(opt, True)
    '''
    if ckpt is not None:
        last_accuracy = evaluate.evaluate_accuracy(model, feeder.dataset, size=evaluate_size)
    else:
        last_accuracy = 0
    '''
    while True:
        run_epoch(opt, model, feeder, optimizer, steps)
        '''
        accuracy = evaluate.evaluate_accuracy(model, feeder.dataset, size=evaluate_size)
        if accuracy > last_accuracy:
            utils.mkdir(config.checkpoint_folder)
            models.save_models(opt, generator, discriminator, g_optimizer, d_optimizer, feeder)
            last_accuracy = accuracy
            print('MODEL SAVED WITH ACCURACY {:>.2F}.'.format(accuracy))
        else:
            if random.randint(0, 4) == 0:
                models.restore(generator, discriminator, g_optimizer, d_optimizer)
                print('MODEL RESTORED {:>.2F}/{:>.2F}.'.format(accuracy, last_accuracy))
            else:
                print('CONTINUE TRAINING {:>.2F}/{:>.2F}.'.format(accuracy, last_accuracy))
        '''


train()