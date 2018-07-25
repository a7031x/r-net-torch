
class AutoDecay(object):
    def __init__(self, optimizer, patience=3, not_learning_counter=5):
        self.learning_rate = optimizer.defaults['lr']
        self.optimizer = optimizer
        self.counter = 0
        self.patience = patience
        self.not_learning_counter = not_learning_counter
        self.decay_counter= 0


    def better(self):
        self.counter = 0
        self.decay_counter = 0


    def worse(self):
        self.counter += 1
        if self.counter >= self.patience:
            self.learning_rate /= 2
            for group in self.optimizer.param_groups:
                group['lr'] = self.learning_rate
            self.counter = 0
            self.decay_counter += 1


    def should_stop(self):
        return self.decay_counter >= self.not_learning_counter

