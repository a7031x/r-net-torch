from allennlp.modules.elmo import Elmo, batch_to_ids
import func
import torch

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


class ElmoEmbedding:
    def __init__(self):
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0)
        if func.gpu_available():
            self.elmo = self.elmo.cuda()
        self.elmo.eval()


    def convert(self, sentences):
        character_ids = func.tensor(batch_to_ids(sentences))
        m = self.elmo(character_ids)
        embeddings = m['elmo_representations']
        embeddings = torch.cat(embeddings, -1)
        return embeddings


    @property
    def dim(self):
        return 1024*2


if __name__ == '__main__':
    import torch

    elmo = ElmoEmbedding()
    # use batch_to_ids to convert sentences to character ids
    sentences = [['First', 'sentence', '.'], ['Another', '.']]
    embeddings = elmo.convert(sentences)
    print(embeddings.shape)
    # m['elmo_representations'] is length two list of tensors.
    # Each element contains one layer of ELMo representations with shape
    # (2, 3, 1024).
    #   2    - the batch size
    #   3    - the sequence length of the batch
    #   1024 - the length of each ELMo vector