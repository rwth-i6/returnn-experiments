from recipe.default_values import SUBWORD_NMT_DIR

from sisyphus import *
Path = setup_path(__package__)


class CreateSubwordsAndVocab(Job):

  def __init__(self, text, num_segments, subword_nmt=SUBWORD_NMT_DIR):
    self.text = text
    self.num_segments = num_segments
    self.subword_nmt = subword_nmt

    self.out_bpe = self.output_path('bpe.codes')
    self.out_vocab = self.output_path('bpe.vocab')
    self.out_vocab_size = self.output_var('bpe_vocab_size')

  def run(self):
    with tk.mktemp() as self.tmp:
      self.sh('zcat -f `cf {text}` > {tmp}')

      self.sh('python3 {subword_nmt}/learn_bpe.py -s {num_segments} --input {tmp} --output {out_bpe}')
      self.sh('python3 {subword_nmt}/create-py-vocab.py --txt {tmp} --bpe {out_bpe} --unk "<unk>" --out {out_vocab}')

      self.sh('rm {tmp}')

      vocab = eval(open(str(self.out_vocab)).read())
      self.out_vocab_size.set(len(set(vocab.values())))


  def tasks(self):
    rqmt = {'cpu': 1,
            'mem': 16,
            'time': 8,
            }

    yield Task('run', rqmt=rqmt)
