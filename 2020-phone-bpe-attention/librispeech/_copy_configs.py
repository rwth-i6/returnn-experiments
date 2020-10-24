#!/usr/bin/env python3

"""
Here we copy all relevant copies from our internal filesystem to this public repo.

In our internal filesystem, the relevant paths are here:

/u/zeineldeen/setups/librispeech/2020-08-31--att-phon

The Latex code of the paper can be found here:
/u/zeineldeen/publications/2020-phone-bpe-att-paper

Configs are mapped to the rows of the tables
"""

import os
import shutil
import better_exchook


base_dirs = [
    "/u/zeineldeen/setups/librispeech/2020-08-31--att-phon"
]

table6_configs = [
    "base2.conv2l.specaug.curric3.single-phon.msl228.end-phon", # single (#)
    "base2.conv2l.specaug.curric3.phon-bpe-5k.msl81.run2", # phone-bpe-5k
    "base2.conv2l.specaug.curric3.phon-bpe-10k.msl74.run2", # phone-bpe-10k
    "base2.conv2l.specaug.curric3.phon-bpe-20k.msl70", # phone-bpe-20k
    "base2.conv2l.specaug.curric3.char-bpe-5k.msl81", # char-bpe-5k
    "base2.conv2l.specaug.curric3.char-bpe-10k.msl75.pre-bs20k", # char-bpe-10k
    "base2.conv2l.specaug.curric3.char-bpe-20k.msl70" # char-bpe-20k
]

configs = table6_configs

def get_filename(config):
    for base_dir in base_dirs:
        fn = "%s/config-train/%s.config" % (base_dir, config)
        print(fn)
        if os.path.exists(fn):
            return fn
    raise Exception("not found: %s" % config)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    for base_dir in base_dirs:
        assert os.path.exists(base_dir)

    for config in configs:
        fn = get_filename(config)
        local_fn = "%s.config" % config
        if not os.path.exists(local_fn):
            shutil.copy(fn, local_fn)


if __name__ == "__main__":
    better_exchook.install()
    main()
