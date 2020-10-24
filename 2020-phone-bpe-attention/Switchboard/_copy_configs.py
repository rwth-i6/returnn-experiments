#!/usr/bin/env python3

"""
Here we copy all relevant copies from our internal filesystem to this public repo.

In our internal filesystem, the relevant paths are here:

/u/tng/setups/switchboard/2019-07-29--att-bpe1k
/u/zeineldeen/setups/switchboard/2020-01-21--att-phon
/u/zeineldeen/setups/switchboard/2020-10-10--att-phon-paper

The Latex code of the paper can be found here:
/u/zeineldeen/publications/2020-phone-bpe-att-paper

Configs are mapped to the rows of the tables
"""

import os
import shutil
import better_exchook


base_dirs = [
    "/u/tng/setups/switchboard/2019-07-29--att-bpe1k",
    "/u/zeineldeen/setups/switchboard/2020-01-21--att-phon",
    "/u/zeineldeen/setups/switchboard/2020-10-10--att-phon-paper"
]

# Using simple decoder + word disamb (if needed)
table1_configs = [
    #------------------------ Single phonemes variants ------------------------#

    "base2.conv2l.specaug4.phone_orth_eow.disamb.msl228",
    # epoch : 190 , #num : 62

    "base2.conv2l.specaug4.phone_orth.end_phon.disamb.end_phon.msl180.fix",
     # epoch: 200, #num: 118

    #-------------------------- Phoneme-BPE -----------------------------------#

    "phone-uni-bpe50.base2.conv2l.specaug4a.disamb.mql130",
    # epoch : 199, #num : 151

    "phone-uni-bpe100.base2.conv2l.specaug4a.disamb.mql143",
    # epoch : 160, #num : 201

    "phone-uni-bpe500.base2.conv2l.specaug4a.disamb.mql125",
    # epoch : 182, #num : 592, bpe 500

    "phone-uni-bpe1k.base2.conv2l.specaug4a.disamb.mql110",
    # epoch : 200, #num : 1091

    "phone-uni-bpe2k.base2.conv2l.specaug4a.disamb.mql96",
    # epoch : 200, #num : 2086

    "phone-uni-bpe5k.base2.conv2l.specaug4a.disamb.mql87",
    # epoch : 191, #num : 5033

    #---------------------------- Single char ---------------------------------#

    "base2.conv2l.specaug4.char.mql250",
    # epoch : 160, #num : 35

    #----------------------------- Char-BPE -----------------------------------#

    "base2.conv2l.specaug4.bpe50.mql154",
    # epoch : 197, #num : 126, bpe 50

    "base2.conv2l.specaug4.bpe100.mql134",
    # epoch : 190, #num : 176, bpe 100

    "base2.conv2l.specaug4.bpe500.mql88",
    # epoch : 199, #num : 534, bpe 500

    "base2.conv2l.specaug4.bpe-1k",
    # epoch : 186, #num : 1030

    "base2.conv2l.specaug4.bpe-2k.mql67",
    # epoch : 195, #num : 2026

    "base2.conv2l.specaug4.bpe-5k.mql60",
    # epoch : 179, #num : 4980

    "base2.conv2l.specaug4.bpe-10k.mql57",
    # epoch : 173, #num : 9795

    "base2.conv2l.specaug4.bpe-20k.mql56",
    # epoch : 162, #num : 18611

    #------------------------------ Words -------------------------------------#

    "base2.conv2l.specaug4.words_orth.mql55",
    # epoch : 173
]

# Using advanced decoder + word-level LM + word disamb
# Models used from table 1 and evaluated with the adv. decoder here for comparison
# Each is mapped to 3 rows in the table
table2_configs = [
    "base2.conv2l.specaug4.phone_orth_eow.disamb.msl228", # single phoneme + eow
    "phone-uni-bpe500.base2.conv2l.specaug4a.disamb.mql125", # phone-bpe-500
    "base2.conv2l.specaug4.char.mql250", # char
    "base2.conv2l.specaug4.bpe500.mql88" # char-bpe-500
]

# Study with or without word disamb
# First is mapped the rows 1,2
# Second is mapped to row 3
# Third is mapped to rows 4,5
# Fourth is mapped to row 6
table3_configs = [
    "base2.conv2l.specaug4.phone_orth_eow.disamb.msl228", # single phoneme + disamb
    "base2.conv2l.specaug4.phone_orth_eow.wo_disamb.msl228", # single phoneme (no disamb)
    "phone-uni-bpe500.base2.conv2l.specaug4a.disamb.mql125", # phone-bpe-500 + disamb
    "base2.conv2l.specaug4a.phone-uni-bpe500.max_seq_len100" # phone-bpe-500 (no disamb)
]

# Comparison between EOW symbol and word-end-phone (#) symbol
table4_configs = [
    "base2.conv2l.specaug4.phone_orth.no_eow.no_disamb.red6.max_seq_len180", # no EOW
    "base2.conv2l.specaug4.phone_orth_eow.disamb.msl228", # with EOW
    "base2.conv2l.specaug4.phone_orth.end_phon.no_disamb.end_phon.msl180" # with #
]

# Literature comparison
table5_configs = [
    "base2.conv2l.specaug4.bpe500.mql88", # char-bpe-500
    "base2.conv2l.specaug4.phone_orth_eow.wo_disamb.msl228", # single phoneme + eow
    "base2.conv2l.specaug4a.phone-uni-bpe500.max_seq_len100", # phone-bpe-500
    "base2.conv2l.specaug4a.phone-uni-bpe500.max_seq_len200" # phone-bpe-500 (max len)
]


configs = []
for table_configs in [table1_configs, table2_configs, table3_configs, table4_configs, table5_configs]:
  configs += table_configs


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
