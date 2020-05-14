#!/usr/bin/env python3

import os
import sys
sys.path.insert(0,'/u/tng/setups/switchboard/2019-07-29--att-bpe1k/crnn')
from argparse import ArgumentParser
from LmDataset import Lexicon

def main():
    argparser = ArgumentParser()
    argparser.add_argument("file", help="by Returnn search, in 'py' format")
    argparser.add_argument("--out", required=True, help="output filename")
    args = argparser.parse_args()
    d = eval(open(args.file, "r").read())
    assert isinstance(d, dict)  # seq_tag -> bpe string
    assert not os.path.exists(args.out)
    
    lex_out = {}
    lexicon_file = "/work/asr4/zeyer/backup/switchboard/tuske-train.lex.v1_0_3.ci.gz"
    lexicon = Lexicon(lexicon_file)
    for word in lexicon.lemmas:
        list_phones = []
        for item in lexicon.lemmas[word]['phons']:
            list_phones.append(item['phon'])
        lex_out[word] = list_phones

    duplicates = {}  # phone -> count
    for word, phones in sorted(lex_out.items()):
        for phone in phones:
            if phone in duplicates:
                lex_out[word].remove(phone)
                lex_out[word].insert(0, '%s #%s' % (phone, duplicates[phone]))
                duplicates[phone] += 1
            else:
                duplicates[phone] = 1
    
    rev_lex = {v[0]: k for k,v in lex_out.items() if len(v)>0}
    with open(args.out, "w") as out:
        out.write("{\n")
        for seq_tag, txt in sorted(d.items()):
            seq = [w.strip() for w in txt.split("<eow>")]
            seq = ' '.join([rev_lex[x] if x in rev_lex else "[UNKNOWN]" for x in seq if len(x)>0]).strip()
            out.write("%r: %r,\n" %(seq_tag, seq))
        out.write("}\n")
    print("# Done.")


if __name__ == "__main__":
    #import better_exchook
    #better_exchook.install()
    main()
