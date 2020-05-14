#!/usr/bin/env python3

import os
from argparse import ArgumentParser


def main():
    argparser = ArgumentParser()
    argparser.add_argument("file", help="by Returnn search, in 'py' format")
    argparser.add_argument("--out", required=True, help="output filename")
    args = argparser.parse_args()
    d = eval(open(args.file, "r").read())
    assert isinstance(d, dict)  # seq_tag -> bpe string
    assert not os.path.exists(args.out)
    map_special = {'L': '[LAUGHTER]', 'N': '[NOISE]', 'V': '[VOCALIZED-NOISE]','S': '[SILENCE]'}
    with open(args.out, "w") as out:
        out.write("{\n")
        for seq_tag, txt in sorted(d.items()):
            seq = [w.strip() for w in txt.split('  ' if "<eow>" not in txt else "<eow>")]
            seq = ' '.join([x.replace(' ','') if x not in map_special else map_special[x] for x in seq]).strip()
            out.write("%r: %r,\n" %(seq_tag, seq))
        out.write("}\n")
    print("# Done.")


if __name__ == "__main__":
    #import better_exchook
    #better_exchook.install()
    main()
