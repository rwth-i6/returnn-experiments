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
    with open(args.out, "w") as out:
        out.write("{\n")
        for seq_tag, txt in sorted(d.items()):
            if '#' in seq_tag:
              print(seq_tag)
              tag_split = seq_tag.split("/")
              recording_name, segment_name = tag_split[2].split('#')
              seq_tag = tag_split[0] + "/" + recording_name + "/" + segment_name
              print(seq_tag)
            out.write("%r: %r,\n" % (seq_tag, txt.replace("@@ ", "")))
        out.write("}\n")
    print("# Done.")


if __name__ == "__main__":
    #import better_exchook
    #better_exchook.install()
    main()
