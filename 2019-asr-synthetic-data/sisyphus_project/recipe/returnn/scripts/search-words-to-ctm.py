#!/usr/bin/env python3

import os
from argparse import ArgumentParser


class SeqInfo:
    __slots__ = ("idx", "tag", "orth_raw", "orth_seq", "audio_path", "audio_start", "audio_end", "rec_name")


def parse_bliss_xml(filename):
    """
    This takes e.g. around 5 seconds for the Switchboard 300h train corpus.
    Should be as fast as possible to get a list of the segments.
    All further parsing and loading can then be done in parallel and lazily.
    :param str filename:
    :param boolean use_compressed_corpus:
    :return: nothing, fills self._segments
    """
    # Also see LmDataset._iter_bliss.
    import gzip
    import xml.etree.ElementTree as etree
    corpus_file = open(filename, 'rb')
    if filename.endswith(".gz"):
      corpus_file = gzip.GzipFile(fileobj=corpus_file)
    context = iter(etree.iterparse(corpus_file, events=('start', 'end')))
    elem_tree = []
    name_tree = []
    cur_recording = None
    cur_recording_name = None
    cur_recording_seg_nr = 0
    idx = 0
    seqs = []
    for event, elem in context:
      if event == "start":
        elem_tree += [elem]
        if elem.tag == "segment":
            cur_recording_seg_nr += 1
        if "name" not in elem.attrib and elem.tag == "segment":
            name_tree.append(str(cur_recording_seg_nr))
        else:
            name_tree.append(elem.attrib.get("name", None))
        if elem.tag == "recording":
          cur_recording = elem.attrib["audio"]
          cur_recording_name = elem.attrib["name"]
          cur_recording_seg_nr = 0
      if event == 'end' and elem.tag == "segment":
        info = SeqInfo()
        info.idx = idx
        info.tag = "/".join(name_tree)
        info.orth_raw = elem.find("orth").text
        info.audio_path = cur_recording
        info.audio_start = float(elem.attrib["start"])
        info.audio_end = float(elem.attrib["end"])
        info.rec_name = cur_recording_name
        seqs.append(info)
        idx += 1
        if elem_tree:
          elem_tree[0].clear()  # free memory
      if event == "end":
        assert elem_tree[-1] is elem
        elem_tree = elem_tree[:-1]
        name_tree = name_tree[:-1]
    return seqs



def main():
    argparser = ArgumentParser()
    argparser.add_argument("file", help="by Returnn search, in 'py' format, words")
    argparser.add_argument("--corpus", required=True, help="Bliss XML corpus")
    argparser.add_argument("--out", required=True, help="output CTM filename")
    argparser.add_argument("--only-segment-name", action="store_true", default=False)
    args = argparser.parse_args()
    d = eval(open(args.file, "r").read())
    assert isinstance(d, dict)  # seq_tag -> words string
    corpus = parse_bliss_xml(args.corpus)
    
    """
    example CTM:

    ;; <name> <track> <start> <duration> <word> <confidence> [<n-best>]
    ;; hub5_00/en_4156a/1 (301.85-302.48)
    en_4156a 1 301.850 0.110 oh 0.9505
    en_4156a 1 301.960 0.510 yeah 0.9966
    ;; hub5_00/en_4156a/2 (304.71-306.72)
    en_4156a 1 304.710 0.170 well 0.6720
    en_4156a 1 304.880 0.130 i'm 0.9975
    en_4156a 1 305.010 0.130 going 0.7758
    en_4156a 1 305.140 0.060 to 0.7755
    en_4156a 1 305.200 0.190 have 0.9999
    en_4156a 1 305.680 0.060 to 0.2071
    en_4156a 1 305.740 0.140 do 0.4953
    en_4156a 1 305.880 0.230 more 0.9929
    en_4156a 1 306.110 0.600 classes 0.9957
    ;; hub5_00/en_4156a/3 (307.63-311.16)
    en_4156a 1 307.690 0.290 now 0.9262
    en_4156a 1 307.980 0.210 i'm 0.8732
    ...
    """

    assert not os.path.exists(args.out)
    with open(args.out, "w") as out:
        out.write(";; <name> <track> <start> <duration> <word> <confidence> [<n-best>]\n")
        for seq in corpus:
            out.write(";; %s (%f-%f)\n" % (seq.tag, seq.audio_start, seq.audio_end))
            if args.only_segment_name:
              seq.tag = seq.tag.split("/")[-1]
            words = d[seq.tag].split()
            # Just linearly interpolate the start/end of each word.
            avg_dur = (seq.audio_end - seq.audio_start) * 0.9 / max(len(words), 1)
            for i in range(len(words)):
                out.write("%s 1 %f %f %s 0.99\n" % (seq.rec_name, seq.audio_start + i * avg_dur, avg_dur, words[i]))
    print("# Done.")


if __name__ == "__main__":
    #import better_exchook
    #better_exchook.install()
    main()
