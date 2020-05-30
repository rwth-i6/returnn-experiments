This is for the recognition experiments on longer seqs,
where we just concated seqs within one Switchboard recording
(for recognition only, not for training).

This was part of a more complex [Sisyphus](https://github.com/rwth-i6/sisyphus) pipeline.
For now, we don't publish the whole pipeline (this is work in progress - we plan to do later)
but just the script/code to create the corpora with concatenated seqs in `concat_seqs.py`.

The main algorithm is fairly simple:
It will go through a recording, and take C consecutive sequences, and concatenates them together,
until it gets to the end of the recording, where less than C sequences are maybe concatenated together.

It should be easy to extract the code from it and use it as-is (independent from Sisyphus).
For reference, we also include the scoring and config.
