# LSH attention implementation

This implements locality-sensitive-hashing (LSH) attention for NMT
as presented in our paper ["Locality-Sensitive Hashing for Long Context Neural Machine Translation" submitted to IWSLT2022](https://aclanthology.org/2022.iwslt-1.4/).

Running this requires some features of [this RETURNN development branch](https://github.com/rwth-i6/returnn/tree/frithjof-self-attention),
e.g. the `CumConcatLayer` and being able to specify optional time axes by dim tag name (e.g. `stag:key-window?`).
It does not work with the current RETURNN master currently.

## Setup

```bash
git clone https://github.com/rwth-i6/returnn.git -b frithjof-self-attention
pip install --user -r returnn/requirements.txt  # install RETURNN requirements
export PYTHONPATH=$PWD/returnn/:$PYTHONPATH  # add RETURNN to your pythonpath
nosetests -v test_lsh.py  # run all tests
```

Clone our RETURNN branch into a local folder,
install the requirements and add the RETURNN installation to your Python path.

After this, run all tests using nosetests to make sure everything is working.

## Usage

We provide functions to generate the net dicts for the attention components:

 - LSH attention: `add_lsh_self_attention_layer` and `add_lsh_cross_attention_layer` in `lsh_attention.py`
 - Vanilla (full) attention: `add_vanilla_self_attention_layer` and `add_vanilla_cross_attention_layer` in `vanilla_attention.py`
 - Vanilla (full) attention but masking away hashes afterwards: `add_full_lsh_cross_attention_layer` in `lsh_attention.py` (useful as a sanity check, compare this using infinity chunk size)

E.g., to add LSH cross-attention to your config, use something like this:

```python
# your net dict, e.g. a vanilla Transformer model that you want to adapt.
net_dict = {
  'encoder': {'class': 'linear', 'n_out': 5, 'activation': None},
  # rest of your model, usually more complex encoder, etc...
  'output': {  # your decoder
    'class': 'rec', 'target': 'classes', 'max_seq_len': 'max_len_from("base:encoder") * 3',
    'from': [],
    'unit': {
      'embed': {'class': 'linear', 'activation': None, 'from': ['prev:output'], "n_out": 7},
      'lsh_att': None,  # your LSH attention. placeholder for now, will be added right after this.
      'output_prob': {'class': 'softmax', 'from': ['lsh_att'], 'target': 'classes'},
      'output': {
        'class': 'choice', 'beam_size': 4, 'target': 'classes', 'from': ['output_prob'], 'initial_output': 'zeros',
        'loss': 'ce', 'is_output_layer': True}
      # more ...
    }
  }
}
# add LSH layers to the net dict
add_lsh_cross_attention_layer(
  net_dict['output']['unit'],  # decoder net dict
  net_dict,  # complete net dict
  input='embed',  # query input
  keys_input='base:encoder',  # key input
  output='lsh',  # will create layers prefixed with "lsh_". Output layer is "lsh_att".
  num_heads=8, key_dim=64, value_dim=64,
  num_hashes=8,  # number of LSH hash classes per round
  num_rounds=1,  # number of hash rounds
  key_chunk_size=10, query_chunk_size=10,  # chunk size
  key_chunks_before=1, key_chunks_after=1,  # each query chunk will attend 
  mask_different_hashes=False,  # whether a query is allowed to attend to a key with a different hash within the its chunk
  allow_duplicate_attention=False,  # whether attention in multiple hash rounds should count twice or not
  chunk_alignment="search_bounds_centered",  # either "identity" (default for self-attention) or "search_bounds_centered" (default for cross attention)
  shuffle_kv=True  # whether to shuffle keys/values before sorting
)
```

Look at the code and in particular our tests in `test_lsh.py` for more detailed usage.

## Complete RETURNN Config Example

Here you can find a complete RETURNN config that you can use directly by swapping out your dataset paths:
[complete-returnn-example.config](https://github.com/rwth-i6/returnn-experiments/blob/master/2022-lsh-attention/complete-returnn-example.config)

## Citation

If you use our work please cite it as:

```
@inproceedings{petrick-etal-2022-locality,
    title = "Locality-Sensitive Hashing for Long Context Neural Machine Translation",
    author = "Petrick, Frithjof  and
      Rosendahl, Jan  and
      Herold, Christian  and
      Ney, Hermann",
    booktitle = "Proceedings of the 19th International Conference on Spoken Language Translation (IWSLT 2022)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland (in-person and online)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.iwslt-1.4",
    pages = "32--42"
}
```
