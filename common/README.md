# Usage

This is intended to be used for the RETURNN `returnn.import_` mechanism.
See [here](https://github.com/rwth-i6/returnn/discussions/436)
for some initial discussion.

Usage example for config:

    from returnn.import_ import import_
    test = import_("github.com/rwth-i6/returnn-experiments", "common/test.py", "20210302-01094bef2761")
    print(test.hello())

You can also make use of auto-completion features in your editor (e.g. PyCharm).
Add `~/returnn/_pkg_import` to your Python paths,
and use this alternative code:

    from returnn.import_ import import_
    import_("github.com/rwth-i6/returnn-experiments", "common", "20210302-01094bef2761")
    from returnn_import.github_com.rwth_i6.returnn_experiments.v20210302133012_01094bef2761.common import test
    print(test.hello())

During development of a new feature in `returnn-experiments`,
you would use a special `None` placeholder for the version,
such that you can directly work in the checked out repo.
The config code looks like this:

    from returnn.import_ import import_
    import_("github.com/rwth-i6/returnn-experiments", "common", None)
    from returnn_import.github_com.rwth_i6.returnn_experiments.dev.common import test
    print(test.hello())

You would also edit the code in `~/returnn/pkg/...`,
and once finished, you would commit and push to `returnn-experiments`,
and then change the config to that specific version (date & commit).


# Code principles

These are the ideas behind the recipes.
If you want to contribute, please try to follow them.
(If something is unclear, or even in general,
better speak with someone before you do changes, or add something.)

## Simplicity

This is supposed to be **simple**.
Functions or classes can have some options
with reasonable defaults.
This should not become too complicated.
E.g. a function to return a Librispeech corpus
should not be totally generic to cover every possible case.
When it doesn't fit your use case,
instead of making the function more complicated,
just provide your alternative `LibrispeechCustomX` class.
There should be reasonable defaults.
E.g. just `Librispeech()` will give you some reasonable dataset.

E.g. maybe ~5 arguments per function is ok
(and each argument should have some good default),
but it should not be much more.
**Better just make separate functions instead
even when there is some amount of duplicate code**
(`make_transformer` which creates a standard Transformer,
vs `make_linformer` which creates a Linformer, etc.).

## Building blocks

It should be simple to use functions
as basic building blocks to build sth more complex.
E.g. when you implement the Transformer model
(put that to `models/segmental/transformer.py`)
make functions `make_trafo_enc_block`
and `make_trafo_encoder(num_layers=..., ...)`
in `models/encoder/transformer.py`,
and then `make_transformer_decoder` and `make_transformer`
in `models/segmental/transformer.py`.
That makes **parts of it easily reusable**.
**Break it down** as much as it is reasonable.

## Code dependencies

The building blocks will naturally depend on each other.
In most cases, you should use **relative imports**
to make use of other building blocks,
and **not `import_`**.

## Data dependencies

Small files (e.g. vocabularies up to a certain size <100kb or so)
could be directly put to the repository next to the Python files.
This should be kept minimal and only be used for the most common files.
(E.g. our Librispeech BPE vocab is stored.)
The repository should stay small,
so try to avoid this if this is not really needed.

For any larger files or other files,
the idea is that this can easily be used across different systems.
So there would be a common directory structure
in some directory which could be some symlinks elsewhere.
(We could also provide some scripts to simplify handling this.)
To refer to such a file path, use the functions in `data.py`.
