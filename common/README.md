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
