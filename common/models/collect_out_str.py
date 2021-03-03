

def get_vocab_tf():
    from returnn.datasets.generating import Vocabulary
    from returnn.tf.util.basic import get_shared_vocab
    from returnn.tf.compat import v1 as tf
    dataset = get_dataset("train")
    vocab = Vocabulary.create_vocab(**dataset["bpe"])
    labels = vocab.labels  # bpe labels ("@@" at end, or not), excluding blank
    labels = [(l + " ").replace("@@ ", "") for l in labels] + [""]
    labels_t = get_shared_vocab(labels)
    return labels_t


def get_vocab_sym(i):
    """
    :param tf.Tensor i: e.g. [B], int32
    :return: same shape as input, string
    :rtype: tf.Tensor
    """
    from returnn.tf.compat import v1 as tf
    return tf.gather(params=get_vocab_tf(), indices=i)


def out_str(source, **kwargs):
    # ["prev:out_str", "output_emit", "output"]
    from returnn.tf.compat import v1 as tf
    from returnn.tf.util.basic import where_bc
    return source(0) + where_bc(source(1), get_vocab_sym(source(2)), tf.constant(""))
