#!crnn/rnn.py
# kate: syntax python;
# -*- mode: python -*-
# sublime: syntax 'Packages/Python Improved/PythonImproved.tmLanguage'
# vim:set expandtab tabstop=4 fenc=utf-8 ff=unix ft=python:

# via:
# /u/irie/setups/switchboard/2018-02-13--end2end-zeyer/config-train/bpe_1k.multihead-mlp-h1.red8.enc6l.encdrop03.decbs.ls01.pretrain2.nbd07.config
# Kazuki BPE1k baseline, from Interspeech paper.
# based on rna3c-attlstmr-irielm2.nohdf.dec5la-ff.mlr50.emit.rep.ctcalign.config
# differences:
# see notes-andre.txt

import os
import numpy
from subprocess import check_output, CalledProcessError
from TFUtil import DimensionTag

# task
use_tensorflow = True
task = config.value("task", "train")
device = "gpu"
multiprocessing = True
update_on_device = True

debug_mode = False
if int(os.environ.get("DEBUG", "0")):
    # print("** DEBUG MODE")
    debug_mode = True

if config.has("beam_size"):
    beam_size = config.int("beam_size", 0)
    print("** beam_size %i" % beam_size)
else:
    if task == "train":
        beam_size = 4
    else:
        beam_size = 12

_cf_cache = {}

def cf(filename):
    """Cache manager"""
    if filename in _cf_cache:
        return _cf_cache[filename]
    if debug_mode or check_output(["hostname"]).strip().decode("utf8") in ["cluster-cn-211", "sulfid"]:
        print("use local file: %s" % filename)
        return filename  # for debugging
    try:
        cached_fn = check_output(["cf", filename]).strip().decode("utf8")
    except CalledProcessError:
        print("Cache manager: Error occured, using local file")
        return filename
    assert os.path.exists(cached_fn)
    _cf_cache[filename] = cached_fn
    return cached_fn

# data
target = "bpe"
target_num_labels = 1030
targetb_num_labels = target_num_labels + 1  # with blank
targetb_blank_idx = 0
time_tag = DimensionTag(kind=DimensionTag.Types.Spatial, description="time")
output_len_tag = DimensionTag(kind=DimensionTag.Types.Spatial, description="output-len")  # it's downsampled time
# use "same_dim_tags_as": {"t": time_tag} if same time tag ("data" and "alignment"). e.g. for RNA. not for RNN-T.
extern_data = {
    "data": {"dim": 40, "same_dim_tags_as": {"t": time_tag}},  # Gammatone 40-dim
    target: {"dim": target_num_labels, "sparse": True},  # see vocab
    "alignment": {"dim": targetb_num_labels, "sparse": True, "same_dim_tags_as": {"t": output_len_tag}},
    "align_score": {"shape": (1,), "dtype": "float32"},
}
if task != "train":
    # During train, we add this via the network (from prev alignment, or linear seg). Otherwise it's not available.
    extern_data["targetb"] = {"dim": targetb_num_labels, "sparse": True, "available_for_inference": False}
EpochSplit = 6

def get_sprint_dataset(data, hdf_files=None):
    assert data in {"train", "devtrain", "cv", "dev", "hub5e_01", "rt03s"}
    epoch_split = {"train": EpochSplit}.get(data, 1)
    corpus_name = {"cv": "train", "devtrain": "train"}.get(data, data)  # train, dev, hub5e_01, rt03s

    # see /u/tuske/work/ASR/switchboard/corpus/readme
    # and zoltans mail https://mail.google.com/mail/u/0/#inbox/152891802cbb2b40
    files = {}
    files["config"] = "config/training.config"
    files["corpus"] = "/work/asr3/irie/data/switchboard/corpora/%s.corpus.gz" % corpus_name
    if data in {"train", "cv", "devtrain"}:
        files["segments"] = "dependencies/seg_%s" % {"train":"train", "cv":"cv_head3000", "devtrain": "train_head3000"}[data]
    files["features"] = "/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.%s.bundle" % corpus_name
    for k, v in sorted(files.items()):
        assert os.path.exists(v), "%s %r does not exist" % (k, v)
    estimated_num_seqs = {"train": 227047, "cv": 3000, "devtrain": 3000}  # wc -l segment-file

    args = [
        "--config=" + files["config"],
        lambda: "--*.corpus.file=" + cf(files["corpus"]),
        lambda: "--*.corpus.segments.file=" + (cf(files["segments"]) if "segments" in files else ""),
        lambda: "--*.feature-cache-path=" + cf(files["features"]),
        "--*.log-channel.file=/dev/null",
        "--*.window-size=1",
    ]
    if not hdf_files:
        args += [
            "--*.corpus.segment-order-shuffle=true",
            "--*.segment-order-sort-by-time-length=true",
            "--*.segment-order-sort-by-time-length-chunk-size=%i" % {"train": epoch_split * 1000}.get(data, -1),
        ]
    d = {
        "class": "ExternSprintDataset", "sprintTrainerExecPath": "sprint-executables/nn-trainer",
        "sprintConfigStr": args,
        "suppress_load_seqs_print": True,  # less verbose
    }
    d.update(sprint_interface_dataset_opts)
    partition_epochs_opts = {
        "partition_epoch": epoch_split,
        "estimated_num_seqs": (estimated_num_seqs[data] // epoch_split) if data in estimated_num_seqs else None,
    }
    d.update(partition_epochs_opts)
    return d

sprint_interface_dataset_opts = {
    "input_stddev": 3.,
    "bpe": {
        'bpe_file': '/work/asr3/irie/data/switchboard/subword_clean/ready/swbd_clean.bpe_code_1k',
        'vocab_file': '/work/asr3/irie/data/switchboard/subword_clean/ready/vocab.swbd_clean.bpe_code_1k',
        # 'seq_postfix': [0]  # no EOS needed for RNN-T
    }}

train = get_sprint_dataset("train")
dev = get_sprint_dataset("cv")
eval_datasets = {"devtrain": get_sprint_dataset("devtrain")}
cache_size = "0"
window = 1


# Note: We control the warmup in the pretrain construction.
learning_rate = 0.001
min_learning_rate = learning_rate / 50.


def summary(name, x):
    """
    :param str name:
    :param tf.Tensor x: (batch,time,feature)
    """
    import tensorflow as tf
    # tf.summary.image wants [batch_size, height,  width, channels],
    # we have (batch, time, feature).
    img = tf.expand_dims(x, axis=3)  # (batch,time,feature,1)
    img = tf.transpose(img, [0, 2, 1, 3])  # (batch,feature,time,1)
    tf.summary.image(name, img, max_outputs=10)
    tf.summary.scalar("%s_max_abs" % name, tf.reduce_max(tf.abs(x)))
    mean = tf.reduce_mean(x)
    tf.summary.scalar("%s_mean" % name, mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(x - mean)))
    tf.summary.scalar("%s_stddev" % name, stddev)
    tf.summary.histogram("%s_hist" % name, tf.reduce_max(tf.abs(x), axis=2))


def _mask(x, batch_axis, axis, pos, max_amount):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param tf.Tensor pos: (batch,)
    :param int|tf.Tensor max_amount: inclusive
    """
    import tensorflow as tf
    ndim = x.get_shape().ndims
    n_batch = tf.shape(x)[batch_axis]
    dim = tf.shape(x)[axis]
    amount = tf.random_uniform(shape=(n_batch,), minval=1, maxval=max_amount + 1, dtype=tf.int32)
    pos2 = tf.minimum(pos + amount, dim)
    idxs = tf.expand_dims(tf.range(0, dim), 0)  # (1,dim)
    pos_bc = tf.expand_dims(pos, 1)  # (batch,1)
    pos2_bc = tf.expand_dims(pos2, 1)  # (batch,1)
    cond = tf.logical_and(tf.greater_equal(idxs, pos_bc), tf.less(idxs, pos2_bc))  # (batch,dim)
    if batch_axis > axis:
        cond = tf.transpose(cond)  # (dim,batch)
    cond = tf.reshape(cond, [tf.shape(x)[i] if i in (batch_axis, axis) else 1 for i in range(ndim)])
    from TFUtil import where_bc
    x = where_bc(cond, 0.0, x)
    return x


def random_mask(x, batch_axis, axis, min_num, max_num, max_dims):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param int|tf.Tensor min_num:
    :param int|tf.Tensor max_num: inclusive
    :param int|tf.Tensor max_dims: inclusive
    """
    import tensorflow as tf
    n_batch = tf.shape(x)[batch_axis]
    if isinstance(min_num, int) and isinstance(max_num, int) and min_num == max_num:
        num = min_num
    else:
        num = tf.random_uniform(shape=(n_batch,), minval=min_num, maxval=max_num + 1, dtype=tf.int32)
    # https://github.com/tensorflow/tensorflow/issues/9260
    # https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    z = -tf.log(-tf.log(tf.random_uniform((n_batch, tf.shape(x)[axis]), 0, 1)))
    _, indices = tf.nn.top_k(z, num if isinstance(num, int) else tf.reduce_max(num))
    # indices should be sorted, and of shape (batch,num), entries (int32) in [0,dim)
    # indices = tf.Print(indices, ["indices", indices, tf.shape(indices)])
    if isinstance(num, int):
        for i in range(num):
            x = _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims)
    else:
        _, x = tf.while_loop(
            cond=lambda i, _: tf.less(i, tf.reduce_max(num)),
            body=lambda i, x: (
                i + 1,
                tf.where(
                    tf.less(i, num),
                    _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims),
                    x)),
            loop_vars=(0, x))
    return x


def transform(data, network, time_factor=1):
    x = data.placeholder
    import tensorflow as tf
    # summary("features", x)
    step = network.global_train_step
    step1 = tf.where(tf.greater_equal(step, 1000), 1, 0)
    step2 = tf.where(tf.greater_equal(step, 2000), 1, 0)
    def get_masked():
        x_masked = x
        x_masked = random_mask(
          x_masked, batch_axis=data.batch_dim_axis, axis=data.time_dim_axis,
          min_num=step1 + step2, max_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // 100, 2) * (1 + step1 + step2 * 2),
          max_dims=20 // time_factor)
        x_masked = random_mask(
          x_masked, batch_axis=data.batch_dim_axis, axis=data.feature_dim_axis,
          min_num=step1 + step2, max_num=2 + step1 + step2 * 2,
          max_dims=data.dim // 5)
        #summary("features_mask", x_masked)
        return x_masked
    x = network.cond_on_train(get_masked, lambda: x)
    return x


def targetb_recomb_train(layer, batch_dim, scores_in, scores_base, base_beam_in, end_flags, **kwargs):
    """
    :param ChoiceLayer layer:
    :param tf.Tensor batch_dim: scalar
    :param tf.Tensor scores_base: (batch,base_beam_in,1). existing beam scores
    :param tf.Tensor scores_in: (batch,base_beam_in,dim). log prob frame distribution
    :param tf.Tensor end_flags: (batch,base_beam_in)
    :param tf.Tensor base_beam_in: int32 scalar, 1 or prev beam size
    :rtype: tf.Tensor
    :return: (batch,base_beam_in,dim), combined scores
    """
    import tensorflow as tf
    from TFUtil import where_bc, nd_indices, tile_transposed
    scores = scores_in + scores_base  # (batch,beam,dim)
    dim = layer.output.dim
    
    u = layer.explicit_search_sources[0].output  # prev:u actually. [B*beam], pos in target [0..decT-1]
    assert u.shape == ()
    u_t = tf.reshape(tf.reshape(u.placeholder, (batch_dim, -1))[:,:base_beam_in], (-1,))  # u beam might differ from base_beam_in
    targets = layer.network.parent_net.extern_data.data[target]  # BPE targets, [B,decT]
    assert targets.shape == (None,) and targets.is_batch_major
    target_lens = targets.get_sequence_lengths()  # [B]
    target_lens_exp = tile_transposed(target_lens, axis=0, multiples=base_beam_in)  # [B*beam]
    missing_targets = target_lens_exp - u_t  # [B*beam]
    allow_target = tf.greater(missing_targets, 0)  # [B*beam]
    targets_exp = tile_transposed(targets.placeholder, axis=0, multiples=base_beam_in)  # [B*beam,decT]
    targets_u = tf.gather_nd(targets_exp, indices=nd_indices(where_bc(allow_target, u_t, 0)))  # [B*beam]
    targets_u = tf.reshape(targets_u, (batch_dim, base_beam_in))  # (batch,beam)
    allow_target = tf.reshape(allow_target, (batch_dim, base_beam_in))  # (batch,beam)
    
    #t = layer.explicit_search_sources[1].output  # prev:t actually. [B*beam], pos in encoder [0..encT-1]
    #assert t.shape == ()
    #t_t = tf.reshape(tf.reshape(t.placeholder, (batch_dim, -1))[:,:base_beam_in], (-1,))  # t beam might differ from base_beam_in
    t_t = layer.network.get_rec_step_index() - 1  # scalar
    inputs = layer.network.parent_net.get_layer("encoder").output  # encoder, [B,encT]
    input_lens = inputs.get_sequence_lengths()  # [B]
    input_lens_exp = tile_transposed(input_lens, axis=0, multiples=base_beam_in)  # [B*beam]
    allow_blank = tf.less(missing_targets, input_lens_exp - t_t)  # [B*beam]
    allow_blank = tf.reshape(allow_blank, (batch_dim, base_beam_in))  # (batch,beam)

    dim_idxs = tf.range(dim)[None,None,:]  # (1,1,dim)
    masked_scores = where_bc(
        tf.logical_or(
            tf.logical_and(tf.equal(dim_idxs, targetb_blank_idx), allow_blank[:,:,None]),
            tf.logical_and(tf.equal(dim_idxs, targets_u[:,:,None]), allow_target[:,:,None])),
        scores, float("-inf"))

    return where_bc(end_flags[:,:,None], scores, masked_scores)


def get_vocab_tf():
    from GeneratingDataset import Vocabulary
    import TFUtil
    import tensorflow as tf
    vocab = Vocabulary.create_vocab(**sprint_interface_dataset_opts["bpe"])
    labels = vocab.labels  # bpe labels ("@@" at end, or not), excluding blank
    labels = [(l + " ").replace("@@ ", "") for l in labels] + [""]
    labels_t = TFUtil.get_shared_vocab(labels)
    return labels_t


def get_vocab_sym(i):
    """
    :param tf.Tensor i: e.g. [B], int32
    :return: same shape as input, string
    :rtype: tf.Tensor
    """
    import tensorflow as tf
    return tf.gather(params=get_vocab_tf(), indices=i)


def out_str(source, **kwargs):
    # ["prev:out_str", "output_emit", "output"]
    import tensorflow as tf
    from TFUtil import where_bc
    return source(0) + where_bc(source(1), get_vocab_sym(source(2)), tf.constant(""))


def get_filtered_score_op(verbose=False):
    cpp_code = """
    #include "tensorflow/core/framework/op.h"
    #include "tensorflow/core/framework/op_kernel.h"
    #include "tensorflow/core/framework/shape_inference.h"
    #include "tensorflow/core/framework/resource_mgr.h"
    #include "tensorflow/core/framework/resource_op_kernel.h"
    #include "tensorflow/core/framework/tensor.h"
    #include "tensorflow/core/platform/macros.h"
    #include "tensorflow/core/platform/mutex.h"
    #include "tensorflow/core/platform/types.h"
    #include "tensorflow/core/public/version.h"
    #include <cmath>
    #include <map>
    #include <set>
    #include <string>
    #include <tuple>

    using namespace tensorflow;

    REGISTER_OP("GetFilteredScore")
    .Input("prev_str: string")
    .Input("scores: float32")
    .Input("labels: string")
    .Output("new_scores: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));
        return Status::OK();
    });

    class GetFilteredScoreOp : public OpKernel {
    public:
    using OpKernel::OpKernel;
    void Compute(OpKernelContext* context) override {
        const Tensor* prev_str = &context->input(0);
        const Tensor* scores = &context->input(1);
        const Tensor* labels = &context->input(2);

        int n_batch = prev_str->shape().dim_size(0);
        int n_beam = prev_str->shape().dim_size(1);

        Tensor* ret;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({n_batch, n_beam}), &ret));
        for(int bat = 0; bat < n_batch; ++bat)
            for(int hyp = 0; hyp < n_beam; ++hyp)
                ret->tensor<float, 2>()(bat, hyp) = scores->tensor<float, 2>()(bat, hyp);
        
        for(int bat = 0; bat < n_batch; ++bat) {
            std::map<std::string, std::set<int> > new_hyps;  // seq -> set of hyp idx

            for(int hyp = 0; hyp < n_beam; ++hyp) {
                auto& seq_set = new_hyps[prev_str->tensor<string, 2>()(bat, hyp)];
                seq_set.insert(hyp);
            }

            for(const auto& items : new_hyps) {
                if(std::get<1>(items).size() > 1) {
                    float best_score = 0.;
                    int best_idx = -1;
                    for(int idx : std::get<1>(items)) {
                        float score = scores->tensor<float, 2>()(bat, idx);
                        if(score > best_score || best_idx == -1) {
                            best_score = score;
                            best_idx = idx;
                        }
                    }

                    float sum_score = 0.;
                    for(int idx : std::get<1>(items)) {
                        float score = scores->tensor<float, 2>()(bat, idx);
                        sum_score += expf(score - best_score);
                    }
                    sum_score = logf(sum_score) + best_score;

                    for(int idx : std::get<1>(items)) {
                        if(idx != best_idx)
                            ret->tensor<float, 2>()(bat, idx) = -std::numeric_limits<float>::infinity();
                        else
                            ret->tensor<float, 2>()(bat, idx) = sum_score;
                    }
                }
            }
        }
    }
    };
    REGISTER_KERNEL_BUILDER(Name("GetFilteredScore").Device(DEVICE_CPU), GetFilteredScoreOp);
    """
    from TFUtil import OpCodeCompiler
    compiler = OpCodeCompiler(
        base_name="GetFilteredScore", code_version=1, code=cpp_code,
        is_cpp=True, use_cuda_if_available=False, verbose=verbose)
    tf_mod = compiler.load_tf_module()
    return tf_mod.get_filtered_score


def get_filtered_score_cpp(prev_str, scores, labels):
    """
    :param tf.Tensor prev_str: (batch,beam)
    :param tf.Tensor scores: (batch,beam)
    :param list[bytes] labels: len (dim)
    :return: scores with logsumexp at best, others -inf, (batch,beam)
    :rtype: tf.Tensor
    """
    import TFUtil
    import tensorflow as tf
    labels_t = TFUtil.get_shared_vocab(labels)
    with tf.device("cpu:0"):
        return get_filtered_score_op()(prev_str, scores, labels_t)


def targetb_recomb_recog(layer, batch_dim, scores_in, scores_base, base_beam_in, end_flags, **kwargs):
    """
    :param ChoiceLayer layer:
    :param tf.Tensor batch_dim: scalar
    :param tf.Tensor scores_base: (batch,base_beam_in,1). existing beam scores
    :param tf.Tensor scores_in: (batch,base_beam_in,dim). log prob frame distribution
    :param tf.Tensor end_flags: (batch,base_beam_in)
    :param tf.Tensor base_beam_in: int32 scalar, 1 or prev beam size
    :rtype: tf.Tensor
    :return: (batch,base_beam_in,dim), combined scores
    """
    import tensorflow as tf
    from TFUtil import where_bc, nd_indices, tile_transposed

    dim = layer.output.dim
    
    prev_str = layer.explicit_search_sources[0].output  # [B*beam], str
    prev_str_t = tf.reshape(prev_str.placeholder, (batch_dim, -1))[:,:base_beam_in]
    prev_out = layer.explicit_search_sources[1].output  # [B*beam], int32
    prev_out_t = tf.reshape(prev_out.placeholder, (batch_dim, -1))[:,:base_beam_in]

    from GeneratingDataset import Vocabulary
    import TFUtil
    import tensorflow as tf
    vocab = Vocabulary.create_vocab(**sprint_interface_dataset_opts["bpe"])
    labels = vocab.labels  # bpe labels ("@@" at end, or not), excluding blank
    labels = [(l + " ").replace("@@ ", "").encode("utf8") for l in labels] + [b""]

    # Pre-filter approx (should be much faster), sum approx (better).
    scores_base = tf.reshape(get_filtered_score_cpp(prev_str_t, tf.reshape(scores_base, (batch_dim, base_beam_in)), labels), (batch_dim, base_beam_in, 1))
    
    scores = scores_in + scores_base  # (batch,beam,dim)

    # Mask -> max approx, in all possible options, slow.
    #mask = get_score_mask_cpp(prev_str_t, prev_out_t, scores, labels)
    #masked_scores = where_bc(mask, scores, float("-inf"))
    # Sum approx in all possible options, slow.
    #masked_scores = get_new_score_cpp(prev_str_t, prev_out_t, scores, labels)
    
    #scores = where_bc(end_flags[:,:,None], scores, masked_scores)
    
    return scores


def rna_loss(source, **kwargs):
    """
    Computes the RNA loss function.

    :param log_prob:
    :return:
    """
    # acts: (B, T, U, V)
    # targets: (B, U-1)
    # input_lengths (B,)
    # label_lengths (B,)
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
    import tensorflow as tf
    log_probs = source(0, as_data=True, auto_convert=False)
    targets = source(1, as_data=True, auto_convert=False)
    encoder = source(2, as_data=True, auto_convert=False)

    enc_lens = encoder.get_sequence_lengths()
    dec_lens = targets.get_sequence_lengths()

    from rna_tf_impl import tf_forward_shifted_rna
    costs = -tf_forward_shifted_rna(log_probs.get_placeholder_as_batch_major(), targets.get_placeholder_as_batch_major(), enc_lens, dec_lens, blank_index=targetb_blank_idx, debug=False)
    costs = tf.where(tf.is_finite(costs), costs, tf.zeros_like(costs))
    return costs

def rna_alignment(source, **kwargs):
    """
    Computes the RNA loss function.

    :param log_prob:
    :return:
    """
    # acts: (B, T, U, V)
    # targets: (B, U-1)
    # input_lengths (B,)
    # label_lengths (B,)
    import sys
    import TFUtil
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
    import tensorflow as tf
    log_probs = source(0, as_data=True, auto_convert=False).get_placeholder_as_batch_major()
    targets = source(1, as_data=True, auto_convert=False)
    encoder = source(2, as_data=True, auto_convert=False)

    enc_lens = encoder.get_sequence_lengths()
    dec_lens = targets.get_sequence_lengths()

    # target_len = TFUtil.get_shape_dim(targets.get_placeholder_as_batch_major(), 1)
    # log_probs = TFUtil.check_input_dim(log_probs, 2, target_len+1)
    # enc_lens = tf.Print(enc_lens, ["enc_lens:", enc_lens,
        # "dec_lens:", dec_lens,
        # "targets:", tf.shape(targets.get_placeholder_as_batch_major()), "log-probs:", tf.shape(log_probs.get_placeholder_as_batch_major())], summarize=-1)

    from rna_tf_impl import tf_forward_shifted_rna
    costs, alignment = tf_forward_shifted_rna(log_probs, targets.get_placeholder_as_batch_major(), enc_lens, dec_lens,
        blank_index=targetb_blank_idx, debug=False, with_alignment=True)
    return alignment # (B, T)


def rna_alignment_out(sources, **kwargs):
    from TFUtil import Data

    log_probs = sources[0].output
    targets = sources[1].output
    encoder = sources[2].output
    enc_lens = encoder.get_sequence_lengths()
    return Data(name="rna_alignment", sparse=True, dim=targetb_num_labels, size_placeholder={0: enc_lens})



def rna_loss_out(sources, **kwargs):
    from TFUtil import Data
    return Data(name="rna_loss", shape=())

#import_model_train_epoch1 = "base/data-train/base2.conv2l.specaug4a/net-model/network.160"
#_train_setup_dir = "data-train/base2.conv2l.specaug4a"
#model = _train_setup_dir + "/net-model/network"
preload_from_files = {
  #"base": {
  #  "init_for_train": True,
  #  "ignore_missing": True,
  #  "filename": "/u/zeyer/setups/switchboard/2018-10-02--e2e-bpe1k/data-train/base2.conv2l.specaug4a/net-model/network.160",
  #},
  #"encoder": {
  #  "init_for_train": True,
  #  "ignore_missing": True,
  #  "filename": "/u/zeyer/setups/switchboard/2017-12-11--returnn/data-train/#dropout01.l2_1e_2.6l.n500.inpstddev3.fl2.max_seqs100.grad_noise03.nadam.lr05e_3.nbm6.nbrl.grad_clip_inf.nbm3.run1/net-model/network.077",
  #},
  # "encoder": {
    # "init_for_train": True,
    # "ignore_missing": True,
    # "ignore_params_prefixes": {"output/"},
    # "filename": "/u/zeyer/setups/switchboard/2019-10-22--e2e-bpe1k/data-train/base2.conv2l.specaug4a.ctc.devtrain/net-model/network.150",
  # }
}
# lm_model_filename = "/work/asr3/irie/experiments/lm/switchboard/2018-01-23--lmbpe-zeyer/data-train/bpe1k_clean_i256_m2048_m2048.sgd_b16_lr0_cl2.newbobabs.d0.2/net-model/network.023"


def get_net_dict(pretrain_idx):
    """
    :param int|None pretrain_idx: starts at 0. note that this has a default repetition factor of 6
    :return: net_dict or None if pretrain should stop
    :rtype: dict[str,dict[str]|int]|None
    """
    # Note: epoch0 is 0-based here! I.e. in contrast to elsewhere, where it is 1-based.
    # Also, we never use #repetition here, such that this is correct.
    # This is important because of sub-epochs and storing the HDF files,
    # to know exactly which HDF files cover the dataset completely.
    epoch0 = pretrain_idx
    net_dict = {}

    # network
    # (also defined by num_inputs & num_outputs)
    EncKeyTotalDim = 200
    AttNumHeads = 1  # must be 1 for hard-att
    AttentionDropout = 0.1
    l2 = 0.0001
    EncKeyPerHeadDim = EncKeyTotalDim // AttNumHeads
    EncValueTotalDim = 2048
    EncValuePerHeadDim = EncValueTotalDim // AttNumHeads
    LstmDim = EncValueTotalDim // 2

    if pretrain_idx is not None:
        net_dict["#config"] = {}

        # Do this in the very beginning.
        #lr_warmup = [0.0] * EpochSplit  # first collect alignments with existing model, no training
        lr_warmup = list(numpy.linspace(learning_rate * 0.1, learning_rate, num=10))
        if pretrain_idx < len(lr_warmup):
            net_dict["#config"]["learning_rate"] = lr_warmup[pretrain_idx]


    # We import the model, thus no growing.
    start_num_lstm_layers = 2
    final_num_lstm_layers = 6
    num_lstm_layers = final_num_lstm_layers
    if pretrain_idx is not None:
        pretrain_idx = max(pretrain_idx, 0) // 5  # Repeat a bit.
        num_lstm_layers = pretrain_idx + start_num_lstm_layers
        pretrain_idx = num_lstm_layers - final_num_lstm_layers
        num_lstm_layers = min(num_lstm_layers, final_num_lstm_layers)

    if final_num_lstm_layers > start_num_lstm_layers:
        start_dim_factor = 0.5
        grow_frac = 1.0 - float(final_num_lstm_layers - num_lstm_layers) / (final_num_lstm_layers - start_num_lstm_layers)
        dim_frac = start_dim_factor + (1.0 - start_dim_factor) * grow_frac
    else:
        dim_frac = 1.

    time_reduction = [3, 2] if num_lstm_layers >= 3 else [6]

    if pretrain_idx is not None and pretrain_idx <= 1 and "learning_rate" not in net_dict["#config"]:
        # Fixed learning rate for the beginning.
        net_dict["#config"]["learning_rate"] = learning_rate

    net_dict["#info"] = {
        "epoch0": epoch0,  # Set this here such that a new construction for every pretrain idx is enforced in all cases.
        "num_lstm_layers": num_lstm_layers,
        "dim_frac": dim_frac,
    }

    # We use this pretrain construction during the whole training time (epoch0 > num_epochs).
    if pretrain_idx is not None and epoch0 % EpochSplit == 0 and epoch0 > num_epochs:
        # Stop pretraining now.
        return None

    net_dict.update({
        "source": {"class": "eval", "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)"},
        "source0": {"class": "split_dims", "axis": "F", "dims": (-1, 1), "from": "source"},  # (T,40,1)

        # Lingvo: ep.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)],  ep.conv_filter_strides = [(2, 2), (2, 2)]
        "conv0": {"class": "conv", "from": "source0", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None, "with_bias": True, "trainable": True},  # (T,40,32)
        "conv0p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv0"},  # (T,20,32)
        "conv1": {"class": "conv", "from": "conv0p", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None, "with_bias": True, "trainable": True},  # (T,20,32)
        "conv1p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv1"},  # (T,10,32)
        "conv_merged": {"class": "merge_dims", "from": "conv1p", "axes": "static"},  # (T,320)

        # Encoder LSTMs added below, resulting in "encoder0".

        "encoder": {"class": "copy", "from": "encoder0"},        
        "enc_ctx0": {"class": "linear", "from": "encoder", "activation": None, "with_bias": False, "n_out": EncKeyTotalDim, "L2": l2, "dropout": 0.2},
        "enc_ctx_win": {"class": "window", "from": "enc_ctx0", "window_size": 5},  # [B,T,W,D]
        "enc_val": {"class": "copy", "from": "encoder"},
        "enc_val_win": {"class": "window", "from": "enc_val", "window_size": 5},  # [B,T,W,D]

        "enc_ctx": {"class": "linear", "from": "encoder", "activation": "tanh", "n_out": EncKeyTotalDim, "L2": l2, "dropout": 0.2},

        "enc_seq_len": {"class": "length", "from": "encoder", "sparse": True},

        # for task "search" / search_output_layer
        "output_wo_b0": {
          "class": "masked_computation", "unit": {"class": "copy"},
          "from": "output", "mask": "output/output_emit"},
        "output_wo_b": {"class": "reinterpret_data", "from": "output_wo_b0", "set_sparse_dim": target_num_labels},
        "decision": {
            "class": "decide", "from": "output_wo_b", "loss": "edit_distance", "target": target,
            'only_on_search': True},
    })

    # Add encoder BLSTM stack.
    src = "conv_merged"
    if num_lstm_layers >= 1:
        net_dict.update({
            "lstm0_fw": {"class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": 1, "from": src, "trainable": True},
            "lstm0_bw": {"class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": -1, "from": src, "trainable": True}})
        src = ["lstm0_fw", "lstm0_bw"]
    for i in range(1, num_lstm_layers):
        red = time_reduction[i - 1] if (i - 1) < len(time_reduction) else 1
        net_dict.update({
            "lstm%i_pool" % (i - 1): {"class": "pool", "mode": "max", "padding": "same", "pool_size": (red,), "from": src}})
        src = "lstm%i_pool" % (i - 1)
        net_dict.update({
            "lstm%i_fw" % i: {"class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": 1, "from": src, "dropout": 0.3 * dim_frac, "trainable": True},
            "lstm%i_bw" % i: {"class": "rec", "unit": "nativelstm2", "n_out": int(LstmDim * dim_frac), "L2": l2, "direction": -1, "from": src, "dropout": 0.3 * dim_frac, "trainable": True}})
        src = ["lstm%i_fw" % i, "lstm%i_bw" % i]
    net_dict["encoder0"] = {"class": "copy", "from": src}  # dim: EncValueTotalDim

    # This is used for training.
    net_dict["lm_input0"] = {"class": "copy", "from": "data:%s" % target}
    net_dict["lm_input1"] = {"class": "prefix_in_time", "from": "lm_input0", "prefix": targetb_blank_idx}
    net_dict["lm_input"] = {"class": "copy", "from": "lm_input1"}

    def get_output_dict(train, search, targetb, beam_size=beam_size):
        return {
        "class": "rec",
        "from": "encoder",  # time-sync
        "include_eos": True,
        "back_prop": (task == "train") and train,
        "unit": {
            "am": {"class": "copy", "from": "data:source"},

            "enc_ctx_win": {"class": "gather_nd", "from": "base:enc_ctx_win", "position": ":i"},  # [B,W,D]
            "enc_val_win": {"class": "gather_nd", "from": "base:enc_val_win", "position": ":i"},  # [B,W,D]
            "att_query": {"class": "linear", "from": "am", "activation": None, "with_bias": False, "n_out": EncKeyTotalDim},
            'att_energy': {"class": "dot", "red1": "f", "red2": "f", "var1": "static:0", "var2": None,
                           "from": ['enc_ctx_win', 'att_query']},  # (B, W)
            'att_weights0': {"class": "softmax_over_spatial", "axis": "static:0", "from": 'att_energy',
                            "energy_factor": EncKeyPerHeadDim ** -0.5},  # (B, W)
            'att_weights1': {"class": "dropout", "dropout_noise_shape": {"*": None},
                             "from": 'att_weights0', "dropout": AttentionDropout},
            "att_weights": {"class": "merge_dims", "from": "att_weights1", "axes": "except_time"},
            'att': {"class": "dot", "from": ['att_weights', 'enc_val_win'],
                    "red1": "static:0", "red2": "static:0", "var1": None, "var2": "f"},  # (B, V)

            
            "prev_out_non_blank": {
                "class": "reinterpret_data", "from": "prev:output", "set_sparse_dim": target_num_labels},
                # "class": "reinterpret_data", "from": "prev:output_wo_b", "set_sparse_dim": target_num_labels},  # [B,]
            "lm_masked": {"class": "masked_computation",
                "mask": "prev:output_emit",
                "from": "prev_out_non_blank",  # in decoding
                "masked_from": "base:lm_input" if task == "train" else None,  # enables optimization if used

                "unit": {
                "class": "subnetwork", "from": "data", "trainable": True,
                "subnetwork": {
                    "input_embed": {"class": "linear", "n_out": 256, "activation": "identity", "trainable": True, "L2": l2, "from": "data"},
                    "lstm0": {"class": "rec", "unit": "nativelstm2", "dropout": 0.2, "n_out": 1024, "L2": l2, "from": "input_embed", "trainable": True},
                    "output": {"class": "copy", "from": "lstm0"}
                    #"output": {"class": "linear", "from": "lstm1", "activation": "softmax", "dropout": 0.2, "n_out": target_num_labels, "trainable": False}
                }}},
            # "lm_embed_masked": {"class": "linear", "activation": None, "n_out": 256, "from": "lm_masked"},
            #"lm_unmask": {"class": "unmask", "from": "lm_masked", "mask": "prev:output_emit"},
            # "lm_embed_unmask": {"class": "unmask", "from": "lm_embed_masked", "mask": "prev:output_emit"},
            "lm": {"class": "copy", "from": "lm_embed_unmask"},  # [B,L]


            # joint network: (W_enc h_{enc,t} + W_pred * h_{pred,u} + b)
            # train : (T-enc, B, F|2048) ; (U+1, B, F|256)
            # search: (B, F|2048) ; (B, F|256)
            "readout_in": {"class": "linear", "from": ["am", "att", "lm_masked"], "activation": None, "n_out": 1000, "L2": l2, "dropout": 0.2,
            "out_type": {"batch_dim_axis": 2 if task == "train" else 0, "shape": (None, None, 1000) if task == "train" else (1000,),
            "time_dim_axis": 0 if task == "train" else None}}, # (T, U+1, B, 1000)

            "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": "readout_in"},
              
            "label_log_prob": {
                "class": "linear", "from": "readout", "activation": "log_softmax", "dropout": 0.3, "n_out": target_num_labels},  # (B, T, U+1, 1030)
            "emit_prob0": {"class": "linear", "from": "readout", "activation": None, "n_out": 1, "is_output_layer": True},  # (B, T, U+1, 1)
            "emit_log_prob": {"class": "activation", "from": "emit_prob0", "activation": "log_sigmoid"},  # (B, T, U+1, 1)
            "blank_log_prob": {"class": "eval", "from": "emit_prob0", "eval": "tf.log_sigmoid(-source(0))"},  # (B, T, U+1, 1)
            "label_emit_log_prob": {"class": "combine", "kind": "add", "from": ["label_log_prob", "emit_log_prob"]},  # (B, T, U+1, 1), scaling factor in log-space
            "output_log_prob": {"class": "copy", "from": ["blank_log_prob", "label_emit_log_prob"]},  # (B, T, U+1, 1031)

            "output_prob": {
                "class": "eval", "from": ["output_log_prob", "base:data:" + target, "base:encoder"], "eval": rna_loss,
                "out_type": rna_loss_out, "loss": "as_is",
            },



            # this only works when the loop has been optimized, i.e. log-probs are (B, T, U, V)
            "rna_alignment" : {"class": "eval", "from": ["output_log_prob", "base:data:"+target, "base:encoder"],
                    "eval": rna_alignment, "out_type": rna_alignment_out, "is_output_layer": True} if task == "train"  # (B, T)
            else {"class": "copy", "from": "output_log_prob"},

            # During training   : targetb = "target"  (RNA-loss)
            # During recognition: targetb = "targetb"
            'output': {
                'class': 'choice', 'target': targetb, 'beam_size': beam_size,
                'from': "output_log_prob", "input_type": "log_prob",
                "initial_output": 0,
                "cheating": "exclusive" if task == "train" else None,
                #"explicit_search_sources": ["prev:u"] if task == "train" else None,
                #"custom_score_combine": targetb_recomb_train if task == "train" else None
                "explicit_search_sources": ["prev:out_str", "prev:output"] if task == "search" else None,
                "custom_score_combine": targetb_recomb_recog if task == "search" else None
                },

            "out_str": {
                "class": "eval", "from": ["prev:out_str", "output_emit", "output"],
                "initial_output": None, "out_type": {"shape": (), "dtype": "string"},
                "eval": out_str},

            "output_is_not_blank": {"class": "compare", "from": "output", "value": targetb_blank_idx, "kind": "not_equal", "initial_output": True},
            
            # initial state=True so that we are consistent to the training and the initial state is correctly set.
            "output_emit": {"class": "copy", "from": "output_is_not_blank", "initial_output": True, "is_output_layer": True},

            "const0": {"class": "constant", "value": 0, "collocate_with": ["du", "dt"]},
            "const1": {"class": "constant", "value": 1, "collocate_with": ["du", "dt"]},
            
            # pos in target, [B]
            # "du": {"class": "switch", "condition": "output_emit", "true_from": "const1", "false_from": "const0"},
            # "u": {"class": "combine", "from": ["prev:u", "du"], "kind": "add", "initial_output": 0},

            # pos in input, [B]
            # RNA is time-sync, so we always advance t
            # output label: stay in t, otherwise advance t (encoder)
            # "dt": {"class": "switch", "condition": "output_is_not_blank", "true_from": "const0", "false_from": "const1"},
            # "t": {"class": "combine", "from": ["prev:t", "dt"], "kind": "add", "initial_output": 0},

            # stop at U+T
            # in recog: stop when all input has been consumed
            # in train: defined by target.
            # "end": {"class": "compare", "from": ["t", "base:enc_seq_len"], "kind": "greater_equal"},
            },
            # "target": targetb,
            # "size_target": targetb if task == "train" else None,
            # "max_seq_len": "max_len_from('base:encoder') * 2"  # actually N+T
        }

    if task == "train":
        net_dict["output"] = get_output_dict(train=True, search=False, targetb=target)
    else:
        net_dict["output"] = get_output_dict(train=True, search=True, targetb="targetb")

    if task in ("train", "forward"):
        net_dict["rna_alignment"] =  {"class": "copy", "from": ["output/rna_alignment"]}  # (B, T)

    return net_dict



network = get_net_dict(pretrain_idx=None)
search_output_layer = "decision"
debug_print_layer_output_template = True

# trainer
batching = "random"
# Seq-length 'data' Stats:
#  37867 seqs
#  Mean: 447.397258827
#  Std dev: 350.353162012
#  Min/max: 15 / 2103
# Seq-length 'bpe' Stats:
#  37867 seqs
#  Mean: 14.1077719386
#  Std dev: 13.3402518828
#  Min/max: 2 / 82
log_batch_size = True
batch_size = 4000
max_seqs = 200
max_seq_length = {"bpe": 75}
#chunking = ""  # no chunking
truncation = -1

def custom_construction_algo(idx, net_dict):
    # For debugging, use: python3 ./crnn/Pretrain.py config...
    return get_net_dict(pretrain_idx=idx)

# No repetitions here. We explicitly do that in the construction.
pretrain = {"copy_param_mode": "subset", "construction_algo": custom_construction_algo}


num_epochs = 250
model = "net-model/network"
cleanup_old_models = True
gradient_clip = 0
#gradient_clip_global_norm = 1.0
adam = True
optimizer_epsilon = 1e-8
accum_grad_multiple_step = 3
#debug_add_check_numerics_ops = True
#debug_add_check_numerics_on_output = True
stop_on_nonfinite_train_score = False
tf_log_memory_usage = True
gradient_noise = 0.0
# lr set above
learning_rate_control = "newbob_multi_epoch"
learning_rate_control_error_measure = "dev_error_output/output_prob"
learning_rate_control_relative_error_relative_lr = True
learning_rate_control_min_num_epochs_per_new_lr = 3
use_learning_rate_control_always = True
newbob_multi_num_epochs = 6
newbob_multi_update_interval = 1
newbob_learning_rate_decay = 0.7
learning_rate_file = "newbob.data"

# log
#log = "| /u/zeyer/dotfiles/system-tools/bin/mt-cat.py >> log/crnn.seq-train.%s.log" % task
log = "log/crnn.%s.log" % task
log_verbosity = 5



