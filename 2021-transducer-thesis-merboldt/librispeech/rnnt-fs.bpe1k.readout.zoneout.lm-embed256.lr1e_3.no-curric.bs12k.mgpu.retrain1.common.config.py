#!crnn/rnn.py
# kate: syntax python;
# vim: ft=python sw=2:
# adapted for import_ & common recipes

from returnn.tf.util.data import DimensionTag, Data
from returnn.import_ import import_

import_("github.com/rwth-i6/returnn-experiments", "common", "20210324-75f7809")
from returnn_import.github_com.rwth_i6.returnn_experiments.v20210324123103_75f78096518b.common.common_config import *
from returnn_import.github_com.rwth_i6.returnn_experiments.v20210324123103_75f78096518b.common.datasets.asr.librispeech import oggzip, vocabs


use_horovod = config.bool("use_horovod", False)
horovod_dataset_distribution = "random_seed_offset"
horovod_reduce_type = "param"
#horovod_param_sync_step = 100
horovod_param_sync_time_diff = 100.
# horovod_scale_lr = True

if use_horovod:
  import socket
  prefix = "%s-pid%i:" % (socket.gethostname(), os.getpid())
  print(prefix, "use_horovod, CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", None))
  import TFHorovod
  # Important: Horovod options must be defined before this call!
  hvd = TFHorovod.get_ctx(config=config)
  print(prefix, "Local rank/size:", hvd.local_rank(), hvd.local_size())

# Workaround for openblas hanging:
# * https://github.com/tensorflow/tensorflow/issues/13802
# * https://github.com/rwth-i6/returnn/issues/323#issuecomment-725384762
# patch_atfork = not use_horovod

# task
use_tensorflow = True
task = config.value("task", "train")
device = "gpu"
multiprocessing = True
update_on_device = True

debug_mode = False
if int(os.environ.get("RETURNN_DEBUG", "0")):
  import sys
  print("** DEBUG MODE", file=sys.stderr)
  debug_mode = True

if config.has("beam_size"):
  beam_size = config.int("beam_size", 0)
  import sys
  print("** beam_size %i" % beam_size, file=sys.stderr)
else:
  beam_size = 12

# data
globals().update(oggzip.Librispeech.old_defaults(vocab=vocabs.bpe1k).get_config_opts())

# Redefine extern_data anyway...
_time_tag = DimensionTag(kind=DimensionTag.Types.Spatial, description="time")
_target = "classes"
_target_num_labels = 1056
_targetb_num_labels = _target_num_labels + 1
_targetb_blank_idx = _target_num_labels

extern_data = {
    _target: {"dim": _target_num_labels, "sparse": True},  # see vocab
    "data": {"dim": 40, "same_dim_tags_as": {"t": _time_tag}},  # Gammatone 40-dim
    }
if task != "train":
  extern_data["targetb"] = {"dim": _targetb_num_labels, "sparse": True, "available_for_inference": False}

_epoch_split = 20


def _mask(x, batch_axis, axis, pos, max_amount, mask_value=0.):
    """
    :param tf.Tensor x: (batch,time,[feature])
    :param int batch_axis:
    :param int axis:
    :param tf.Tensor pos: (batch,)
    :param int|tf.Tensor max_amount: inclusive
    :param float|int mask_value:
    """
    from returnn.tf.compat import v1 as tf
    from returnn.tf.util.basic import where_bc
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
    x = where_bc(cond, mask_value, x)
    return x


def random_mask(x, batch_axis, axis, min_num, max_num, max_dims, mask_value=0.):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param int|tf.Tensor min_num:
    :param int|tf.Tensor max_num: inclusive
    :param int|tf.Tensor max_dims: inclusive
    :param float|int mask_value:
    """
    from returnn.tf.compat import v1 as tf
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
            x = _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims, mask_value=mask_value)
    else:
        _, x = tf.while_loop(
            cond=lambda i, _: tf.less(i, tf.reduce_max(num)),
            body=lambda i, x: (
                i + 1,
                tf.where(
                    tf.less(i, num),
                    _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims, mask_value=mask_value),
                    x)),
            loop_vars=(0, x))
    return x

def transform(source, **kwargs):
  from returnn.tf.compat import v1 as tf
  data = source(0, as_data=True)
  time_factor = 1  #  for switchout == 6
  x = data.placeholder
  network = kwargs["self"].network
  from returnn.tf.compat import v1 as tf
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
    return x_masked
  x = network.cond_on_train(get_masked, lambda: x)
  return x


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
            std::map<tstring, std::set<int> > new_hyps;  // seq -> set of hyp idx

            for(int hyp = 0; hyp < n_beam; ++hyp) {
                auto& seq_set = new_hyps[prev_str->tensor<tstring, 2>()(bat, hyp)];
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
    from returnn.tf.util.basic import OpCodeCompiler
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
    from returnn.tf.compat import v1 as tf
    from returnn.tf.util.basic import where_bc, get_shared_vocab
    with tf.device("/cpu:0"):
        labels_t = get_shared_vocab(labels)
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
   from returnn.tf.compat import v1 as tf
   from returnn.tf.util.basic import where_bc, nd_indices, tile_transposed
   from returnn.datasets.generating import Vocabulary

   dim = layer.output.dim
   
   prev_str = layer.explicit_search_sources[0].output  # [B*beam], str
   prev_str_t = tf.reshape(prev_str.placeholder, (batch_dim, -1))[:,:base_beam_in]
   prev_out = layer.explicit_search_sources[1].output  # [B*beam], int32
   prev_out_t = tf.reshape(prev_out.placeholder, (batch_dim, -1))[:,:base_beam_in]

   dataset = get_dataset("train")
   vocab = Vocabulary.create_vocab(**dataset["bpe"])
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



def rnnt_loss(source, **kwargs):
    """
    Computes the RNN-T loss function.

    :param log_prob:
    :return:
    """
    # acts: (B, T, U + 1, V)
    # targets: (B, T)
    # input_lengths (B,)
    # label_lengths (B,)
    from returnn.extern.HawkAaronWarpTransducer import rnnt_loss

    log_probs = source(0, as_data=True, auto_convert=False)
    targets = source(1, as_data=True, auto_convert=False)
    encoder = source(2, as_data=True, auto_convert=False)

    enc_lens = encoder.get_sequence_lengths()
    dec_lens = targets.get_sequence_lengths()

    costs = rnnt_loss(log_probs.get_placeholder_as_batch_major(), targets.get_placeholder_as_batch_major(), enc_lens, dec_lens,
            blank_label=_targetb_blank_idx)
    costs.set_shape((None,))  # (B,)
    return costs


def rnnt_alignment(source, **kwargs):
    """
    Computes the RNN-T alignment (forced alignment).

    :param log_prob:
    :return:
    """
    # alignment-length (B,T+U+1)
    # acts: (B, T, U+1, V)
    # targets: (B, U)
    # input_lengths (B,)
    # label_lengths (B,)
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
    from rnnt_tf_impl import rnnt_loss

    alignment_length = source(0, as_data=True, auto_convert=False)
    log_probs = source(1, as_data=True, auto_convert=False).get_placeholder_as_batch_major()
    targets = source(2, as_data=True, auto_convert=False)
    encoder = source(3, as_data=True, auto_convert=False)

    enc_lens = encoder.get_sequence_lengths()
    dec_lens = targets.get_sequence_lengths()

    # print_op = tf.print({"max(U+T)": tf.reduce_max(enc_lens+dec_lens), "alignment-length": alignment_length.get_sequence_lengths()}, summarize=-1)
    # with tf.control_dependencies([print_op]):
        # log_probs = tf.identity(log_probs)

    costs, alignment = rnnt_loss(log_probs, targets.get_placeholder_as_batch_major(), enc_lens, dec_lens,
                                 blank_index=_targetb_blank_idx, debug=False, with_alignment=True)
    return alignment # (B, T)


# network, 5 epochs warmup, 5 epochs pretraining
_pretrain_warmup_lr_frac = 0.5
_range_epochs_pretrain_fullsum = (0, 10)
_range_epochs_full_sum = (0, _epoch_split * 100)


def _get_network_align(epoch0: int):
  net_dict = _get_network(full_sum_alignment=True, target="bpe" if task == "train" else "targetb")
  net_dict["#trainable"] = False  # disable training
  net_dict["#finish_all_data"] = True  # in case of multi-GPU training or so
  subnet = net_dict["output"]["unit"]
  subnet["fullsum_alignment"] = {
          "class": "eval",
          "from": ["output_log_prob", "base:data:" + _target, "base:encoder"],
          "eval": rna_fullsum_alignment,
          "out_type": lambda sources, **kwargs: Data(name="rna_alignment_output", sparse=True, dim=_targetb_num_labels,
                                                     size_placeholder={0: sources[2].output.size_placeholder[0]}),
          "is_output_layer": True
      }
  align_dir = os.path.dirname(model)
  subnet["_align_dump"] = {
    "class": "hdf_dump",
    "from": "fullsum_alignment",
    "is_output_layer": True,
    "dump_per_run": True,
    "extend_existing_file": epoch0 % EpochSplit > 0,
    "filename": (lambda **opts: "%s/align.{dataset_name}.hdf".format(**opts) % align_dir),
  }
  return net_dict


def get_network(epoch: int, **kwargs):
    epoch0 = epoch - 1  # RETURNN starts with epoch 1, but 0-indexed is easier here
    if _range_epochs_full_sum[0] <= epoch0 < _range_epochs_full_sum[1]:
        print("Epoch %i: Constructing network using full-sum formulation." % epoch)
        return _get_network_full_sum(epoch0=epoch0)


def _get_network_full_sum(epoch0: int):
  if epoch0 < _range_epochs_pretrain_fullsum[0]:
    pretrain_frac = 0
  elif epoch0 < _range_epochs_pretrain_fullsum[1]:
    pretrain_frac = (
        float(epoch0 - _range_epochs_pretrain_fullsum[0]) /
        (_range_epochs_pretrain_fullsum[1] - _range_epochs_pretrain_fullsum[0]))
  else:
    pretrain_frac = 1
  print("Epoch %i: Constructing network using full-sum, pretrain_frac=%.1f" % (epoch0+1, pretrain_frac))
  net_dict = _get_network(full_sum_loss=True, grow_encoder=False, pretrain_frac=pretrain_frac, target=_target if task == "train" else "targetb")
  net_dict["#copy_param_mode"] = "subset"
  return net_dict

def _get_network(target: str, full_sum_loss: bool = False, full_sum_alignment: bool = False, ce_loss: bool = False,
                 pretrain_frac: float = 1, grow_encoder: bool = True):
  full_sum = full_sum_loss or full_sum_alignment
  net_dict = {"#config": {}}
  if pretrain_frac < _pretrain_warmup_lr_frac:
    start_lr = learning_rate / 10.
    net_dict["#config"]["learning_rate"] = start_lr + (1/_pretrain_warmup_lr_frac)*pretrain_frac * learning_rate
  elif pretrain_frac < 1:  # constant for the rest of pretraining
    net_dict["#config"]["learning_rate"] = learning_rate


  EncKeyTotalDim = 200
  AttentionDropout = 0.1
  EncValueTotalDim = 2048
  LstmDim = 1024
  AttNumHeads = 1
  EncKeyPerHeadDim = EncKeyTotalDim // AttNumHeads
  l2 = 0.0001
  net_dict.update({
    "source": {"class": "eval", "eval": transform},
    "source0": {"class": "split_dims", "axis": "F", "dims": (-1, 1), "from": "source"},  # (T,40,1)

    # Lingvo: ep.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)],  ep.conv_filter_strides = [(2, 2), (2, 2)]
    "conv0": {"class": "conv", "from": "source0", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None, "with_bias": True},  # (T,40,32)
    "conv0p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv0"},  # (T,20,32)
    "conv1": {"class": "conv", "from": "conv0p", "padding": "same", "filter_size": (3, 3), "n_out": 32, "activation": None, "with_bias": True},  # (T,20,32)
    "conv1p": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "conv1"},  # (T,10,32)
    "conv_merged": {"class": "merge_dims", "from": "conv1p", "axes": "static"},  # (T,320)

    # Encoder LSTMs added below, resulting in "encoder0".
    "encoder": {"class": "copy", "from": "encoder0"},
    "enc_ctx0": {"class": "linear", "from": "encoder", "activation": None, "with_bias": False, "n_out": EncKeyTotalDim},
    "enc_ctx_win": {"class": "window", "from": "enc_ctx0", "window_size": 5},  # [B,T,W,D]
    "enc_val": {"class": "copy", "from": "encoder"},
    "enc_val_win": {"class": "window", "from": "enc_val", "window_size": 5},  # [B,T,W,D]

    "enc_seq_len": {"class": "length", "from": "encoder", "sparse": False},

    # for task "search" / search_output_layer
    "output_wo_b0": {
      "class": "masked_computation", "unit": {"class": "copy"},
      "from": "output", "mask": "output/output_emit"},
    "output_wo_b": {"class": "reinterpret_data", "from": "output_wo_b0", "set_sparse_dim": _target_num_labels},
    "decision": {
        "class": "decide", "from": "output_wo_b", "loss": "edit_distance", "target": _target,
        'only_on_search': True},

    "_target_masked": {"class": "masked_computation",
                       "mask": "output/output_emit",
                       "from": "output",
                       "unit": {"class": "copy"}},
    "3_target_masked": {
        "class": "reinterpret_data", "from": "_target_masked",
        "set_sparse_dim": _target_num_labels,  # we masked blank away
        "enforce_batch_major": True,  # ctc not implemented otherwise...
        "register_as_extern_data": "targetb_masked" if task == "train" else None},
    })

  # Add encoder BLSTM stack.
  start_num_lstm_layers = 2
  final_num_lstm_layers = 6
  start_dim_factor = 0.5
  if grow_encoder:
    num_lstm_layers = start_num_lstm_layers + int((final_num_lstm_layers-start_num_lstm_layers)*pretrain_frac)
    grow_frac = 1.0 - float(final_num_lstm_layers - num_lstm_layers) / (final_num_lstm_layers - start_num_lstm_layers)
    dim_frac = start_dim_factor + (1.0 - start_dim_factor) * grow_frac
  else:
    num_lstm_layers = final_num_lstm_layers
    dim_frac = 1.
  time_reduction = [3, 2] if num_lstm_layers >= 3 else [6]
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
  net_dict["encoder0"] = {"class": "copy", "from": src}
  net_dict["lm_input0"] = {"class": "copy", "from": "data:%s" % target}
  net_dict["lm_input1"] = {"class": "prefix_in_time", "from": "lm_input0", "prefix": 0}
  net_dict["lm_input"] = {"class": "copy", "from": "lm_input1"}

  def get_output_dict(train, search, target, beam_size=beam_size):
      return {
          "class": "rec",
          "from": "encoder",
          "include_eos": True,
          "back_prop": (task == "train") and train,
          "unit": {
              "am0": {"class": "gather_nd", "from": "base:encoder", "position": "prev:t"},  # [B,D]
              "am": {"class": "copy", "from": "data:source" if task == "train" else "am0"},

              "prev_out_non_blank": {
                  "class": "reinterpret_data", "from": "prev:output_", "set_sparse_dim": _target_num_labels},
              "lm_masked": {"class": "masked_computation",
                  "mask": "prev:output_emit",
                  "from": "prev_out_non_blank",  # in decoding
                  "masked_from": "base:lm_input" if task == "train" else None,  # enables optimization if used
                  "unit": {
                  "class": "subnetwork", "from": "data",
                  "subnetwork": {
                      "input_embed": {"class": "linear", "activation": None, "with_bias": False, "from": "data", "n_out": 256},
                      "embed_dropout": {"class": "dropout", "from": "input_embed", "dropout": 0.2},
                      # "lstm0": {"class": "rec", "unit": "nativelstm2", "n_out": LstmDim, "from": ["embed_dropout"], "L2": l2},
                      "lstm0_zoneout": {"class": "rnn_cell", "unit": "ZoneoutLSTM", "unit_opts": {"zoneout_factor_cell": 0.15, "zoneout_factor_output": 0.05}, "from": ["embed_dropout"], "n_out": 500},
                      "output": {"class": "copy", "from": "lstm0_zoneout"}
                  }}},
              "readout_in": {"class": "linear", "from": ["am", "lm_masked"], "activation": None, "n_out": 1000, "L2": l2, "dropout": 0.2,
                "out_type": {"batch_dim_axis": 2 if task == "train" else 0, "shape": (None, None, 1000) if task == "train" else (1000,),
                  "time_dim_axis": 0 if task == "train" else None}}, # (T, U+1, B, 1000
              "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": "readout_in"},

              "label_log_prob": {
                    "class": "linear", "from": "readout", "activation": "log_softmax", "dropout": 0.3, "n_out": _target_num_labels},  # (B, T, U+1, 1030)
              "emit_prob0": {"class": "linear", "from": "readout", "activation": None, "n_out": 1, "is_output_layer": True},  # (B, T, U+1, 1)
              "emit_log_prob": {"class": "activation", "from": "emit_prob0", "activation": "log_sigmoid"},  # (B, T, U+1, 1)
              "blank_log_prob": {"class": "eval", "from": "emit_prob0", "eval": "tf.compat.v1.log_sigmoid(-source(0))"},  # (B, T, U+1, 1)
              "label_emit_log_prob": {"class": "combine", "kind": "add", "from": ["label_log_prob", "emit_log_prob"]},  # (B, T, U+1, 1), scaling factor in log-space
              "output_log_prob": {"class": "copy", "from": ["label_emit_log_prob", "blank_log_prob"]},  # (B, T, U+1, 1031)

              "output": {
                  "class": 'choice', 'target': target, 'beam_size': beam_size,
                  'from': "output_log_prob", "input_type": "log_prob",
                  "initial_output": 0,
                  "length_normalization": False,
                  "cheating": "exclusive" if task == "train" else None,
                  "explicit_search_sources": ["prev:out_str", "prev:output"] if task == "search" else None,
                  "custom_score_combine": targetb_recomb_recog if task == "search" else None
              },
              # switchout only applicable to viterbi training, added below.
              "output_": {"class": "copy", "from": "output", "initial_output": 0},

              # "alignment_length0": {"class": "prefix_in_time", "from": "base:lm_input0", "repeat": "base:enc_seq_len", "prefix": 0, "register_as_extern_data": "alignment"},

              # "fullsum_alignment0": {
                  # "class": "eval",
                  # "from": ["alignment_length0", "output_log_prob", "base:data:" + _target, "base:encoder"],
                  # "eval": rnnt_alignment,
                  # "size_target": "alignment",
                  # "out_type": lambda sources, **kwargs: Data(name="rnnt_alignment_output", sparse=True, dim=_targetb_num_labels,
                      # size_placeholder={}),
                  # "is_output_layer": True,
              # },

              "out_str": {
                  "class": "eval", "from": ["prev:out_str", "output_emit", "output"],
                  "initial_output": None, "out_type": {"shape": (), "dtype": "string"},
                  "eval": out_str},

              "output_is_not_blank": {"class": "compare", "from": "output_", "value": _targetb_blank_idx,
                  "kind": "not_equal", "initial_output": True},

              # initial state=True so that we are consistent to the training and the initial state is correctly set.
              "output_emit": { "class": "copy", "from": "output_is_not_blank", "is_output_layer": True, "initial_output": True},

              "const0": {"class": "constant", "value": 0, "collocate_with": ["du", "dt", "t", "u"], "dtype": "int32"},
              "const1": {"class": "constant", "value": 1, "collocate_with": ["du", "dt", "t", "u"], "dtype": "int32"},
              
              # pos in target, [B]
              "du": {"class": "switch", "condition": "output_emit", "true_from": "const1", "false_from": "const0"},
              "u": {"class": "combine", "from": ["prev:u", "du"], "kind": "add", "initial_output": 0},

              # pos in input, [B]
              # output label: stay in t, otherwise advance t (encoder)
              "dt": {"class": "switch", "condition": "output_is_not_blank", "true_from": "const0", "false_from": "const1"},
              "t": {"class": "combine", "from": ["dt", "prev:t"], "kind": "add", "initial_output": 0},

              # stop at U+T
              # in recog: stop when all input has been consumed
              # in train: defined by target.
              "end": {"class": "compare", "from": ["t", "base:enc_seq_len"], "kind": "greater"},
          },
          "max_seq_len": "max_len_from('base:encoder') * 3",
      }

  net_dict["output"] = get_output_dict(train=(task=="train"), search=(task != "train"), target=target, beam_size=beam_size)

  subnet = net_dict["output"]["unit"]

  if ce_loss:  # Viterbi training, uses a more powerful state-layer
      subnet["output_prob"] = {
              "class": "activation", "from": "output_log_prob", "activation": "exp",
              "target": target, "loss": "ce", "loss_opts": {"focal_loss_factor": 2.0}
          }
      if task == "train":  # SwitchOut in training
          subnet["output_"] = {
                  "class": "eval", "from": "output", "eval": switchout_target, "initial_output": 0
                  }
  if full_sum:
      # Fullsum loss requires way more memory
      del net_dict["_target_masked"]
      del net_dict["3_target_masked"]
      # Dropout regularization
      net_dict["enc_ctx0"]["dropout"] = 0.2
      net_dict["enc_ctx0"]["L2"] = l2
      subnet["output_prob"] = {
              "class": "eval",
              "from": ["output_log_prob", "base:data:" + _target, "base:encoder"],
              "eval": rnnt_loss,
              "out_type": lambda sources, **kwargs: Data(name="rnnt_loss", shape=()), "loss": "as_is",
          }
  return net_dict


search_output_layer = "decision"
debug_print_layer_output_template = True

# trainer
batching = "random"
log_batch_size = True
batch_size = 1000 if debug_mode else 12000
max_seqs = 10 if debug_mode else 200
max_seq_length = {"classes": 75}
#chunking = ""  # no chunking
truncation = -1


num_epochs = _range_epochs_full_sum [1] + 1
model = "net-model/network"
cleanup_old_models = True
gradient_clip = 0
#gradient_clip_global_norm = 1.0

adam = True
optimizer_epsilon = 1e-8
#debug_add_check_numerics_ops = True
#debug_add_check_numerics_on_output = True
stop_on_nonfinite_train_score = False
tf_log_memory_usage = True
gradient_noise = 0.0
learning_rate = 0.001
learning_rate_control = "newbob_multi_epoch"
#learning_rate_control_error_measure = "dev_score_output"
learning_rate_control_relative_error_relative_lr = True
learning_rate_control_min_num_epochs_per_new_lr = 3
use_learning_rate_control_always = True
newbob_multi_num_epochs = _epoch_split
newbob_multi_update_interval = 1
newbob_learning_rate_decay = 0.9
learning_rate_file = "newbob.data"

# log
#log = "| /u/zeyer/dotfiles/system-tools/bin/mt-cat.py >> log/crnn.seq-train.%s.log" % task
#model_name = os.path.splitext(os.path.basename(__file__))[0]
#log = "/var/tmp/am540506/log/%s/crnn.%s.log" % (model_name, task)
log_verbosity = 5

