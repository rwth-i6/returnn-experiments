#!rnn.py

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def get_network(epoch, **kwargs):
    from networks import networks_dict

    for epoch_ in sorted(networks_dict.keys(), reverse=True):
        if epoch_ <= epoch:
            return networks_dict[epoch_]
    assert False, "Error, no networks found"


def summary(name, x):
    """
    :param str name:
    :param tf.Tensor x: (batch,time,feature)
    """
    from returnn.tf.compat import v1 as tf

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
    from returnn.tf.compat import v1 as tf

    ndim = x.get_shape().ndims
    n_batch = tf.shape(x)[batch_axis]
    dim = tf.shape(x)[axis]
    amount = tf.random_uniform(
        shape=(n_batch,), minval=1, maxval=max_amount + 1, dtype=tf.int32
    )
    pos2 = tf.minimum(pos + amount, dim)
    idxs = tf.expand_dims(tf.range(0, dim), 0)  # (1,dim)
    pos_bc = tf.expand_dims(pos, 1)  # (batch,1)
    pos2_bc = tf.expand_dims(pos2, 1)  # (batch,1)
    cond = tf.logical_and(
        tf.greater_equal(idxs, pos_bc), tf.less(idxs, pos2_bc)
    )  # (batch,dim)
    if batch_axis > axis:
        cond = tf.transpose(cond)  # (dim,batch)
    cond = tf.reshape(
        cond, [tf.shape(x)[i] if i in (batch_axis, axis) else 1 for i in range(ndim)]
    )
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
    from returnn.tf.compat import v1 as tf

    n_batch = tf.shape(x)[batch_axis]
    if isinstance(min_num, int) and isinstance(max_num, int) and min_num == max_num:
        num = min_num
    else:
        num = tf.random_uniform(
            shape=(n_batch,), minval=min_num, maxval=max_num + 1, dtype=tf.int32
        )
    # https://github.com/tensorflow/tensorflow/issues/9260
    # https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    z = -tf.log(-tf.log(tf.random_uniform((n_batch, tf.shape(x)[axis]), 0, 1)))
    _, indices = tf.nn.top_k(z, num if isinstance(num, int) else tf.reduce_max(num))
    # indices should be sorted, and of shape (batch,num), entries (int32) in [0,dim)
    # indices = tf.Print(indices, ["indices", indices, tf.shape(indices)])
    if isinstance(num, int):
        for i in range(num):
            x = _mask(
                x,
                batch_axis=batch_axis,
                axis=axis,
                pos=indices[:, i],
                max_amount=max_dims,
            )
    else:
        _, x = tf.while_loop(
            cond=lambda i, _: tf.less(i, tf.reduce_max(num)),
            body=lambda i, x: (
                i + 1,
                tf.where(
                    tf.less(i, num),
                    _mask(
                        x,
                        batch_axis=batch_axis,
                        axis=axis,
                        pos=indices[:, i],
                        max_amount=max_dims,
                    ),
                    x,
                ),
            ),
            loop_vars=(0, x),
        )
    return x


def transform(data, network, time_factor=1):
    x = data.placeholder
    from returnn.tf.compat import v1 as tf

    # summary("features", x)
    step = network.global_train_step
    step1 = tf.where(tf.greater_equal(step, 1000), 1, 0)
    step2 = tf.where(tf.greater_equal(step, 2000), 1, 0)

    def get_masked():
        x_masked = x
        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.time_dim_axis,
            min_num=step1 + step2,
            max_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // 100, 2)
            * (1 + step1 + step2 * 2),
            max_dims=20 // time_factor,
        )
        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.feature_dim_axis,
            min_num=step1 + step2,
            max_num=2 + step1 + step2 * 2,
            max_dims=data.dim // 5,
        )
        # summary("features_mask", x_masked)
        return x_masked

    x = network.cond_on_train(get_masked, lambda: x)
    return x


def speed_pert(audio, sample_rate, random_state):
    import librosa

    new_sample_rate = int(sample_rate * (1 + random_state.randint(-1, 2) * 0.1))
    if new_sample_rate != sample_rate:
        audio = librosa.core.resample(
            audio, sample_rate, new_sample_rate, res_type="kaiser_fast"
        )
    return audio


accum_grad_multiple_step = 2
adam = True
batch_size = 2400000
batching = "random"
cleanup_old_models = True
debug_mode = False
debug_print_layer_output_template = True
dev = {
    "class": "MetaDataset",
    "data_map": {
        "audio_features": ("zip_dataset", "data"),
        "bpe_labels": ("zip_dataset", "classes"),
    },
    "datasets": {
        "zip_dataset": {
            "class": "OggZipDataset",
            "path": "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.Wgp6724p1XD2/output/out.ogg.zip",
            "use_cache_manager": True,
            "audio": {
                "features": "raw",
                "peak_normalization": True,
                "preemphasis": None,
            },
            "targets": {
                "class": "BytePairEncoding",
                "bpe_file": "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.codes",
                "vocab_file": "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.vocab",
                "unknown_label": None,
                "seq_postfix": [0],
            },
            "segment_file": None,
            "partition_epoch": 1,
            "seq_ordering": "sorted_reverse",
        }
    },
    "seq_order_control_dataset": "zip_dataset",
}
device = "gpu"
eval_datasets = {
    "devtrain": {
        "class": "MetaDataset",
        "data_map": {
            "audio_features": ("zip_dataset", "data"),
            "bpe_labels": ("zip_dataset", "classes"),
        },
        "datasets": {
            "zip_dataset": {
                "class": "OggZipDataset",
                "path": "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.kjOfflyMvuHh/output/out.ogg.zip",
                "use_cache_manager": True,
                "audio": {
                    "features": "raw",
                    "peak_normalization": True,
                    "preemphasis": None,
                },
                "targets": {
                    "class": "BytePairEncoding",
                    "bpe_file": "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.codes",
                    "vocab_file": "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.vocab",
                    "unknown_label": None,
                    "seq_postfix": [0],
                },
                "segment_file": None,
                "partition_epoch": 1,
                "seq_ordering": "sorted_reverse",
                "fixed_random_subset": 507,
            }
        },
        "seq_order_control_dataset": "zip_dataset",
    }
}
extern_data = {
    "audio_features": {"available_for_inference": True, "shape": (None, 1), "dim": 1},
    "bpe_labels": {
        "available_for_inference": False,
        "shape": (None,),
        "dim": 1057,
        "sparse": True,
    },
}
gradient_clip = 0.0
gradient_noise = 0.0
learning_rate = 0.0008
learning_rate_control = "newbob_multi_epoch"
learning_rate_control_min_num_epochs_per_new_lr = 3
learning_rate_control_relative_error_relative_lr = True
learning_rate_file = "learning_rates"
learning_rates = [
    8e-05,
    8.402234636871508e-05,
    8.804469273743018e-05,
    9.206703910614525e-05,
    9.608938547486035e-05,
    0.00010011173184357542,
    0.00010413407821229052,
    0.00010815642458100559,
    0.00011217877094972067,
    0.00011620111731843576,
    0.00012022346368715085,
    0.00012424581005586593,
    0.000128268156424581,
    0.0001322905027932961,
    0.0001363128491620112,
    0.00014033519553072627,
    0.00014435754189944135,
    0.00014837988826815643,
    0.0001524022346368715,
    0.00015642458100558658,
    0.0001604469273743017,
    0.00016446927374301677,
    0.00016849162011173187,
    0.00017251396648044695,
    0.00017653631284916203,
    0.0001805586592178771,
    0.00018458100558659218,
    0.00018860335195530726,
    0.00019262569832402234,
    0.00019664804469273744,
    0.00020067039106145252,
    0.00020469273743016763,
    0.0002087150837988827,
    0.00021273743016759778,
    0.00021675977653631286,
    0.00022078212290502794,
    0.00022480446927374302,
    0.0002288268156424581,
    0.00023284916201117317,
    0.0002368715083798883,
    0.00024089385474860338,
    0.00024491620111731846,
    0.00024893854748603354,
    0.0002529608938547486,
    0.0002569832402234637,
    0.0002610055865921788,
    0.00026502793296089385,
    0.00026905027932960893,
    0.000273072625698324,
    0.00027709497206703914,
    0.0002811173184357542,
    0.0002851396648044693,
    0.0002891620111731844,
    0.00029318435754189945,
    0.00029720670391061453,
    0.0003012290502793296,
    0.0003052513966480447,
    0.00030927374301675976,
    0.0003132960893854749,
    0.00031731843575419,
    0.00032134078212290505,
    0.00032536312849162013,
    0.0003293854748603352,
    0.0003334078212290503,
    0.00033743016759776536,
    0.00034145251396648044,
    0.0003454748603351955,
    0.0003494972067039106,
    0.0003535195530726257,
    0.0003575418994413408,
    0.0003615642458100559,
    0.00036558659217877096,
    0.00036960893854748604,
    0.0003736312849162011,
    0.0003776536312849162,
    0.0003816759776536313,
    0.00038569832402234635,
    0.00038972067039106143,
    0.00039374301675977656,
    0.00039776536312849164,
    0.0004017877094972067,
    0.0004058100558659218,
    0.0004098324022346369,
    0.00041385474860335195,
    0.00041787709497206703,
    0.0004218994413407821,
    0.0004259217877094972,
    0.0004299441340782123,
    0.0004339664804469274,
    0.0004379888268156425,
    0.00044201117318435755,
    0.00044603351955307263,
    0.0004500558659217877,
    0.0004540782122905028,
    0.00045810055865921787,
    0.00046212290502793294,
    0.000466145251396648,
    0.00047016759776536315,
    0.00047418994413407823,
    0.0004782122905027933,
    0.0004822346368715084,
    0.00048625698324022347,
    0.0004902793296089386,
    0.0004943016759776537,
    0.0004983240223463688,
    0.0005023463687150838,
    0.0005063687150837989,
    0.000510391061452514,
    0.0005144134078212291,
    0.0005184357541899441,
    0.0005224581005586592,
    0.0005264804469273743,
    0.0005305027932960894,
    0.0005345251396648045,
    0.0005385474860335195,
    0.0005425698324022347,
    0.0005465921787709498,
    0.0005506145251396649,
    0.00055463687150838,
    0.000558659217877095,
    0.0005626815642458101,
    0.0005667039106145252,
    0.0005707262569832403,
    0.0005747486033519553,
    0.0005787709497206704,
    0.0005827932960893855,
    0.0005868156424581006,
    0.0005908379888268157,
    0.0005948603351955307,
    0.0005988826815642458,
    0.0006029050279329609,
    0.000606927374301676,
    0.000610949720670391,
    0.0006149720670391061,
    0.0006189944134078212,
    0.0006230167597765363,
    0.0006270391061452514,
    0.0006310614525139664,
    0.0006350837988826816,
    0.0006391061452513967,
    0.0006431284916201118,
    0.0006471508379888269,
    0.0006511731843575419,
    0.000655195530726257,
    0.0006592178770949721,
    0.0006632402234636872,
    0.0006672625698324022,
    0.0006712849162011173,
    0.0006753072625698324,
    0.0006793296089385475,
    0.0006833519553072626,
    0.0006873743016759776,
    0.0006913966480446927,
    0.0006954189944134078,
    0.0006994413407821229,
    0.000703463687150838,
    0.0007074860335195531,
    0.0007115083798882682,
    0.0007155307262569833,
    0.0007195530726256984,
    0.0007235754189944134,
    0.0007275977653631285,
    0.0007316201117318436,
    0.0007356424581005587,
    0.0007396648044692738,
    0.0007436871508379888,
    0.0007477094972067039,
    0.000751731843575419,
    0.0007557541899441341,
    0.0007597765363128491,
    0.0007637988826815642,
    0.0007678212290502793,
    0.0007718435754189944,
    0.0007758659217877095,
    0.0007798882681564246,
    0.0007839106145251397,
    0.0007879329608938548,
    0.0007919553072625699,
    0.000795977653631285,
    0.0008,
    0.0008,
    0.000795977653631285,
    0.0007919553072625699,
    0.0007879329608938548,
    0.0007839106145251397,
    0.0007798882681564246,
    0.0007758659217877096,
    0.0007718435754189945,
    0.0007678212290502794,
    0.0007637988826815643,
    0.0007597765363128491,
    0.0007557541899441341,
    0.000751731843575419,
    0.0007477094972067039,
    0.0007436871508379888,
    0.0007396648044692738,
    0.0007356424581005587,
    0.0007316201117318436,
    0.0007275977653631285,
    0.0007235754189944134,
    0.0007195530726256984,
    0.0007155307262569833,
    0.0007115083798882682,
    0.0007074860335195531,
    0.0007034636871508381,
    0.000699441340782123,
    0.0006954189944134079,
    0.0006913966480446927,
    0.0006873743016759777,
    0.0006833519553072626,
    0.0006793296089385475,
    0.0006753072625698324,
    0.0006712849162011173,
    0.0006672625698324022,
    0.0006632402234636872,
    0.0006592178770949721,
    0.000655195530726257,
    0.0006511731843575419,
    0.0006471508379888269,
    0.0006431284916201118,
    0.0006391061452513967,
    0.0006350837988826816,
    0.0006310614525139665,
    0.0006270391061452515,
    0.0006230167597765363,
    0.0006189944134078213,
    0.0006149720670391061,
    0.0006109497206703912,
    0.000606927374301676,
    0.0006029050279329609,
    0.0005988826815642458,
    0.0005948603351955307,
    0.0005908379888268157,
    0.0005868156424581006,
    0.0005827932960893855,
    0.0005787709497206704,
    0.0005747486033519553,
    0.0005707262569832403,
    0.0005667039106145252,
    0.0005626815642458101,
    0.000558659217877095,
    0.0005546368715083798,
    0.0005506145251396649,
    0.0005465921787709497,
    0.0005425698324022347,
    0.0005385474860335195,
    0.0005345251396648046,
    0.0005305027932960894,
    0.0005264804469273744,
    0.0005224581005586592,
    0.0005184357541899441,
    0.0005144134078212291,
    0.000510391061452514,
    0.0005063687150837989,
    0.0005023463687150838,
    0.0004983240223463688,
    0.0004943016759776537,
    0.0004902793296089386,
    0.00048625698324022347,
    0.0004822346368715084,
    0.0004782122905027933,
    0.00047418994413407823,
    0.00047016759776536315,
    0.0004661452513966481,
    0.000462122905027933,
    0.0004581005586592179,
    0.00045407821229050284,
    0.0004500558659217877,
    0.00044603351955307263,
    0.00044201117318435755,
    0.0004379888268156425,
    0.0004339664804469274,
    0.0004299441340782123,
    0.00042592178770949724,
    0.00042189944134078216,
    0.0004178770949720671,
    0.000413854748603352,
    0.0004098324022346369,
    0.0004058100558659218,
    0.0004017877094972067,
    0.00039776536312849164,
    0.00039374301675977656,
    0.0003897206703910615,
    0.0003856983240223464,
    0.00038167597765363133,
    0.00037765363128491625,
    0.0003736312849162011,
    0.00036960893854748604,
    0.00036558659217877096,
    0.0003615642458100559,
    0.0003575418994413408,
    0.00035351955307262573,
    0.00034949720670391065,
    0.0003454748603351956,
    0.0003414525139664805,
    0.00033743016759776536,
    0.0003334078212290503,
    0.0003293854748603352,
    0.00032536312849162013,
    0.00032134078212290505,
    0.00031731843575419,
    0.0003132960893854749,
    0.00030927374301675976,
    0.0003052513966480447,
    0.0003012290502793296,
    0.00029720670391061453,
    0.00029318435754189945,
    0.0002891620111731844,
    0.0002851396648044693,
    0.0002811173184357542,
    0.00027709497206703914,
    0.00027307262569832406,
    0.000269050279329609,
    0.0002650279329608939,
    0.00026100558659217883,
    0.00025698324022346375,
    0.00025296089385474867,
    0.0002489385474860336,
    0.0002449162011173184,
    0.00024089385474860333,
    0.00023687150837988825,
    0.00023284916201117317,
    0.0002288268156424581,
    0.00022480446927374302,
    0.00022078212290502794,
    0.00021675977653631286,
    0.00021273743016759778,
    0.0002087150837988827,
    0.00020469273743016763,
    0.00020067039106145255,
    0.00019664804469273747,
    0.0001926256983240224,
    0.00018860335195530732,
    0.00018458100558659224,
    0.00018055865921787716,
    0.00017653631284916208,
    0.0001725139664804469,
    0.00016849162011173182,
    0.00016446927374301674,
    0.00016044692737430166,
    0.00015642458100558658,
    0.0001524022346368715,
    0.00014837988826815643,
    0.00014435754189944135,
    0.00014033519553072627,
    0.0001363128491620112,
    0.00013229050279329612,
    0.00012826815642458104,
    0.00012424581005586596,
    0.00012022346368715088,
    0.0001162011173184358,
    0.00011217877094972073,
    0.00010815642458100565,
    0.00010413407821229057,
    0.00010011173184357538,
    9.60893854748603e-05,
    9.206703910614523e-05,
    8.804469273743015e-05,
    8.402234636871507e-05,
    8e-05,
    8e-05,
    7.797435897435898e-05,
    7.594871794871796e-05,
    7.392307692307693e-05,
    7.18974358974359e-05,
    6.987179487179487e-05,
    6.784615384615385e-05,
    6.582051282051282e-05,
    6.37948717948718e-05,
    6.176923076923078e-05,
    5.974358974358975e-05,
    5.771794871794872e-05,
    5.5692307692307696e-05,
    5.366666666666667e-05,
    5.164102564102564e-05,
    4.961538461538462e-05,
    4.758974358974359e-05,
    4.556410256410257e-05,
    4.3538461538461545e-05,
    4.1512820512820514e-05,
    3.948717948717949e-05,
    3.7461538461538466e-05,
    3.5435897435897435e-05,
    3.341025641025641e-05,
    3.1384615384615386e-05,
    2.9358974358974362e-05,
    2.7333333333333338e-05,
    2.5307692307692307e-05,
    2.3282051282051283e-05,
    2.125641025641026e-05,
    1.9230769230769228e-05,
    1.7205128205128204e-05,
    1.517948717948718e-05,
    1.3153846153846156e-05,
    1.1128205128205131e-05,
    9.102564102564107e-06,
    7.076923076923083e-06,
    5.0512820512820455e-06,
    3.0256410256410213e-06,
    1e-06,
]
log = ["./returnn.log"]
log_batch_size = True
log_verbosity = 5
max_seqs = 200
min_learning_rate = 1.6e-05
model = "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.JAdkojbfUofl/output/models/epoch"
newbob_learning_rate_decay = 0.9
newbob_multi_num_epochs = 20
newbob_multi_update_interval = 1
num_epochs = 400
optimizer_epsilon = 1e-08
save_interval = 1
search_output_layer = "decision"
target = "classes"
task = "train"
tf_log_memory_usage = True
tf_session_opts = {"gpu_options": {"per_process_gpu_memory_fraction": 0.92}}
train = {
    "class": "MetaDataset",
    "data_map": {
        "audio_features": ("zip_dataset", "data"),
        "bpe_labels": ("zip_dataset", "classes"),
    },
    "datasets": {
        "zip_dataset": {
            "class": "OggZipDataset",
            "path": "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/oggzip/BlissToOggZipJob.kjOfflyMvuHh/output/out.ogg.zip",
            "use_cache_manager": True,
            "audio": {
                "features": "raw",
                "peak_normalization": True,
                "preemphasis": None,
                "pre_process": speed_pert,
            },
            "targets": {
                "class": "BytePairEncoding",
                "bpe_file": "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.codes",
                "vocab_file": "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.vocab",
                "unknown_label": None,
                "seq_postfix": [0],
            },
            "segment_file": None,
            "partition_epoch": 4,
            "seq_ordering": "laplace:.1000",
        }
    },
    "seq_order_control_dataset": "zip_dataset",
}
truncation = -1
use_learning_rate_control_always = True
use_tensorflow = True
config = {}

locals().update(**config)

import sys

sys.setrecursionlimit(3000)
