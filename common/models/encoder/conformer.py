
from ...asr.specaugment import specaugment_eval_func


def make_encoder(source="data", **kwargs):
  conformer_enc = ConformerEncoder(**kwargs)
  conformer_enc.create_network()
  return {"class": "subnetwork", "subnetwork": conformer_enc.network.get_net(), "from": source}


class ConformerEncoder:
  """
  Represents Conformer Encoder Architecture

  * Conformer: Convolution-augmented Transformer for Speech Recognition
  * Ref: https://arxiv.org/abs/2005.08100
  """

  def __init__(self, source='data',
               use_specaugment=True,
               input_layer='conv',
               num_blocks=16, conv_kernel_size=32, pos_enc='rel',
               activation='swish', block_final_norm=True, ff_dim=512, ff_init=None, ff_bias=True,
               embed_dropout=0.1, dropout=0.1, att_dropout=0.1,
               enc_key_dim=256, att_num_heads=4,
               target='bpe', l2=None,
               lstm_dropout=0.1, rec_weight_dropout=0., with_ctc=False, native_ctc=False, ctc_dropout=0., ctc_l2=0.,
               ctc_opts=None):
    """
    :param str source: input layer name
    :param bool use_specaugment:
    :param str input_layer: type of input layer which does subsampling
    :param int num_blocks: number of Conformer blocks
    :param int conv_kernel_size: kernel size for conv layers in Convolution module
    :param str|None activation: activation used to sandwich modules
    :param bool block_final_norm: if True, apply layer norm at the end of each conformer block
    :param bool final_norm: if True, apply layer norm to the output of the encoder
    :param int|None ff_dim: dimension of the first linear layer in FF module
    :param str|None ff_init: FF layers initialization
    :param bool|None ff_bias: If true, then bias is used for the FF layers
    :param float embed_dropout: dropout applied to the source embedding
    :param float dropout: general dropout
    :param float att_dropout: dropout applied to attention weights
    :param int enc_key_dim: encoder key dimension, also denoted as d_model, or d_key
    :param int att_num_heads: the number of attention heads
    :param str target: target labels key name
    :param float l2: add L2 regularization for trainable weights parameters
    :param float lstm_dropout: dropout applied to the input of the LSTMs in case they are used
    :param float rec_weight_dropout: dropout applied to the hidden-to-hidden weight matrices of the LSTM in case used
    :param bool with_ctc: if true, CTC loss is used
    :param bool native_ctc: if true, use returnn native ctc implementation instead of TF implementation
    :param float ctc_dropout: dropout applied on input to ctc
    :param float ctc_l2: L2 applied to the weight matrix of CTC softmax
    :param dict[str] ctc_opts: options for CTC
    """

    self.source = source
    self.input_layer = input_layer

    self.num_blocks = num_blocks
    self.conv_kernel_size = conv_kernel_size

    self.pos_enc = pos_enc

    self.ff_init = ff_init
    self.ff_bias = ff_bias

    self.use_specaugment = use_specaugment

    self.activation = activation

    self.block_final_norm = block_final_norm

    self.embed_dropout = embed_dropout
    self.dropout = dropout
    self.att_dropout = att_dropout
    self.lstm_dropout = lstm_dropout

    # key and value dimensions are the same
    self.enc_key_dim = enc_key_dim
    self.enc_value_dim = enc_key_dim
    self.att_num_heads = att_num_heads
    self.enc_key_per_head_dim = enc_key_dim // att_num_heads
    self.enc_val_per_head_dim = enc_key_dim // att_num_heads

    self.ff_dim = ff_dim
    if self.ff_dim is None:
      self.ff_dim = 2 * self.enc_key_dim

    self.target = target

    self.l2 = l2
    self.rec_weight_dropout = rec_weight_dropout

    self.with_ctc = with_ctc
    self.native_ctc = native_ctc
    self.ctc_dropout = ctc_dropout
    self.ctc_l2 = ctc_l2
    self.ctc_opts = ctc_opts
    if not self.ctc_opts:
      self.ctc_opts = {}

    self.network = _NetworkMakerHelper()

  def _create_ff_module(self, prefix_name, i, source):
    """
    Add Feed Forward Module:
      LN -> FFN -> Swish -> Dropout -> FFN -> Dropout

    :param str prefix_name: some prefix name
    :param int i: FF module index
    :param str source: name of source layer
    :return: last layer name of this module
    :rtype: str
    """
    prefix_name = prefix_name + '_ffmod_{}'.format(i)

    ln = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), source)

    ff1 = self.network.add_linear_layer(
      '{}_ff1'.format(prefix_name), ln, n_out=self.ff_dim, l2=self.l2, forward_weights_init=self.ff_init,
      with_bias=self.ff_bias)

    swish_act = self.network.add_activation_layer('{}_swish'.format(prefix_name), ff1, activation=self.activation)

    drop1 = self.network.add_dropout_layer('{}_drop1'.format(prefix_name), swish_act, dropout=self.dropout)

    ff2 = self.network.add_linear_layer(
      '{}_ff2'.format(prefix_name), drop1, n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,
      with_bias=self.ff_bias)

    drop2 = self.network.add_dropout_layer('{}_drop2'.format(prefix_name), ff2, dropout=self.dropout)

    half_step_ff = self.network.add_eval_layer('{}_half_step'.format(prefix_name), drop2, eval='0.5 * source(0)')

    ff_module_res = self.network.add_combine_layer(
      '{}_res'.format(prefix_name), kind='add', source=[half_step_ff, source], n_out=self.enc_key_dim)

    return ff_module_res

  def _create_mhsa_module(self, prefix_name, source):
    """
    Add Multi-Headed Selft-Attention Module:
      LN + MHSA + Dropout

    :param str prefix: some prefix name
    :param str source: name of source layer
    :return: last layer name of this module
    :rtype: str
    """
    prefix_name = '{}_self_att'.format(prefix_name)
    ln = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), source)
    ln_rel_pos_enc = None
    if self.pos_enc == 'rel':
      ln_rel_pos_enc = self.network.add_relative_pos_encoding_layer(
        '{}_ln_rel_pos_enc'.format(prefix_name), ln, n_out=self.enc_key_per_head_dim, forward_weights_init=self.ff_init)
    mhsa = self.network.add_self_att_layer(
      '{}'.format(prefix_name), ln, n_out=self.enc_value_dim, num_heads=self.att_num_heads,
      total_key_dim=self.enc_key_dim, att_dropout=self.att_dropout, forward_weights_init=self.ff_init,
      key_shift=ln_rel_pos_enc if ln_rel_pos_enc is not None else None)
    mhsa_linear = self.network.add_linear_layer(
      '{}_linear'.format(prefix_name), mhsa, n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,
      with_bias=False)
    drop = self.network.add_dropout_layer('{}_dropout'.format(prefix_name), mhsa_linear, dropout=self.dropout)
    mhsa_res = self.network.add_combine_layer(
      '{}_res'.format(prefix_name), kind='add', source=[drop, source], n_out=self.enc_value_dim)
    return mhsa_res

  def _create_convolution_module(self, prefix_name, source):
    """
    Add Convolution Module:
      LN + point-wise-conv + GLU + depth-wise-conv + BN + Swish + point-wise-conv + Dropout

    :param str prefix_name: some prefix name
    :param int i: conformer module index
    :param str source: name of source layer
    :return: last layer name of this module
    :rtype: str
    """
    prefix_name = '{}_conv_mod'.format(prefix_name)

    ln = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), source)

    pointwise_conv1 = self.network.add_linear_layer(
      '{}_pointwise_conv1'.format(prefix_name), ln, n_out=2 * self.enc_key_dim, activation=None, l2=self.l2,
      with_bias=self.ff_bias, forward_weights_init=self.ff_init)

    glu_act = self.network.add_gating_layer('{}_glu'.format(prefix_name), pointwise_conv1)

    depthwise_conv = self.network.add_conv_layer(
      '{}_depthwise_conv2'.format(prefix_name), glu_act, n_out=self.enc_key_dim,
      filter_size=(self.conv_kernel_size,), groups=self.enc_key_dim, l2=self.l2)

    bn = self.network.add_batch_norm_layer('{}_bn'.format(prefix_name), depthwise_conv)

    swish_act = self.network.add_activation_layer('{}_swish'.format(prefix_name), bn, activation='swish')

    pointwise_conv2 = self.network.add_linear_layer(
      '{}_pointwise_conv2'.format(prefix_name), swish_act, n_out=self.enc_key_dim, activation=None, l2=self.l2,
      with_bias=self.ff_bias, forward_weights_init=self.ff_init)

    drop = self.network.add_dropout_layer('{}_drop'.format(prefix_name), pointwise_conv2, dropout=self.dropout)

    res = self.network.add_combine_layer(
      '{}_res'.format(prefix_name), kind='add', source=[drop, source], n_out=self.enc_key_dim)
    return res

  def _create_conformer_block(self, i, source):
    """
    Add the whole Conformer block:
      x1 = x0 + 1/2 * FFN(x0)             (FFN module 1)
      x2 = x1 + MHSA(x1)                  (MHSA)
      x3 = x2 + Conv(x2)                  (Conv module)
      x4 = LayerNorm(x3 + 1/2 * FFN(x3))  (FFN module 2)

    :param int i:
    :param str source: name of source layer
    :return: last layer name of this module
    :rtype: str
    """
    prefix_name = 'conformer_block_%02i' % i
    ff_module1 = self._create_ff_module(prefix_name, 1, source)
    mhsa = self._create_mhsa_module(prefix_name, ff_module1)
    conv_module = self._create_convolution_module(prefix_name, mhsa)
    ff_module2 = self._create_ff_module(prefix_name, 2, conv_module)
    res = ff_module2
    if self.block_final_norm:
      res = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), res)
    res = self.network.add_copy_layer(prefix_name, res)
    return res

  def create_network(self):
    """
    ConvSubsampling/LSTM -> Linear -> Dropout -> [Conformer Blocks] x N
    """
    data = self.source
    if self.use_specaugment:
      data = self.network.add_eval_layer('source', data, eval=specaugment_eval_func)

    subsampled_input = None
    if 'lstm' in self.input_layer:
      sample_factor = int(self.input_layer.split('-')[1])
      pool_sizes = None
      if sample_factor == 4:
        pool_sizes = [2, 2]
      elif sample_factor == 6:
        pool_sizes = [3, 2]
      # add 2 LSTM layers with max pooling to subsample and encode positional information
      subsampled_input = self.network.add_lstm_layers(
          data, num_layers=2, lstm_dim=self.enc_key_dim, dropout=self.lstm_dropout, bidirectional=True,
          rec_weight_dropout=self.rec_weight_dropout, l2=self.l2, pool_sizes=pool_sizes)
    elif self.input_layer == 'conv':
      # subsample by 4
      subsampled_input = self.network.add_conv_block(
        'conv_merged', data, hwpc_sizes=[((3, 3), (2, 2), self.enc_key_dim), ((3, 3), (2, 2), self.enc_key_dim)],
        l2=self.l2, activation='relu')
    elif self.input_layer == 'vgg':
      subsampled_input = self.network.add_conv_block(
        'vgg_conv_merged', data, hwpc_sizes=[((3, 3), (2, 2), 32), ((3, 3), (2, 2), 64)], l2=self.l2, activation='relu')

    assert subsampled_input is not None

    source_linear = self.network.add_linear_layer(
      'source_linear', subsampled_input, n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,
      with_bias=False)

    # add positional encoding
    if self.pos_enc == 'abs':
      source_linear = self.network.add_pos_encoding_layer('{}_abs_pos_enc'.format(subsampled_input), source_linear)

    source_dropout = self.network.add_dropout_layer('source_dropout', source_linear, dropout=self.embed_dropout)

    conformer_block_src = source_dropout
    for i in range(1, self.num_blocks + 1):
      conformer_block_src = self._create_conformer_block(i, conformer_block_src)

    encoder = self.network.add_copy_layer('encoder', conformer_block_src)

    if self.with_ctc:
      default_ctc_loss_opts = {'beam_width': 1}
      if self.native_ctc:
        default_ctc_loss_opts['use_native'] = True
      else:
        self.ctc_opts.update({"ignore_longer_outputs_than_inputs": True})  # always enable
      if self.ctc_opts:
        default_ctc_loss_opts['ctc_opts'] = self.ctc_opts
      self.network.add_softmax_layer(
        'ctc', encoder, l2=self.ctc_l2, target=self.target, loss='ctc', dropout=self.ctc_dropout,
        loss_opts=default_ctc_loss_opts)
    return encoder


class _NetworkMakerHelper:
  """
  Represents a generic RETURNN network
  see docs: https://returnn.readthedocs.io/en/latest/
  """

  def __init__(self):
    self._net = {}

  def get_net(self):
    return self._net

  def add_copy_layer(self, name, source, **kwargs):
    self._net[name] = {'class': 'copy', 'from': source}
    self._net[name].update(kwargs)
    return name

  def add_eval_layer(self, name, source, eval, **kwargs):
    self._net[name] = {'class': 'eval', 'eval': eval, 'from': source}
    self._net[name].update(kwargs)
    return name

  def add_split_dim_layer(self, name, source, axis='F', dims=(-1, 1), **kwargs):
    self._net[name] = {'class': 'split_dims', 'axis': axis, 'dims': dims, 'from': source}
    self._net[name].update(kwargs)
    return name

  def add_conv_layer(self, name, source, filter_size, n_out, l2, padding='same', activation=None, with_bias=True,
                     **kwargs):
    d = {
      'class': 'conv', 'from': source, 'padding': padding, 'filter_size': filter_size, 'n_out': n_out,
      'activation': activation, 'with_bias': with_bias
    }
    if l2:
      d['L2'] = l2
    d.update(kwargs)
    self._net[name] = d
    return name

  def add_linear_layer(self, name, source, n_out, activation=None, with_bias=True, dropout=0., l2=0.,
                       forward_weights_init=None, **kwargs):
    d = {
      'class': 'linear', 'activation': activation, 'with_bias': with_bias, 'from': source, 'n_out': n_out
    }
    if dropout:
      d['dropout'] = dropout
    if l2:
      d['L2'] = l2
    if forward_weights_init:
      d['forward_weights_init'] = forward_weights_init
    d.update(kwargs)
    self._net[name] = d
    return name

  def add_pool_layer(self, name, source, pool_size, mode='max', **kwargs):
    self._net[name] = {'class': 'pool', 'from': source, 'pool_size': pool_size, 'mode': mode, 'trainable': False}
    self._net[name].update(kwargs)
    return name

  def add_merge_dims_layer(self, name, source, axes='static', **kwargs):
    self._net[name] = {'class': 'merge_dims', 'from': source, 'axes': axes}
    self._net[name].update(kwargs)
    return name

  def add_rec_layer(self, name, source, n_out, l2, rec_weight_dropout, direction=1, unit='nativelstm2', **kwargs):
    d = {'class': 'rec', 'unit': unit, 'n_out': n_out, 'direction': direction, 'from': source}
    if l2:
      d['L2'] = l2
    if rec_weight_dropout:
      if 'unit_opts' not in d:
        d['unit_opts'] = {}
      d['unit_opts'].update({'rec_weight_dropout': rec_weight_dropout})
    d.update(kwargs)
    self._net[name] = d
    return name

  def add_choice_layer(self, name, source, target, beam_size=12, initial_output=0, **kwargs):
    self._net[name] = {'class': 'choice', 'target': target, 'beam_size': beam_size, 'from': source,
                      'initial_output': initial_output}
    self._net[name].update(kwargs)
    return name

  def add_compare_layer(self, name, source, value, kind='equal', **kwargs):
    self._net[name] = {'class': 'compare', 'kind': kind, 'from': source, 'value': value}
    self._net[name].update(kwargs)
    return name

  def add_combine_layer(self, name, source, kind, n_out, **kwargs):
    self._net[name] = {'class': 'combine', 'kind': kind, 'from': source, 'n_out': n_out}
    self._net[name].update(kwargs)
    return name

  def add_activation_layer(self, name, source, activation, **kwargs):
    self._net[name] = {'class': 'activation', 'activation': activation, 'from': source}
    self._net[name].update(kwargs)
    return name

  def add_softmax_over_spatial_layer(self, name, source, **kwargs):
    self._net[name] = {'class': 'softmax_over_spatial', 'from': source}
    self._net[name].update(kwargs)
    return name

  def add_generic_att_layer(self, name, weights, base, **kwargs):
    self._net[name] = {'class': 'generic_attention', 'weights': weights, 'base': base}
    self._net[name].update(kwargs)
    return name

  def add_rnn_cell_layer(self, name, source, n_out, unit='LSTMBlock', l2=0., **kwargs):
    d = {'class': 'rnn_cell', 'unit': unit, 'n_out': n_out, 'from': source}
    if l2:
      d['L2'] = l2
    d.update(kwargs)
    self._net[name] = d
    return name

  def add_softmax_layer(self, name, source, l2=None, loss=None, target=None, dropout=0., loss_opts=None,
                        forward_weights_init=None, **kwargs):
    d = {'class': 'softmax', 'from': source}
    if dropout:
      d['dropout'] = dropout
    if target:
      d['target'] = target
    if loss:
      d['loss'] = loss
      if loss_opts:
        d['loss_opts'] = loss_opts
    if l2:
      d['L2'] = l2
    if forward_weights_init:
      d['forward_weights_init'] = forward_weights_init
    d.update(kwargs)
    self._net[name] = d
    return name

  def add_dropout_layer(self, name, source, dropout, dropout_noise_shape=None, **kwargs):
    self._net[name] = {'class': 'dropout', 'from': source, 'dropout': dropout}
    if dropout_noise_shape:
      self._net[name]['dropout_noise_shape'] = dropout_noise_shape
    self._net[name].update(kwargs)
    return name

  def add_reduceout_layer(self, name, source, num_pieces=2, mode='max', **kwargs):
    self._net[name] = {'class': 'reduce_out', 'from': source, 'num_pieces': num_pieces, 'mode': mode}
    self._net[name].update(kwargs)
    return name

  def add_subnet_rec_layer(self, name, unit, target, source=None, **kwargs):
    if source is None:
      source = []
    self._net[name] = {
      'class': 'rec', 'from': source, 'unit': unit, 'target': target, 'max_seq_len': "max_len_from('base:encoder')"}
    self._net[name].update(kwargs)
    return name

  def add_decide_layer(self, name, source, target, loss='edit_distance', **kwargs):
    self._net[name] = {'class': 'decide', 'from': source, 'loss': loss, 'target': target}
    self._net[name].update(kwargs)
    return name

  def add_slice_layer(self, name, source, axis, **kwargs):
    self._net[name] = {'class': 'slice', 'from': source, 'axis': axis, **kwargs}
    return name

  def add_subnetwork(self, name, source, subnetwork_net, **kwargs):
    self._net[name] = {'class': 'subnetwork', 'from': source, 'subnetwork': subnetwork_net, **kwargs}
    return name

  def add_layer_norm_layer(self, name, source, **kwargs):
    self._net[name] = {'class': 'layer_norm', 'from': source, **kwargs}
    return name

  def add_batch_norm_layer(self, name, source, **kwargs):
    self._net[name] = {'class': 'batch_norm', 'from': source, **kwargs}
    return name

  def add_self_att_layer(self, name, source, n_out, num_heads, total_key_dim, att_dropout=0., key_shift=None,
                         forward_weights_init=None, **kwargs):
    d = {
      'class': 'self_attention', 'from': source, 'n_out': n_out, 'num_heads': num_heads, 'total_key_dim': total_key_dim
    }
    if att_dropout:
      d['attention_dropout'] = att_dropout
    if key_shift:
      d['key_shift'] = key_shift
    if forward_weights_init:
      d['forward_weights_init'] = forward_weights_init
    d.update(kwargs)
    self._net[name] = d
    return name

  def add_pos_encoding_layer(self, name, source, add_to_input=True, **kwargs):
    self._net[name] = {'class': 'positional_encoding', 'from': source, 'add_to_input': add_to_input}
    self._net[name].update(kwargs)
    return name

  def add_relative_pos_encoding_layer(self, name, source, n_out, forward_weights_init=None, **kwargs):
    self._net[name] = {'class': 'relative_positional_encoding', 'from': source, 'n_out': n_out}
    if forward_weights_init:
      self._net[name]['forward_weights_init'] = forward_weights_init
    self._net[name].update(kwargs)
    return name

  def add_constant_layer(self, name, value, **kwargs):
    self._net[name] = {'class': 'constant', 'value': value}
    self._net[name].update(kwargs)
    return name

  def add_gating_layer(self, name, source, activation='identity', **kwargs):
    """
    out = activation(a) * gate_activation(b)  (gate_activation is sigmoid by default)
    In case of one source input, it will split by 2 over the feature dimension
    """
    self._net[name] = {'class': 'gating', 'from': source, 'activation': activation}
    self._net[name].update(kwargs)
    return name

  def add_pad_layer(self, name, source, axes, padding, **kwargs):
    self._net[name] = {'class': 'pad', 'from': source, 'axes': axes, 'padding': padding}
    self._net[name].update(**kwargs)
    return name

  def add_conv_block(self, name, source, hwpc_sizes, l2, activation):
    src = self.add_split_dim_layer('source0', source)
    for idx, hwpc in enumerate(hwpc_sizes):
      filter_size, pool_size, n_out = hwpc
      src = self.add_conv_layer('conv%i' % idx, src, filter_size=filter_size, n_out=n_out, l2=l2, activation=activation)
      if pool_size:
        src = self.add_pool_layer('conv%ip' % idx, src, pool_size=pool_size, padding='same')
    return self.add_merge_dims_layer(name, src)

  def add_lstm_layers(self, input, num_layers, lstm_dim, dropout, l2, rec_weight_dropout, pool_sizes,  bidirectional):
    src = input
    pool_idx = 0
    for layer in range(num_layers):
      lstm_fw_name = self.add_rec_layer(
        name='lstm%i_fw' % layer, source=src, n_out=lstm_dim, direction=1, dropout=dropout, l2=l2,
        rec_weight_dropout=rec_weight_dropout)
      if bidirectional:
        lstm_bw_name = self.add_rec_layer(
          name='lstm%i_bw' % layer, source=src, n_out=lstm_dim, direction=-1, dropout=dropout, l2=l2,
          rec_weight_dropout=rec_weight_dropout)
        src = [lstm_fw_name, lstm_bw_name]
      else:
        src = lstm_fw_name
      if pool_sizes and pool_idx < len(pool_sizes):
        lstm_pool_name = 'lstm%i_pool' % layer
        src = self.add_pool_layer(
          name=lstm_pool_name, source=src, pool_size=(pool_sizes[pool_idx],), padding='same')
        pool_idx += 1
    return src

  def add_dot_layer(self, name, source, **kwargs):
    self._net[name] = {'class': 'dot', 'from': source}
    self._net[name].update(kwargs)
    return name

  def __setitem__(self, key, value):
    self._net[key] = value

  def __getitem__(self, item):
    return self._net[item]

  def update(self, d: dict):
    self._net.update(d)

  def __str__(self):
    """
    Only for debugging
    """
    res = 'network = {\n'
    for k, v in self._net.items():
      res += '%s: %r\n' % (k, v)
    return res + '}'
