class ReturnnNetwork:
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
