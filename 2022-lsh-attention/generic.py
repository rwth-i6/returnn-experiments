import tensorflow as tf

from returnn.tf.layers.basic import register_layer_class, _ConcatInputLayer
from returnn.tf.util.data import Data, DimensionTag


class NameAxisLayer(_ConcatInputLayer):
  """
  Adds a DimensionTag to an axis s.t. it will be unique.
  """
  layer_class = "name_axis"

  def __init__(self, axis, description, **kwargs):
    super(NameAxisLayer, self).__init__(**kwargs)

    # Maybe we still need to unbroadcast a size_placeholder
    # As the output does not necessarily have a batch dim, but the size_placeholder still needs a batch dim,
    # we use the global batch dim here.
    from returnn.tf.layers.base import LayerBase
    batch_dim = LayerBase.get_recent_layer().get_batch_info().dim
    for i, dyn_size in self.output.size_placeholder.items():
      if len(dyn_size.shape) == 0 or dyn_size.shape[0] == 1:
        dim_tag = DimensionTag.get_tag_from_size_tensor(dyn_size)
        new_dyn_size = tf.broadcast_to(dyn_size, [batch_dim])
        dim_tag.set_tag_on_size_tensor(new_dyn_size)
        dim_tag.dyn_size = new_dyn_size  # override this explicitly: dim_tag.set_tag_on_size_tensor does not reset it.
        self.output.size_placeholder[i] = new_dyn_size

  @classmethod
  def get_out_data_from_opts(cls, name, axis, description, sources, **kwargs):
    """
    :param str name:
    :param str|int|list[str|int]|tuple[str|int] axis:
    :param str|None|list[str|None]|tuple[str|None] description:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    data = Data.get_common_data([s.output for s in sources])
    data = data.copy(name="%s_output" % name)

    if not isinstance(axis, (list, tuple)):
      axis = [axis]
    if not isinstance(description, (list, tuple)):
      description = [description]
    assert len(axis) == len(description)

    for ax, descr in zip(axis, description):
      if isinstance(ax, int):
        data = data.copy_as_batch_major()
      if isinstance(ax, str) and '|' in ax:
        possible_axes = ax.split('|')
        found_ax = None
        for possible_ax in possible_axes:
          try:
            found_ax = data.get_axis_from_description(possible_ax)
            break
          except:
            continue
        assert found_ax is not None, '%r: axis %r not found in %r' % (cls, ax, data)
        ax = found_ax
      if isinstance(ax, str) and len(ax) >= 3 and ax[-2] == '+':
        ax_offset = int(ax[-1])
        ax = ax[:-2]
      else:
        ax_offset = 0
      ax = data.get_axis_from_description(ax, allow_int=True) + ax_offset
      ax_wo_batch = data.get_batch_axis_excluding_batch(ax)
      if descr is None:
        del data.size_placeholder[ax_wo_batch]
      else:
        if ax_wo_batch in data.size_placeholder:
          dyn_size = tf.identity(data.size_placeholder[ax_wo_batch])
        else:
          assert data.batch_shape[ax] is not None
          # this must actually be a [B]-tensor, but here it is not. we fix that later when we actually now the
          # placeholder (with the size we need to unbroadcast to)
          dyn_size = tf.constant(data.batch_shape[ax], shape=(1,))
        from returnn.tf.util.basic import DimensionTag
        tag = DimensionTag(
          description=descr,
          kind=DimensionTag.Types.Time)
        data.size_placeholder[ax_wo_batch] = dyn_size
        tag.set_tag_on_size_tensor(dyn_size)
    return data


register_layer_class(NameAxisLayer)


def _query_key_time_default(query_time_axis, key_time_axis):
  """
  :param None|str query_time_axis:
  :param None|str key_time_axis:
  :rtype: tuple[str,str]
  """
  assert (query_time_axis is None) == (key_time_axis is None)
  if query_time_axis is None:
    query_time_axis = 'stag:extern_data:classes'
    key_time_axis = 'stag:extern_data:data'
  assert query_time_axis.startswith('stag:')
  assert key_time_axis.startswith('stag:')
  return query_time_axis, key_time_axis


normalize_eval = 'tf.math.divide_no_nan(source(0), tf.norm(source(0), axis=source(0, as_data=True).feature_dim_axis, ' \
                 'keepdims=True))'

argsort_eval = 'tf.argsort(source(0), axis=source(0, as_data=True).get_axis_from_description("%s"), ' \
               'direction="ASCENDING", stable=True)'

clip_eval = 'tf.where(tf.equal(source(0), mask_value), 0, source(0))'


def make_lsh_hash_gen(d, output, key_dim, num_hashes, num_heads, num_rounds,
                      hash_init="variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0):
  """
  :param dict[str,dict] d: the network dict to write into
  :param str output: prefix of all layers generated. Output is written into output + '_hash_gen' layer.
  :param int key_dim:
  :param int num_hashes:
  :param int num_heads:
  :param int num_rounds:
  :param str hash_init: initializer for the hash generator matrix
  """
  assert num_hashes % 2 == 0
  d[output + '_top_unnamed'] = {
    'class': 'variable', 'shape': (num_heads, num_rounds, key_dim, num_hashes // 2),
    'trainable': False, 'init': hash_init, 'add_batch_axis': True}  # [B,n,r,d_k,F|d_h/2]
  d[output + '_top'] = {
    'class': 'name_axis', 'axis': ['static:0', 'static:1'], 'description': ['att-heads', 'att-rounds'],
    'from': [output + '_top_unnamed']}  # [B,n,r,d_k,F|d_h/2]
  d[output + '_bottom'] = {
    'class': 'eval', 'eval': '-source(0)',
    'from': [output + '_top']}  # [B,n,r,d_k,F|d_h/2]
  d[output] = {
    'class': 'copy',
    'from': [output + '_top', output + '_bottom']}  # [B,n,r,d_k,F|d_h]


def apply_lsh_hash_gen(d, input, hash_gen_input, output, num_hashes, time_axis, hash_mask_value=2 ** 31 - 1,
                       hash_dropin=0.0):
  """
  :param dict[str,dict] d:
  :param str input:
  :param str hash_gen_input:
  :param str output:
  :param int num_hashes:
  :param str time_axis:
  :param int|None hash_mask_value: or None if you do not want masking
  :param float hash_dropin:
  """
  d[output + '_linear'] = {
    'class': 'dot', 'from': [hash_gen_input, input], 'debug': True,
    'red1': 'static:-2', 'red2': 'F', 'var1': ['stag:att-rounds', 'static:-1'],
    'var2': time_axis + '?', 'add_var2_if_empty': False}  # [B,T|classes?,n,r,F|d_h]
  d[output + '_sparse'] = {
    'class': 'reduce', 'mode': 'argmax', 'axes': 'static:-1',
    'from': [output + '_linear']}  # [B,T|classes?,n,r] :: d_h
  d[output + '_actual'] = {
    'class': 'reinterpret_data', 'from': [output + '_sparse'],
    'set_sparse': False, 'set_axes': {'F': None}}  # [B,T|classes?,n,r] :: d_h
  # DropoutLayer does not support inputs that are not of type float.
  d[output + '_dropin_decision_ones'] = {
    'class': 'eval', 'from': [output + '_actual'], 'eval': 'tf.ones_like(source(0), dtype="float32")',
    'out_type': {'dtype': 'float32'}}  # [B,T|classes?,n,r] :: 1.0
  d[output + '_dropin_decision_float'] = {
    'class': 'dropout', 'dropout': hash_dropin, 'dropout_noise_shape': {'B': -1, 'except_time': -1, 'T': 1},
    'from': [output + '_dropin_decision_ones']}  # [B,T|classes?,n,r] :: 0.0/1.0
  d[output + '_dropin_decision'] = {
    'class': 'compare', 'from': [output + '_dropin_decision_float'], 'kind': 'greater',
    'value': 0.5}  # [B,T|classes?,n,r] :: False/True
  d[output + '_dropin_hashes'] = {
    'class': 'eval',
    'eval': 'tf.random.uniform(tf.shape(source(0)), minval=0, maxval=%s, dtype="int32")' % num_hashes,
    'from': [output + '_actual'], 'out_type': {'dtype': 'int32'}}  # [B,T|classes?,n,r] :: d_h
  d[output + '_unmasked'] = {
    'class': 'switch', 'condition': output + '_dropin_decision', 'true_from': output + '_actual',
    'false_from': output + '_dropin_hashes'}  # [B,T|classes?,n,r] :: d_h
  if hash_mask_value is not None:
    d[output] = {
      'class': 'seq_len_mask', 'from': [output + '_unmasked'], 'axis': time_axis,
      'mask_value': hash_mask_value}  # [B,T|classes?,n,r] :: d_h
  else:
    d[output] = {'class': 'copy', 'from': [output + '_unmasked']}  # [B,T|classes?,n,r] :: d_h


def _make_cross_attention_qkv(
  d, db, input, keys_input, output, num_heads=8, key_dim=64, value_dim=64,
  ff_init="variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0):
  d[output + '_query0'] = {
    'class': 'linear', 'activation': None, 'with_bias': False, 'from': [input],
    'n_out': num_heads * key_dim, 'forward_weights_init': ff_init}  # [B,query-T?,F|n*d_k]
  db[output + '_key0'] = {
    'class': 'linear', 'activation': None, 'with_bias': False, 'from': [keys_input],
    'n_out': num_heads * key_dim, 'forward_weights_init': ff_init}  # [B,key-T,F|n*d_k]
  db[output + '_value0'] = {
    'class': 'linear', 'activation': None, 'with_bias': False, 'from': [keys_input],
    'n_out': num_heads * value_dim, 'forward_weights_init': ff_init}  # [B,key-T,F|n*d_v]
  d[output + '_query_unnamed'] = {
    'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, key_dim),
    'from': [output + '_query0']}  # [B,query-T?,n,F|d_k]
  db[output + '_key_unnamed'] = {
    'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, key_dim),
    'from': [output + '_key0']}  # [B,key-T,n,F|d_k]
  db[output + '_value_unnamed'] = {
    'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, value_dim),
    'from': [output + '_value0']}  # [B,key-T,n,F|d_v]
  d[output + '_query'] = {
    'class': 'name_axis', 'axis': 'static:-2', 'description': 'att-heads',
    'from': [output + '_query_unnamed']}  # [B,query-T?,n,F|d_k]
  db[output + '_key'] = {
    'class': 'name_axis', 'axis': 'static:-2', 'description': 'att-heads',
    'from': [output + '_key_unnamed']}  # [B,key-T,n,F|d_k]
  db[output + '_value'] = {
    'class': 'name_axis', 'axis': 'static:-2', 'description': 'att-heads',
    'from': [output + '_value_unnamed']}  # [B,key-T,n,F|d_v]
