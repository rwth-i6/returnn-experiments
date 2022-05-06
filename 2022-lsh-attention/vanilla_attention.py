from generic import _make_cross_attention_qkv, normalize_eval, _query_key_time_default
from returnn.tf.util.data import DimensionTag


def add_vanilla_self_attention_layer(
  d, input, output, inside_rec_layer=True, past_only=None, time_axis=None,
  num_heads=8, key_dim=64, value_dim=64, dropout=0.0,
  ff_init = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0,
  share_key_query=False, normalize_keys=None,
  mask_current=False, mask_current_value=float(-10**5)):
  """
  Essentially this does
    d[output + '_att'] = {"class": "self_attention", "num_heads": num_heads,
      "total_key_dim": num_heads * key_dim,
      "n_out": num_heads * value_dim, "from": [input],
      "attention_left_only": past_only,
      "attention_dropout": dropout, "forward_weights_init": self.ff_init}
  But using multiple layers that can be extended on
  """
  if past_only is None:
    past_only = inside_rec_layer
  if time_axis is None:
    time_axis = 'stag:extern_data:classes' if inside_rec_layer else 'stag:extern_data:data'
  assert time_axis.startswith('stag:')
  assert not inside_rec_layer or past_only
  if normalize_keys is None:
    normalize_keys = share_key_query

  # Create (non-accumulated) query, key and value
  if not share_key_query:
    d[output + '_qkv0'] = {
      'class': 'linear', 'activation': None, 'with_bias': False, 'from': [input],
      'n_out': num_heads * (2 * key_dim + value_dim), 'forward_weights_init': ff_init}  # [B,T?,F|n*(2d_k+d_v)]
    d[output + '_qkv'] = {
      'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, 2 * key_dim + value_dim),
      'from': [output + '_qkv0']}  # [B,T?,n,F|2d_k+d_v]
    d[output + '_qkv_split'] = {
      'class': 'split', 'axis': 'F', 'size_splits': (key_dim, key_dim, value_dim),
      'from': [output + '_qkv']}
    d[output + '_query'] = {
      'class': 'copy', 'from': [output + '_qkv_split/0']}  # [B,T?,n,F|d_k]
    if normalize_keys:
      d[output + '_key'] = {
        'class': 'eval', 'eval': normalize_eval, 'from': [output + '_qkv_split/1']}  # [B,T?,n,F|d_k]
    else:
      d[output + '_key'] = {
        'class': 'copy', 'from': [output + '_qkv_split/1']}  # [B,T?,n,F|d_k]
    d[output + '_value'] = {
      'class': 'copy', 'from': [output + '_qkv_split/2']}  # [B,T?,n,F|d_v]
  else:  # share_key_query
    d[output + '_qv0'] = {
      'class': 'linear', 'activation': None, 'with_bias': False, 'from': [input],
      'n_out': num_heads * (key_dim + value_dim), 'forward_weights_init': ff_init}  # [B,T?,F|n*(d_k+d_v)]
    d[output + '_qv'] = {
      'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, key_dim + value_dim),
      'from': [output + '_qv0']}  # [B,T?,n,F|d_k+d_v]
    d[output + '_qv_split'] = {
      'class': 'split', 'axis': 'F', 'size_splits': (key_dim, value_dim),
      'from': [output + '_qv']}
    d[output + '_query'] = {
      'class': 'copy', 'from': [output + '_qv_split/0']}  # [B,T?,n,F|d_k]
    if normalize_keys:
      d[output + '_key'] = {
        'class': 'eval', 'eval': normalize_eval, 'from': [output + '_query']}  # [B,T?,n,F|d_k]
    else:
      d[output + '_key'] = {'class': 'copy', 'from': [output + '_query']}  # [B,T?,n,F|d_k]
    d[output + '_value'] = {
      'class': 'copy', 'from': [output + '_qv_split/1']}  # [B,T?,n,F|d_v]

  # Accumulate keys/values or rename the axis
  if inside_rec_layer:
    d[output + '_key_accum'] = {
      'class': 'cum_concat', 'from': [output + '_key']}  # [B,T|rec-history,n,F|d_k]
    d[output + '_value_accum'] = {
      'class': 'cum_concat', 'from': [output + '_value']}  # [B,T|rec-history,n,F|d_v]
    key_axis = 'stag:rec-history'
  else:
    key_dim_tag = DimensionTag(kind=DimensionTag.Types.Time, description='self-att-keys')
    d[output + '_key_accum'] = {
      'class': 'reinterpret_data', 'set_dim_tags': {time_axis: key_dim_tag},
      'from': [output + '_key']}  # [B,T|keys,n,F|d_k]
    d[output + '_value_accum'] = {
      'class': 'reinterpret_data', 'set_dim_tags': {time_axis: key_dim_tag},
      'from': [output + '_value']}  # [B,T|keys,n,F|d_v]
    key_axis = 'stag:' + key_dim_tag.description

  # Calculate the energies
  d[output + '_energy'] = {
    'class': 'dot', 'from': [output + '_query', output + '_key_accum'],
    'red1': 'static:-1', 'red2': 'static:-1', 'var1': time_axis + '?', 'var2': key_axis}  # [B,n,T?,T|rec-history]

  need_indices = past_only or mask_current
  if need_indices:
    if inside_rec_layer:
      query_indices_from = ':i'
    else:
      d[output + '_query_indices'] = {'class': 'range_in_axis', 'from': [input], 'axis': time_axis,
        'keepdims': False}  # [T]
      query_indices_from = output + '_query_indices'
    d[output + '_key_accum_indices'] = {
      'class': 'range_in_axis', 'from': [output + '_key_accum'], 'axis': key_axis,
      'keepdims': False}  # [T|rec-history]
  if past_only:
    d[output + '_energy_unmasked'] = d[output + '_energy']
    d[output + '_energy_mask'] = {
      'class': 'compare', 'kind': 'greater_equal', 'from': [query_indices_from, output + '_key_accum_indices']}
    d[output + '_energy'] = {
      'class': 'switch', 'true_from': output + '_energy_unmasked', 'false_from': float('-inf'),
      'condition': output + '_energy_mask'}  # [B,n,T?,T|rec-history]
  if mask_current:
    d[output + '_energy_unmasked_current'] = d[output + '_energy']
    d[output + '_energy_mask_current'] = {
      'class': 'compare', 'kind': 'equal', 'from': [query_indices_from, output + '_key_accum_indices']}
    d[output + '_energy'] = {
      'class': 'switch', 'true_from': mask_current_value, 'false_from': output + '_energy_unmasked_current',
      'condition': output + '_energy_mask_current'}  # [B,n,T?,T|rec-history]

  # If past_only=True, do not apply a time mask here, as we apply our own masking using energy_mask.
  # If we would apply additional masking here, we would mask away all keys for queries that are unmasked, giving
  # attention weights NaN for these queries. Even though these are masked away later in the forward pass, the gradient
  # can still become NaN.
  # If past_only=False, do apply the normal time mask.
  d[output + '_weights'] = {
    'class': 'softmax_over_spatial', 'from': [output + '_energy'], 'axis': key_axis,
    'energy_factor': key_dim ** -0.5,
    'use_time_mask': not past_only}  # [B,n,T?,T|rec-history]
  d[output + '_weights_drop'] = {
    'class': 'dropout', 'dropout_noise_shape': {'*': None}, 'from': [output + '_weights'],
    'dropout': dropout}  # [B,n,T?,T|rec-history]

  d[output + '_output'] = {
    'class': 'dot', 'from': [output + '_weights_drop', output + '_value_accum'],
    'red1': key_axis, 'red2': key_axis, 'var1': time_axis + '?', 'var2': 'static:-1'}  # [B,n,T?,F|d_v]
  d[output + '_att'] = {
    'class': 'merge_dims', 'axes': 'static', 'from': [output + '_output']}  # [B,T?,F|n*d_v]


def add_vanilla_cross_attention_layer(
  d, db, input, keys_input, output, query_time_axis=None, key_time_axis=None,
  num_heads=8, key_dim=64, value_dim=64, dropout=0.0,
  ff_init = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0):
  """
  Add a cross-attention layer.

  :param dict[str, Any] d:
  :param dict[str, Any] db:
  :param str input:
  :param str keys_input:
  :param str output:
  :param None|str query_time_axis:
  :param None|str key_time_axis:
  :param int num_heads:
  :param int key_dim:
  :param int value_dim:
  :param float dropout:
  :param str ff_init:
  """
  query_time_axis, key_time_axis = _query_key_time_default(query_time_axis, key_time_axis)

  assert keys_input.startswith('base:')
  keys_input = keys_input[len('base:'):]

  # Create query, key and value
  _make_cross_attention_qkv(
    d=d, db=db, input=input, keys_input=keys_input, output=output, num_heads=num_heads, key_dim=key_dim,
    value_dim=value_dim, ff_init=ff_init)

  # Calculate the energies + weights
  d[output + '_energy'] = {
    'class': 'dot', 'from': [output + '_query', 'base:' + output + '_key'], 'red1': 'static:-1', 'red2': 'static:-1',
    'var1': query_time_axis + '?', 'var2': key_time_axis}  # [B,n,query-T?,key-T]
  d[output + '_weights'] = {
    'class': 'softmax_over_spatial', 'from': [output + '_energy'], 'axis': key_time_axis,
    'energy_factor': key_dim ** -0.5,
    'use_time_mask': True}  # [B,n,query-T?,key-T]
  d[output + '_weights_drop'] = {
    'class': 'dropout', 'dropout_noise_shape': {'*': None}, 'from': [output + '_weights'],
    'dropout': dropout}  # [B,n,query-T?,key-T]

  d[output + '_output_named'] = {
    'class': 'dot', 'from': [output + '_weights_drop', 'base:' + output + '_value'], 'red1': key_time_axis,
    'red2': key_time_axis, 'var1': query_time_axis + '?', 'var2': 'static:-1'}  # [B,n,query-T?,d_v]
  d[output + '_output'] = {
    'class': 'name_axis', 'from': [output + '_output_named'], 'axis': 'stag:att-heads',
    'description': None}  # [B,n,query-T?,d_v]
  d[output + '_att'] = {
    'class': 'merge_dims', 'axes': 'static',
    'from': [output + '_output']}  # [B,query-T?,F|n*d_v]

  # there is a bug in HDF dump layer when naming static dimensions. Expose this here to extract the att weights.
  d[output + '_weights_unnamed'] = {
    'class': 'name_axis', 'from': [output + '_weights'], 'axis': 'stag:att-heads',
    'description': None}  # [B,n,query-T?,key-T]
