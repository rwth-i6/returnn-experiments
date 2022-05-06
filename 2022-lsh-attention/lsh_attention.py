from generic import make_lsh_hash_gen, apply_lsh_hash_gen, argsort_eval, normalize_eval, clip_eval, \
  _query_key_time_default, _make_cross_attention_qkv
from vanilla_attention import add_vanilla_cross_attention_layer


def generic_add_lsh_attention_layer(
  d, queries_input, keys_input, values_input, output, *,
  query_time_axis, key_time_axis, num_heads=8, num_rounds=1, key_dim=64, value_dim=64, dropout=0.0,
  num_hashes, query_chunk_size, key_chunk_size, key_chunks_before=None, key_chunks_after=None,
  hash_init="variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0,
  small_mask_value=float(-10**5),
  past_only=None, mask_current=None, mask_different_hashes=True, allow_duplicate_attention=False,
  chunk_alignment, shuffle_kv=False, query_hash_dropin=0.0, key_hash_dropin=0.0, debug_print=False):
  """
  Computes LSH attention for an entire sequence.

  :param dict[str,dict] d:
  :param str queries_input:
  :param str keys_input:
  :param str values_input:
  :param str output:
  :param str query_time_axis:
  :param str key_time_axis: key and value time axis
  :param int num_heads:
  :param int num_rounds:
  :param int key_dim:
  :param int value_dim:
  :param float dropout:
  :param int num_hashes:
  :param int query_chunk_size:
  :param int key_chunk_size:
  :param int|None key_chunks_before:
  :param int|None key_chunks_after:
  :param str hash_init: for hash generator matrices
  :param float small_mask_value:
  :param None|bool past_only: for self attention
  :param None|bool mask_current: for self attention
  :param bool mask_different_hashes:
  :param bool allow_duplicate_attention:
  :param str chunk_alignment:
  :param bool shuffle_kv:
  :param float query_hash_dropin: chance of assigning a random hash to a query
  :param float key_hash_dropin: chance of assigning a random hash to a key
  :param bool debug_print:
  """
  assert query_time_axis.startswith('stag:') and key_time_axis.startswith('stag:')
  self_attention = query_time_axis == key_time_axis
  assert self_attention == (past_only is not None) == (mask_current is not None)
  if key_chunks_before is None:
    key_chunks_before = 1
  if key_chunks_after is None:
    key_chunks_after = 0 if past_only and chunk_alignment == 'identity' else 1
  assert key_chunks_before >= 0 and key_chunks_after >= 0
  hash_mask_value = 2 ** 31 - 1
  assert hash_mask_value > num_hashes
  assert chunk_alignment in {'identity', 'search_bounds_centered'}
  assert small_mask_value < 0
  even_smaller_mask_value = small_mask_value * 10**5

  def chunk_query_sequence(name, pad_value, have_feature_dim=False):
    """
    :param str name:
    :param float pad_value:
    :param bool have_feature_dim:
    """
    d[output + '_sorted_chunked_%s_%s' % (name, 'unnamed' if have_feature_dim else 'feature')] = {
      'class': 'split_dims', 'from': [output + '_sorted_%s' % name], 'pad_value': pad_value,
      'axis': query_time_axis, 'dims': [-1, query_chunk_size]}  # [B,n,r,query-chunk,query-window,F] :: query-time
    if not have_feature_dim:
      d[output + '_sorted_chunked_%s_unnamed' % name] = {
        'class': 'reinterpret_data', 'from': [output + '_sorted_chunked_%s_feature' % name],
        'set_axes': {'F': None}}  # [B,n,r,query-chunk,query-window,F] :: query-time
    d[output + '_sorted_chunked_%s' % name] = {
      'class': 'name_axis', 'axis': ['T', 'T+1'], 'description': ['query-chunk', 'query-window'],
      'from': [
        output + '_sorted_chunked_%s_unnamed' % name]}  # [B,n,r,query-chunk,query-window,F] :: query-time

  def chunk_key_sequence(name, pad_value, have_feature_dim=False):
    """
    :param str name:
    :param float pad_value:
    :param bool have_feature_dim:
    """
    d[output + '_sorted_chunked_%s_%s' % (name, 'unnamed' if have_feature_dim else 'feature')] = {
      'class': 'split_dims', 'from': [output + '_sorted_%s' % name], 'pad_value': pad_value,
      'axis': key_time_axis, 'dims': [-1, key_chunk_size]}  # [B,n,r,key-chunk,key-window,F] :: key-time
    if not have_feature_dim:
      d[output + '_sorted_chunked_%s_unnamed' % name] = {
        'class': 'reinterpret_data', 'from': [output + '_sorted_chunked_%s_feature' % name],
        'set_axes': {'F': None}}  # [B,n,r,key-chunk,key-window,F] :: key-time
    d[output + '_sorted_chunked_%s' % name] = {
      'class': 'name_axis', 'axis': ['T', 'T+1'], 'description': ['key-chunk', 'key-window'],
      'from': [output + '_sorted_chunked_%s_unnamed' % name]}  # [B,n,r,key-chunk,key-window,F] :: key-time

  def stack_chunked_key_sequence(name):
    """
    :param str name:
    """
    d[output + '_sorted_chunked_stacked_%s_unflattened' % name] = {
      'class': 'gather', 'from': [output + '_sorted_chunked_%s' % name], 'position': output + '_query_chunk_alignment',
      'axis': 'stag:key-chunk'}  # [B,n,r,query-chunk,key-chunk-offset,key-window,F]
    d[output + '_sorted_chunked_stacked_%s_unnamed' % name] = {
      'class': 'merge_dims', 'from': [output + '_sorted_chunked_stacked_%s_unflattened' % name], 'keep_order': True,
      'axes': ['stag:key-chunk-offset', 'stag:key-window']}  # [B,n,r,query-chunk,key-chunk-offset*key-window,F]
    d[output + '_sorted_chunked_stacked_%s' % name] = {
      'class': 'name_axis', 'from': [output + '_sorted_chunked_stacked_%s_unnamed' % name],
      'axis': 'stag:query-chunk+1',
      'description': 'stacked-key-window'}  # [B,n,r,query-chunk,stacked-key-window,F]

  # Receive sequence positions
  d[output + '_keys_all_indices'] = {
    'class': 'range_in_axis', 'from': [keys_input], 'axis': key_time_axis,
    'keepdims': False}  # [key-time] :: key-time
  d[output + '_queries_all_indices'] = {
    'class': 'range_in_axis', 'from': [queries_input], 'axis': query_time_axis,
    'keepdims': False}  # [query-time] :: query-time

  # Hash the queries and keys
  make_lsh_hash_gen(
    d, output + '_hash_gen', key_dim=key_dim, num_hashes=num_hashes, num_heads=num_heads, num_rounds=num_rounds,
    hash_init=hash_init)  # [B,n,r,d_k,F|d_h]
  for neg, mask_value in [('', hash_mask_value), ('_neg_mask', -hash_mask_value)]:
    apply_lsh_hash_gen(
      d, input=queries_input, hash_gen_input=output + '_hash_gen', output=output + '_queries_hashed%s' % neg,
      time_axis=query_time_axis, num_hashes=num_hashes, hash_mask_value=mask_value,
      hash_dropin=query_hash_dropin)  # [B,query-time,n,r] :: d_h
    apply_lsh_hash_gen(
      d, input=keys_input, hash_gen_input=output + '_hash_gen', output=output + '_keys_hashed%s' % neg,
      time_axis=key_time_axis, num_hashes=num_hashes, hash_mask_value=mask_value,
      hash_dropin=key_hash_dropin)  # [B,key-time,n,r] :: d_h

  # Compute a permutation by looking at the unsorted hashes
  d[output + '_sorted_queries_orig_indices'] = {
    'class': 'eval', 'eval': argsort_eval % query_time_axis,
    'from': [output + '_queries_hashed']}  # [B,sorted-query-time,n,r] :: query-time
  if shuffle_kv:
    d[output + '_shuffled_keys_orig_indices'] = {
      'class': 'eval', 'from': [output + '_keys_all_indices'],
      'eval': 'tf.random.shuffle(source(0))'}  # [shuffled-key-time] :: key-time
    d[output + '_shuffled_keys_hashed'] = {
      'class': 'gather', 'from': [output + '_keys_hashed'],
      'position': output + '_shuffled_keys_orig_indices', 'axis': key_time_axis}  # [B,shuffled-key-time,n,r] :: d_h
    d[output + '_sorted_keys_shuffled_indices'] = {
      'class': 'eval', 'eval': argsort_eval % key_time_axis,
      'from': [output + '_shuffled_keys_hashed']}  # [B,sorted-key-time,n,r] :: shuffled-key-time
    d[output + '_sorted_keys_orig_indices'] = {
      'class': 'gather', 'from': [output + '_shuffled_keys_orig_indices'],
      'position': output + '_sorted_keys_shuffled_indices',
      'axis': key_time_axis}  # [B,sorted-key-time,n,r] :: key-time
  else:
    d[output + '_sorted_keys_orig_indices'] = {
      'class': 'eval', 'eval': argsort_eval % key_time_axis,
      'from': [output + '_keys_hashed']}  # [B,sorted-key-time,n,r] :: key-time
  chunk_query_sequence('queries_orig_indices', pad_value=hash_mask_value)  # [B,n,r,query-chunk,query-window] :: query-time  # noqa
  chunk_key_sequence('keys_orig_indices', pad_value=hash_mask_value)  # [B,n,r,key-chunk,key-window] :: key-time
  stack_chunked_key_sequence('keys_orig_indices')  # [B,n,r,query-chunk,stacked-key-window] :: key-time
  # clip to avoid index error on CPU when accessing masked away positions
  d[output + '_sorted_chunked_queries_orig_indices_clipped'] = {
    'class': 'eval', 'from': [output + '_sorted_chunked_queries_orig_indices'],
    'eval': clip_eval, 'eval_locals': {'mask_value': hash_mask_value}}  # [B,n,r,query-chunk,query-window] :: d_h
  d[output + '_sorted_chunked_stacked_keys_orig_indices_clipped'] = {
    'class': 'eval', 'from': [output + '_sorted_chunked_stacked_keys_orig_indices'],
    'eval': clip_eval, 'eval_locals': {'mask_value': hash_mask_value}}  # [B,n,r,query-chunk,stacked-key-window] :: d_h

  # Invert permutation to undo sorting later
  d[output + '_queries_sort_indices'] = {
    'class': 'scatter_nd', 'from': [output + '_queries_all_indices'],
    'position': output + '_sorted_queries_orig_indices', 'position_axis': query_time_axis,
    'output_dim_via_time_from': output + '_sorted_queries_orig_indices'}  # [B,n,r,query-time] :: sorted-query-time
  # note: this is not needed for training/search, but our score job uses this to reconstruct the attention weights.
  d[output + '_keys_sort_indices'] = {
    'class': 'scatter_nd', 'from': [output + '_keys_all_indices'],
    'position': output + '_sorted_keys_orig_indices', 'position_axis': key_time_axis,
    'output_dim_via_time_from': output + '_sorted_keys_orig_indices'}  # [B,n,r,key-time] :: sorted-key-time

  # Sort hashes themselves
  d[output + '_sorted_queries_hashed'] = {
    'class': 'gather', 'from': [output + '_queries_hashed'], 'axis': query_time_axis,
    'position': output + '_sorted_queries_orig_indices'}  # [B,sorted-query-time,n,r] :: d_h
  d[output + '_sorted_keys_hashed'] = {
    'class': 'gather', 'from': [output + '_keys_hashed'], 'axis': key_time_axis,
    'position': output + '_sorted_keys_orig_indices'}  # [B,sorted-key-time,n,r] :: d_h
  d[output + '_sorted_queries_hashed_neg_mask'] = {
    'class': 'gather', 'from': [output + '_queries_hashed_neg_mask'], 'axis': query_time_axis,
    'position': output + '_sorted_queries_orig_indices'}  # [B,sorted-query-time,n,r] :: d_h
  d[output + '_sorted_keys_hashed_neg_mask'] = {
    'class': 'gather', 'from': [output + '_keys_hashed_neg_mask'], 'axis': key_time_axis,
    'position': output + '_sorted_keys_orig_indices'}  # [B,sorted-key-time,n,r] :: d_h
  chunk_query_sequence('queries_hashed', pad_value=hash_mask_value)  # [B,n,r,query-chunk,query-window] :: d_h
  chunk_key_sequence('keys_hashed', pad_value=hash_mask_value)  # [B,n,r,key-chunk,key-window] :: d_h
  stack_chunked_key_sequence('keys_hashed')  # [B,n,r,query-chunk,stacked-key-window] :: d_h
  chunk_query_sequence('queries_hashed_neg_mask', pad_value=-hash_mask_value)  # [B,n,r,query-chunk,query-window] :: d_h
  chunk_key_sequence('keys_hashed_neg_mask', pad_value=-hash_mask_value)  # [B,n,r,key-chunk,key-window] :: d_h

  # Sort the queries, keys, values by applying the permutation
  d[output + '_sorted_queries_unscaled'] = {
    'class': 'gather', 'from': [queries_input], 'axis': query_time_axis,
    'position': output + '_sorted_queries_orig_indices'}  # [B,sorted-query-time,n,r,F|d_k]
  d[output + '_sorted_queries'] = {
    'class': 'eval', 'eval': '%s * source(0)' % (key_dim ** -0.5),
    'from': [output + '_sorted_queries_unscaled']}  # [B,sorted-query-time,n,r,F|d_k]
  d[output + '_sorted_keys'] = {
    'class': 'gather', 'from': [keys_input], 'axis': key_time_axis,
    'position': output + '_sorted_keys_orig_indices'}  # [B,sorted-key-time,n,r,F|d_k]
  d[output + '_sorted_values'] = {
    'class': 'gather', 'from': [values_input], 'axis': key_time_axis,
    'position': output + '_sorted_keys_orig_indices'}  # [B,sorted-key-time,n,r,F|d_v]

  # Chunk the sorted queries and keys and values
  chunk_query_sequence('queries', pad_value=0.0, have_feature_dim=True)  # [B,n,r,query-chunk,query-window,F|d_k]
  chunk_key_sequence('keys', pad_value=0.0, have_feature_dim=True)  # [B,n,r,key-chunk,key-window,F|d_k]
  chunk_key_sequence('values', pad_value=0.0, have_feature_dim=True)  # [B,n,r,key-chunk,key-window,F|d_v]

  # Compute invalid positions
  d[output + '_sorted_chunked_valid_query_position'] = {
    'class': 'compare',
    'from': [output + '_sorted_chunked_queries_hashed'], 'value': hash_mask_value,
    'kind': 'not_equal'}  # [B,n,r,query-chunk,query-window]
  d[output + '_sorted_chunked_valid_key_position'] = {
    'class': 'compare',
    'from': [output + '_sorted_chunked_stacked_keys_hashed'], 'value': hash_mask_value,
    'kind': 'not_equal'}  # [B,n,r,query-chunk,stacked-key-window]

  # Compute chunk alignment from query chunks to a fixed-sized set of key chunks
  if chunk_alignment == 'identity':
    d[output + '_query_chunk_alignment_center'] = {
      'class': 'range_in_axis', 'axis': 'stag:query-chunk', 'keepdims': False,
      'from': [output + '_sorted_chunked_queries']}  # [query-chunk] :: key-chunk
  elif chunk_alignment == 'search_bounds_centered':
    assert key_chunks_before == key_chunks_after
    d[output + '_sorted_chunked_queries_hashed_min'] = {
      'class': 'reduce', 'mode': 'min', 'axis': 'stag:query-window',
      'from': [output + '_sorted_chunked_queries_hashed']}  # [B,n,r,query-chunk] :: d_h
    d[output + '_sorted_chunked_queries_hashed_max'] = {
      'class': 'reduce', 'mode': 'max', 'axis': 'stag:query-window',
      'from': [output + '_sorted_chunked_queries_hashed_neg_mask']}  # [B,n,r,query-chunk] :: d_h
    d[output + '_sorted_chunked_keys_hashed_min'] = {
      'class': 'reduce', 'mode': 'min', 'axis': 'stag:key-window',
      'from': [output + '_sorted_chunked_keys_hashed']}  # [B,n,r,key-chunk] :: d_h
    d[output + '_sorted_chunked_keys_hashed_max'] = {
      'class': 'reduce', 'mode': 'max', 'axis': 'stag:key-window',
      'from': [output + '_sorted_chunked_keys_hashed_neg_mask']}  # [B,n,r,key-chunk] :: d_h
    d[output + '_query_chunk_alignment_lower_key_chunk'] = {
      'class': 'search_sorted', 'axis': 'stag:key-chunk', 'side': 'left',
      'sorted_sequence': output + '_sorted_chunked_keys_hashed_min',
      'values': output + '_sorted_chunked_queries_hashed_min'}  # [B,n,r,query-chunk] :: key-chunk
    # the upper_key_chunk will be one value to high. correct this later when computing the center.
    d[output + '_query_chunk_alignment_upper_key_chunk'] = {
      'class': 'search_sorted', 'axis': 'stag:key-chunk', 'side': 'right',
      'sorted_sequence': output + '_sorted_chunked_keys_hashed_max',
      'values': output + '_sorted_chunked_queries_hashed_max'}  # [B,n,r,query-chunk] :: key-chunk
    d[output + '_query_chunk_alignment_center'] = {
      'class': 'eval',
      'from': [output + '_query_chunk_alignment_lower_key_chunk', output + '_query_chunk_alignment_upper_key_chunk'],
      'eval': 'tf.cast(tf.round((source(0) + source(1) - 1) / 2), dtype="int32")',
      'out_type': {'dtype': 'int32'}}  # [B,n,r,query-chunk] :: key-chunk in float
  d[output + '_query_chunk_alignment_offset_unnamed'] = {
    'class': 'range', 'start': -key_chunks_before, 'delta': 1, 'limit': key_chunks_after + 1}  # [key-chunk-offset]
  d[output + '_query_chunk_alignment_offset'] = {
    'class': 'name_axis', 'from': [output + '_query_chunk_alignment_offset_unnamed'], 'axis': 'F',
    'description': 'key-chunk-offset'}  # [key-chunk-offset]
  d[output + '_query_chunk_alignment_unbounded'] = {
    'class': 'combine', 'from': [output + '_query_chunk_alignment_center', output + '_query_chunk_alignment_offset'],
    'kind': 'add'}  # [query-chunk,key-chunk-offset] :: key-chunk
  d[output + '_key_chunk_count_individual'] = {
    'class': 'length', 'from': [output + '_sorted_chunked_keys']}  # [B]
  d[output + '_key_chunk_count'] = {
    'class': 'reduce', 'mode': 'max', 'from': [output + '_key_chunk_count_individual'], 'axis': 'B'}  # []
  d[output + '_query_chunk_alignment'] = {
    'class': 'eval', 'from': [output + '_query_chunk_alignment_unbounded', output + '_key_chunk_count'],
    'eval': 'tf.math.floormod(source(0), source(1))'}  # [query-chunk,key-chunk-offset] :: key-chunk

  # Compute chunk alignment duplicate mask: mask[i] is true iff alignment[j] == alignment[i] for any j < i
  d[output + '_query_chunk_alignment_other'] = {
    'class': 'name_axis', 'from': [output + '_query_chunk_alignment'], 'axis': 'stag:key-chunk-offset',
    'description': 'other-key-chunk-offset'}  # [query-chunk,other-key-chunk-offset] :: key-chunk
  d[output + '_query_chunk_alignment_indices'] = {
    'class': 'range_in_axis', 'from': [output + '_query_chunk_alignment'], 'axis': 'stag:key-chunk-offset',
    'keepdims': False}  # [key-chunk-offset]
  d[output + '_query_chunk_alignment_other_indices'] = {
    'class': 'range_in_axis', 'from': [output + '_query_chunk_alignment_other'], 'axis': 'stag:other-key-chunk-offset',
    'keepdims': False}  # [other-key-chunk-offset]
  d[output + '_query_chunk_alignment_compare'] = {
    'class': 'compare', 'from': [output + '_query_chunk_alignment', output + '_query_chunk_alignment_other'],
    'kind': 'equal'}  # [query-chunk,key-chunk-offset,other-key-chunk-offset]
  d[output + '_query_chunk_alignment_left_only'] = {
    'class': 'compare',
    'from': [output + '_query_chunk_alignment_indices', output + '_query_chunk_alignment_other_indices'],
    'kind': 'greater'}  # [key-chunk-offset,other-key-chunk-offset]
  d[output + '_query_chunk_alignment_compare_left_only'] = {
    'class': 'combine',
    'from': [output + '_query_chunk_alignment_compare', output + '_query_chunk_alignment_left_only'],
    'kind': 'logical_and'}  # [query-chunk,key-chunk-offset,other-key-chunk-offset]
  d[output + '_query_chunk_alignment_duplicate_mask'] = {
    'class': 'reduce', 'mode': 'any', 'from': [output + '_query_chunk_alignment_compare_left_only'],
    'axis': 'stag:other-key-chunk-offset'}  # [query-chunk,key-chunk-offset]

  # Compute multiple hash round duplicate mask:
  # mask[round,query-chunk,query-chunk-offset,key-chunk-offset] is true iff
  # unsorted_hashes[other_round,original_query_pos(round, query_chunk, query_chunk_offset)]
  #  == unsorted_hashes[other_round, original_key_pos(round, query_chunk, key_chunk_offset)]
  # for any other_round < round
  if not allow_duplicate_attention:
    assert num_rounds == 1 or mask_different_hashes, 'cannot be implemented efficiently'
    d[output + '_other_round_queries_hashed'] = {
      'class': 'name_axis', 'from': [output + '_queries_hashed'], 'axis': 'stag:att-round',
      'description': 'other-att-round'}  # [B,query-time,n,other_round] :: d_h
    d[output + '_sorted_chunked_other_round_queries_hashed'] = {
      'class': 'gather', 'from': [output + '_other_round_queries_hashed'],
      'position': output + '_sorted_chunked_queries_orig_indices_clipped',
      'axis': query_time_axis}  # [B,n,r,query-chunk,query-window,other_round] :: d_h
    d[output + '_other_round_keys_hashed'] = {
      'class': 'name_axis', 'from': [output + '_keys_hashed'], 'axis': 'stag:att-round',
      'description': 'other-att-round'}  # [B,key-time,n,other_round] :: d_h
    d[output + '_sorted_chunked_stacked_other_round_keys_hashed'] = {
      'class': 'gather', 'from': [output + '_other_round_keys_hashed'],
      'position': output + '_sorted_chunked_stacked_keys_orig_indices_clipped',
      'axis': key_time_axis}  # [B,n,r,query-chunk,stacked-key-window,other_round] :: d_h
    d[output + '_sorted_chunked_other_round_compare'] = {
      'class': 'compare',
      'from': [output + '_sorted_chunked_other_round_queries_hashed', output + '_sorted_chunked_stacked_other_round_keys_hashed'],  # noqa
      'kind': 'equal'}  # [B,n,r,query-chunk,query-window,stacked-key-window,other-round] :: bool

    d[output + '_round_indices'] = {
      'class': 'range_in_axis', 'from': [output + '_queries_hashed'], 'axis': 'stag:att-round',
      'keepdims': False}  # [round]
    d[output + '_other_round_indices'] = {
      'class': 'range_in_axis', 'from': [output + '_other_round_queries_hashed'],
      'axis': 'stag:other-att-round', 'keepdims': False}  # [other-round]
    d[output + '_round_left_only'] = {
      'class': 'compare',
      'from': [output + '_round_indices', output + '_other_round_indices'],
      'kind': 'greater'}  # [round,other-round]
    d[output + '_sorted_chunked_other_round_compare_left_only'] = {
      'class': 'combine',
      'from': [
        output + '_sorted_chunked_other_round_compare', output + '_round_left_only',
        output + '_sorted_chunked_valid_query_position', output + '_sorted_chunked_valid_key_position'],
      'kind': 'logical_and'}  # [B,n,r,query-chunk,query-window,stacked-key-window,other-round] :: bool
    d[output + '_sorted_chunked_round_duplicate_mask'] = {
      'class': 'reduce', 'mode': 'any', 'from': [output + '_sorted_chunked_other_round_compare_left_only'],
      'axis': 'stag:other-att-round'}  # [B,n,r,query-chunk,query-window,stacked-key-window] :: bool

  # Collect stacked key and value chunks
  stack_chunked_key_sequence('keys')  # [B,n,r,query-chunk,stacked-key-window,F|d_k]
  stack_chunked_key_sequence('values')  # [B,n,r,query-chunk,stacked-key-window,F|d_v]

  # Compute chunked masking (True = mask away by setting to -inf) and small mask (True = set to small_mask_value)
  large_masking_layers_from = []  # with -inf
  small_masking_layers_from = []  # with -10*5 or so
  if past_only:
    d[output + '_sorted_chunked_mask_past_only'] = {
      'class': 'compare',
      'from': [output + '_sorted_chunked_queries_orig_indices', output + '_sorted_chunked_stacked_keys_orig_indices'],
      'kind': 'less'}  # [B,n,r,query-chunk,query-window,stacked-key-window]
    large_masking_layers_from.append(output + '_sorted_chunked_mask_past_only')
  if mask_different_hashes:
    # if mask_current is disabled, we want to fallback to the different hashes
    is_small = not mask_current
    d[output + '_sorted_chunked%s_mask_matching_hash%s' % (('_small', '_all') if is_small else ('', ''))] = {
      'class': 'compare',
      'from': [output + '_sorted_chunked_queries_hashed', output + '_sorted_chunked_stacked_keys_hashed'],
      'kind': 'not_equal'}  # [B,n,r,query-chunk,query-window,stacked-key-window]
    if is_small:
      # only set those positions to the small mask value that were NOT masked away with -inf before.
      d[output + '_sorted_chunked_small_mask_matching_hash'] = {
        'class': 'eval',
        'from': [output + '_sorted_chunked_small_mask_matching_hash_all', output + '_sorted_chunked_mask'],
        'eval': 'tf.logical_and(source(0), tf.logical_not(source(1)))'}  # [B,n,r,query-chunk,query-window,stacked-key-window]  # noqa
      small_masking_layers_from.append(output + '_sorted_chunked_small_mask_matching_hash')
    else:
      large_masking_layers_from.append(output + '_sorted_chunked_mask_matching_hash')
  d[output + '_sorted_chunked_mask_valid_key_position'] = {
    'class': 'eval', 'from': [output + '_sorted_chunked_valid_key_position'],
    'eval': 'tf.logical_not(source(0))'}  # [B,n,r,query-chunk,query-window]
  large_masking_layers_from.append(output + '_sorted_chunked_mask_valid_key_position')
  d[output + '_sorted_chunked_mask_key_chunk_duplicates_unnamed'] = {
    'class': 'repeat', 'from': [output + '_query_chunk_alignment_duplicate_mask'],
    'repetitions': key_chunk_size, 'axis': 'stag:key-chunk-offset'}  # [B,n,r,query-chunk,key-chunk-offset*key-window]
  d[output + '_sorted_chunked_mask_key_chunk_duplicates'] = {
    'class': 'name_axis', 'from': [output + '_sorted_chunked_mask_key_chunk_duplicates_unnamed'],
    'axis': 'stag:repeated|stag:key-chunk-offset',
    'description': 'stacked-key-window'}  # [B,n,r,query-chunk,stacked-key-window]
  large_masking_layers_from.append(output + '_sorted_chunked_mask_key_chunk_duplicates')
  if not allow_duplicate_attention:
    large_masking_layers_from.append(output + '_sorted_chunked_round_duplicate_mask')
  if mask_current:
    d[output + '_sorted_chunked_small_mask_current'] = {
      'class': 'compare',
      'from': [output + '_sorted_chunked_queries_orig_indices', output + '_sorted_chunked_stacked_keys_orig_indices'],
      'kind': 'equal'}  # [B,n,r,query-chunk,query-window,stacked-key-window]
    small_masking_layers_from.append(output + '_sorted_chunked_small_mask_current')
  if len(large_masking_layers_from) > 1:
    d[output + '_sorted_chunked_mask'] = {
      'class': 'combine', 'from': large_masking_layers_from,
      'kind': 'logical_or'}  # [B,n,r,query-chunk,query-window,stacked-key-window]
  else:
    d[output + '_sorted_chunked_mask'] = {
      'class': 'copy', 'from': large_masking_layers_from}  # [B,n,r,query-chunk,query-window,stacked-key-window]
  if len(small_masking_layers_from) > 1:
    d[output + '_sorted_chunked_small_mask'] = {
      'class': 'combine', 'from': small_masking_layers_from,
      'kind': 'logical_or'}  # [B,n,r,query-chunk,query-window,stacked-key-window]
  elif len(small_masking_layers_from) == 1:
    d[output + '_sorted_chunked_small_mask'] = {
      'class': 'copy', 'from': small_masking_layers_from}  # [B,n,r,query-chunk,query-window,stacked-key-window]
  else:
    d[output + '_sorted_chunked_small_mask'] = {
      'class': 'constant', 'value': False, 'dtype': 'bool'}  # []
  # We never want the attention weights to be NaN for any query (even for unmasked queries),
  # and thus need to have at least one masked key for every query.
  # Otherwise, the gradients will be NaN.
  # We ensure this by masking all energies with a small (finite) number.
  d[output + '_sorted_chunked_final_mask'] = {
    'class': 'reduce', 'from': [output + '_sorted_chunked_mask'],
    'mode': 'all', 'axis': 'stag:stacked-key-window'}  # [B,n,r,query-chunk,query-window]

  # Compute chunked energy by comparing chunked queries and keys for each query chunk
  d[output + '_sorted_chunked_energy_unmasked1'] = {
    'class': 'dot', 'red1': 'static:-1', 'red2': 'static:-1',
    'var1': 'stag:query-window', 'var2': 'stag:stacked-key-window',
    'from': [output + '_sorted_chunked_queries', output + '_sorted_chunked_stacked_keys'],
    'debug': True}  # [B,n,r,query-chunk,query-window,stacked-key-window]
  d[output + '_sorted_chunked_energy_unmasked2'] = {
    'class': 'switch', 'condition': output + '_sorted_chunked_mask',
    'true_from': float('-inf'),
    'false_from': output + '_sorted_chunked_energy_unmasked1'}  # [B,n,r,query-chunk,query-window,stacked-key-window]
  d[output + '_sorted_chunked_energy_unmasked3'] = {
    'class': 'switch', 'condition': output + '_sorted_chunked_small_mask',
    'true_from': small_mask_value,
    'false_from': output + '_sorted_chunked_energy_unmasked2'}  # [B,n,r,query-chunk,query-window,stacked-key-window]
  # in multi-round att: ignore rounds that cannot attend anywhere. (for single round the mask value does not matter.)
  # however, prioritize e.g. mask_current => use smaller value here
  d[output + '_sorted_chunked_energy_feature'] = {
    'class': 'switch', 'condition': output + '_sorted_chunked_final_mask',
    'true_from': even_smaller_mask_value,
    'false_from': output + '_sorted_chunked_energy_unmasked3'}  # [B,n,r,query-chunk,query-window,stacked-key-window]
  d[output + '_sorted_chunked_energy'] = {
    'class': 'reinterpret_data', 'from': [output + '_sorted_chunked_energy_feature'],
    'set_axes': {'F': None}}  # [B,n,r,query-chunk,query-window,stacked-key-window]

  # Compute attention output of each round
  d[output + '_sorted_chunked_energy_logsumexp'] = {
    'class': 'reduce', 'mode': 'logsumexp', 'axis': 'stag:stacked-key-window',
    'from': [output + '_sorted_chunked_energy']}  # [B,n,r,query-chunk,query-window]
  d[output + '_sorted_chunked_weights'] = {
    'class': 'eval', 'from': [output + '_sorted_chunked_energy', output + '_sorted_chunked_energy_logsumexp'],
    'eval': 'tf.exp(source(0) - source(1))'}  # [B,n,r,query-chunk,query-window,stacked-key-window]
  d[output + '_sorted_chunked_weights_drop'] = {
    'class': 'dropout', 'dropout_noise_shape': {'*': None}, 'from': [output + '_sorted_chunked_weights'],
    'dropout': dropout}  # [B,n,r,query-chunk,query-window,stacked-key-window]
  d[output + '_sorted_chunked_round_output'] = {
    'class': 'dot', 'red1': 'stag:stacked-key-window', 'red2': 'stag:stacked-key-window',
    'var1': 'stag:query-window', 'var2': 'static:-1', 'debug': True,
    'from': [
      output + '_sorted_chunked_weights_drop',
      output + '_sorted_chunked_stacked_values']}  # [B,n,r,query-chunk,query-window,F|d_v]

  # Undo chunking and undo sorting
  d[output + '_sorted_round_output'] = {
    'class': 'merge_dims', 'axes': ['stag:query-chunk', 'stag:query-window'], 'keep_order': True,
    'from': [output + '_sorted_chunked_round_output']}  # [B,n,r,sorted-query-time=query-chunk*query-window,F|d_v]
  d[output + '_round_output'] = {
    'class': 'gather', 'from': [output + '_sorted_round_output'], 'axis': 'T',
    'position': output + '_queries_sort_indices'}  # [B,n,r,query-time,F|d_v]
  d[output + '_sorted_energy_logsumexp'] = {
    'class': 'merge_dims', 'axes': ['stag:query-chunk', 'stag:query-window'], 'keep_order': True,
    'from': [output + '_sorted_chunked_energy_logsumexp']}  # [ B,n,r,sorted-query-time=query-chunk*query-window]
  d[output + '_energy_logsumexp'] = {
    'class': 'gather', 'from': [output + '_sorted_energy_logsumexp'], 'axis': 'T',
    'position': output + '_queries_sort_indices'}  # [B,n,r,query-time]

  # Combine the context vectors of the different rounds
  d[output + '_round_output_weights'] = {
    'class': 'softmax_over_spatial', 'axis': 'stag:att-rounds', 'use_time_mask': False, 'energy_factor': 1.0,
    'from': [output + '_energy_logsumexp']}  # [B,n,r,query-time]
  d[output + '_round_output_weighted'] = {
    'class': 'combine', 'kind': 'mul',
    'from': [output + '_round_output_weights', output + '_round_output']}  # [B,n,query-time,F|d_v]
  d[output + '_output'] = {
    'class': 'reduce', 'axis': 'stag:att-rounds', 'mode': 'sum',
    'from': [output + '_round_output_weighted']}  # [B,n,query-time,F|d_v]
  d[output + '_output_unnamed'] = {
    'class': 'name_axis', 'axis': 'stag:att-heads', 'description': None,
    'from': [output + '_output']}  # [B,query-time,F|n*d_v]
  d[output + '_att_all'] = {
    'class': 'merge_dims', 'axes': 'static',
    'from': [output + '_output_unnamed']}  # [B,query-time,F|n*d_v]

  if debug_print:
    for name in [output + n for n in [
      '_keys_hashed', '_queries_hashed',
      '_sorted_queries_orig_indices', '_sorted_keys_orig_indices', '_queries_sort_indices',
      '_sorted_queries', '_sorted_keys', '_sorted_values',
      '_sorted_chunked_queries', '_sorted_chunked_keys',
      '_sorted_chunked_stacked_keys', '_sorted_chunked_stacked_values',
      '_sorted_queries_hashed', '_sorted_keys_hashed', '_sorted_chunked_keys_hashed',
      '_query_chunk_alignment',
      '_sorted_chunked_queries_orig_indices', '_sorted_chunked_stacked_keys_orig_indices',
      '_sorted_chunked_stacked_keys_hashed',
      '_query_chunk_alignment_duplicate_mask',
      '_sorted_chunked_mask', '_sorted_chunked_small_mask',
      '_sorted_chunked_energy_unmasked1', '_sorted_chunked_energy_unmasked2',
      '_sorted_chunked_energy',
      '_sorted_chunked_weights',
      '_sorted_chunked_round_output',
      '_att_all']] + large_masking_layers_from + small_masking_layers_from + [keys_input, queries_input]:
      if name.startswith('base:'):
        d['print_' + name[len('base:'):]] = {'class': 'print', 'from': [name], 'is_output_layer': True}
      else:
        assert name in d and name + '_orig' not in d
        d[name + '_orig'] = d[name]
        d[name] = {'class': 'print', 'from': [name + '_orig']}


def add_lsh_self_attention_layer(
  d, input, output, inside_rec_layer=True, past_only=None, time_axis=None, *,
  num_heads=8, num_rounds=1, key_dim=64, value_dim=64, dropout=0.0, num_hashes, chunk_size, chunks_before=None,
  chunks_after=None, ff_init="variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0,
  mask_current=True, small_mask_value=float(-10**5),
  share_key_query=True, normalize_keys=None,
  mask_different_hashes=True, allow_duplicate_attention=False,
  chunk_alignment, shuffle_kv=False, debug_print=False):
  """
  Essentially this does (but for LSH attention)
    d[output + '_att'] = {"class": "self_attention", "num_heads": num_heads,
      "total_key_dim": num_heads * key_dim,
      "n_out": num_heads * value_dim, "from": [input],
      "attention_left_only": left_only,
      "attention_dropout": dropout, "forward_weights_init": self.ff_init}
  But using multiple layers.

  :param dict[str,dict] d: the network dict to write into
  :param str input: input layer, of shape [B,query_axis?,F]
  :param str output: prefix of all layers generated. Output is written into output + '_att' layer.
    Will use the name output + '_...' for all internal layers here.
  :param bool inside_rec_layer: whether this is used inside a RecLayer, meaning that the time axis may or may not always
    exist.
  :param bool|None past_only: if set, will mask attention s.t. it cannot attend to the future.
    Must be set if used inside a RecLayer.
  :param str|None time_axis: name of the time axis
  :param int num_heads: number of attention heads
  :param int num_rounds: number of hashing rounds.
    Similar to attention heads but attend to same query/key/value sequence but with different hash matrix.
  :param int key_dim: feature dimension of keys and queries
  :param int value_dim: feature dimension of values
  :param int dropout: apply dropout to the attention weights
  :param int num_hashes: number of different attention hashes, must be an even number
  :param int chunk_size: window size within a single chunk
  :param int|None chunks_before: number of chunks we look into the past
  :param int|None chunks_after: number of chunks we look into the future
  :param str ff_init: initializer for the weight matrices, including the hash generator matrices
  :param bool mask_current: whether a query may attend to the key corresponding to the same position
  :param float|None mask_current_value: if mask_current, the attention energy if query=key is set to this.
    All other masked values are set to -inf, thus if mask_current_value is something low but higher than -inf, will
    attend to key=query exactly iff it is the only possible key to attend to
  :param bool small_mask_value: whether a query may only attend to keys with the same hash
  :param bool share_key_query: whether to set the key sequence equal to the query sequence
  :param bool normalize_keys: whether to normalize the key sequence in euclidean norm
  :param bool allow_duplicate_attention: whether to mask attention s.t. it only attends to each key once.
    Attending to a key twice can e.g. happen for multi-round attention,
    or if the (effective) chunk size is larger than the sequence length.
  :param str chunk_alignment:
  :param bool shuffle_kv: whether to shuffle the keys and values before sorting them by their hashes.
  :param bool debug_print: will print layers contents for debugging
  """
  if past_only is None:
    past_only = inside_rec_layer
  if time_axis is None:
    time_axis = 'stag:extern_data:classes' if inside_rec_layer else 'stag:extern_data:data'
  assert time_axis.startswith('stag:')
  assert not inside_rec_layer or past_only
  if normalize_keys is None:
    normalize_keys = share_key_query

  # Assume input [B,T|classes?,F|d_model]
  if share_key_query:
    d[output + '_qv0'] = {
      'class': 'linear', 'activation': None, 'with_bias': False, 'from': [input],
      'n_out': num_heads * (key_dim + value_dim), 'forward_weights_init': ff_init}  # [B,T|classes?,F|n*(d_k+d_v)]
    d[output + '_qv_unnamed'] = {
      'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, key_dim + value_dim),
      'from': [output + '_qv0']}  # [B,T|classes?,n,F|d_k+d_v]
    d[output + '_qv'] = {
      'class': 'name_axis', 'axis': 'static:-2', 'description': 'att-heads',
      'from': [output + '_qv_unnamed']}  # [B,T|classes?,n,F|d_k+d_v]
    d[output + '_qv_split'] = {
      'class': 'split', 'axis': 'F', 'size_splits': (key_dim, value_dim),
      'from': [output + '_qv']}
    d[output + '_query'] = {
      'class': 'copy', 'from': [output + '_qv_split/0']}  # [B,T|classes?,n,F|d_k]
    if normalize_keys:
      d[output + '_key'] = {
        'class': 'eval', 'eval': normalize_eval, 'from': [output + '_query']}  # [B,T|classes?,n,F|d_k]
    else:
      d[output + '_key'] = {
        'class': 'copy', 'from': [output + '_query']}  # [B,T|classes?,n,F|d_k]
    d[output + '_value'] = {
      'class': 'copy', 'from': [output + '_qv_split/1']}  # [B,T|classes?,n,F|d_v]
  else:  # not share_key_query
    d[output + '_qkv0'] = {
      'class': 'linear', 'activation': None, 'with_bias': False, 'from': [input],
      'n_out': num_heads * (2 * key_dim + value_dim), 'forward_weights_init': ff_init}  # [B,T?,F|n*(2d_k+d_v)]
    d[output + '_qkv_unnamed'] = {
      'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, 2 * key_dim + value_dim),
      'from': [output + '_qkv0']}  # [B,T?,n,F|2d_k+d_v]
    d[output + '_qkv'] = {
      'class': 'name_axis', 'axis': 'static:-2', 'description': 'att-heads',
      'from': [output + '_qkv_unnamed']}  # [B,T|classes?,n,F|2d_k+d_v]
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

  if inside_rec_layer:
    queries_input, keys_input, values_input = output + '_query_accum', output + '_key_accum', output + '_value_accum'
    for qkv in ('query', 'key', 'value'):
      d[output + '_%s_accum' % qkv] = {'class': 'cum_concat', 'from': [output + '_%s' % qkv], 'axis': time_axis}
    time_axis_ = 'stag:rec-history'
  else:
    queries_input, keys_input, values_input = output + '_query', output + '_key', output + '_value'
    time_axis_ = time_axis

  # this always computes attention for an entire sequence
  generic_add_lsh_attention_layer(
    d, queries_input=queries_input, keys_input=keys_input, values_input=values_input,
    output=output, query_time_axis=time_axis_, key_time_axis=time_axis_,
    num_heads=num_heads, num_rounds=num_rounds, key_dim=key_dim, value_dim=value_dim, dropout=dropout,
    num_hashes=num_hashes, query_chunk_size=chunk_size, key_chunk_size=chunk_size,
    key_chunks_before=chunks_before, key_chunks_after=chunks_after, hash_init=ff_init,
    small_mask_value=small_mask_value, past_only=past_only, mask_current=mask_current,
    mask_different_hashes=mask_different_hashes, allow_duplicate_attention=allow_duplicate_attention,
    chunk_alignment=chunk_alignment, shuffle_kv=shuffle_kv, debug_print=debug_print)

  if inside_rec_layer:
    d[output + '_att'] = {'class': 'gather', 'from': [output + '_att_all'], 'position': ':i', 'axis': time_axis_}
  else:
    d[output + '_att'] = {'class': 'copy', 'from': [output + '_att_all']}


def add_lsh_cross_attention_layer(
  d, db, input, keys_input, output, query_time_axis=None, key_time_axis=None, *,
  num_heads=8, num_rounds=1, key_dim=64, value_dim=64, dropout=0.0, num_hashes, key_chunk_size, query_chunk_size,
  key_chunks_before=None, key_chunks_after=None,
  ff_init="variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0,
  hash_init="variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0,
  small_mask_value=float(-10**5), mask_different_hashes=True, allow_duplicate_attention=False,
  chunk_alignment, shuffle_kv=False, query_hash_dropin=0.0, key_hash_dropin=0.0, debug_print=False):
  query_time_axis, key_time_axis = _query_key_time_default(query_time_axis, key_time_axis)
  assert keys_input.startswith('base:')
  keys_input = keys_input[len('base:'):]

  # Create query, key and value
  _make_cross_attention_qkv(
    d=d, db=db, input=input, keys_input=keys_input, output=output, num_heads=num_heads, key_dim=key_dim,
    value_dim=value_dim, ff_init=ff_init)

  # Accumulate the queries
  queries_input = output + '_query_accum'
  d[output + '_query_accum'] = {'class': 'cum_concat', 'from': [output + '_query'], 'axis': query_time_axis}

  # Apply lsh hashing for all queries
  generic_add_lsh_attention_layer(
    d=d, queries_input=queries_input, keys_input='base:' + output + '_key',
    values_input='base:' + output + '_value', output=output, query_time_axis='stag:rec-history',
    key_time_axis=key_time_axis, num_heads=num_heads, num_rounds=num_rounds, key_dim=key_dim, value_dim=value_dim,
    dropout=dropout, num_hashes=num_hashes, query_chunk_size=query_chunk_size, key_chunk_size=key_chunk_size,
    key_chunks_before=key_chunks_before, key_chunks_after=key_chunks_after, hash_init=hash_init,
    small_mask_value=small_mask_value, mask_different_hashes=mask_different_hashes,
    allow_duplicate_attention=allow_duplicate_attention, chunk_alignment=chunk_alignment, shuffle_kv=shuffle_kv,
    query_hash_dropin=query_hash_dropin, key_hash_dropin=key_hash_dropin,
    debug_print=debug_print)

  # Select the context vector for the query we actually want
  d[output + '_att'] = {'class': 'gather', 'from': [output + '_att_all'], 'position': ':i', 'axis': 'stag:rec-history'}


def add_full_lsh_cross_attention_layer(
  d, db, input, keys_input, output, query_time_axis=None, key_time_axis=None,
  num_heads=8, key_dim=64, value_dim=64, dropout=0.0,
  ff_init = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0,
  num_hashes=14, num_rounds=1, mask_current_value=float(-10**5), mask_different_hashes=True, debug_print=False):
  """
  Add a cross-attention layer with masking as in the LSH case.
  This way, you can e.g. train a system using LSH attention, but then do search using this.

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
  :param int num_hashes:
  :param int num_rounds:
  :param float mask_current_value:
  :param bool mask_different_hashes:
  :param bool debug_print:
  """
  query_time_axis, key_time_axis = _query_key_time_default(query_time_axis, key_time_axis)

  # build standard self-attention
  assert keys_input.startswith('base:')
  add_vanilla_cross_attention_layer(
    d=d, db=db, input=input, keys_input=keys_input, output=output, query_time_axis=query_time_axis,
    key_time_axis=key_time_axis, num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, dropout=dropout,
    ff_init=ff_init)  # [B,n,r,d_k,F|d_h]

  # hash keys and queries
  assert mask_different_hashes, 'can just call add_vanilla_cross_attention_layer(..) instead'
  make_lsh_hash_gen(
    db, output + '_hash_gen', key_dim=key_dim, num_hashes=num_hashes, num_heads=num_heads, num_rounds=num_rounds,
    hash_init=ff_init)  # [B,n,r,d_k,F|d_h]
  apply_lsh_hash_gen(
    d, input=output + '_query', hash_gen_input='base:' + output + '_hash_gen', output=output + '_query_hash',
    time_axis=query_time_axis, num_hashes=num_hashes, hash_mask_value=None)  # [B,n,r,query-T?] :: d_h
  apply_lsh_hash_gen(
    db, input=output + '_key', hash_gen_input=output + '_hash_gen', output=output + '_key_hash',
    time_axis=key_time_axis, num_hashes=num_hashes, hash_mask_value=None)  # [B,n,r,key-T] :: d_h
  assert num_rounds == 1, 'not implemented yet otherwise'

  # build and apply additional energy mask
  d[output + '_energy_mask_rounds'] = {
    'class': 'compare', 'from': [output + '_query_hash', 'base:' + output + '_key_hash'],
    'kind': 'equal'}  # [B,n,r,T-query?,T-key]
  d[output + '_energy_mask'] = {
    'class': 'squeeze', 'axis': 'stag:att-round', 'from': [output + '_energy_mask_rounds']}  # [B,n,T-query?,T-key]
  assert (output + '_energy') in d
  d[output + '_energy_unmasked'] = d[output + '_energy']  # [B,n,query-T?,key-T]
  d[output + '_energy'] = {
    'class': 'switch', 'condition': output + '_energy_mask',
    'true_from': output + '_energy_unmasked', 'false_from': mask_current_value}  # [B,n,query-T?,key-T]

  if debug_print:
    for name in [output + n for n in [
      '_query', '_query_hash', '_energy_mask', '_energy_unmasked', '_energy', '_weights']] + [
      'base:' + output + '_key', 'base:' + output + '_key_hash']:
      if name.startswith('base:'):
        name = name[len('base:'):]
        assert name in db and name + '_orig' not in db
        db[name + '_orig'] = db[name]
        db[name] = {'class': 'print', 'from': [name + '_orig']}
      else:
        assert name in d and name + '_orig' not in d
        d[name + '_orig'] = d[name]
        d[name] = {'class': 'print', 'from': [name + '_orig']}

