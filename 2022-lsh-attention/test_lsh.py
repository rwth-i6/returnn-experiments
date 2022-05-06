import sys

from typing import Any

import numpy as np
from numpy.testing import assert_almost_equal

import returnn
import os

import unittest
from pprint import pprint
from nose.tools import assert_equal
from returnn.util import better_exchook
from returnn.config import Config
from returnn.tf.network import *
from returnn.tf.layers.basic import *

# add the returnn/tests folder to the syspath
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(returnn.__file__)), "tests"))
from tests.test_TFNetworkLayer import make_scope, make_feed_dict
from tests.test_TFNetworkRecLayer import check_reclayer_optimize_out

from vanilla_attention import *
from lsh_attention import *


def _test_lsh_attention_optimize_out(chunk_size, chunks_before, chunks_after, num_hashes=26, chunk_align='identity'):
  num_heads, key_dim, value_dim = 2, 3, 3
  # check_reclayer_optimize_out uses n_time = 7.
  network = {}
  add_lsh_self_attention_layer(
    network, 'data:source', 'att', chunks_before=chunks_before, chunks_after=chunks_after, chunk_size=chunk_size,
    num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, num_hashes=num_hashes, inside_rec_layer=True,
    past_only=True, allow_duplicate_attention=False, time_axis='stag:extern_data:data', chunk_alignment=chunk_align)

  check_reclayer_optimize_out(
    {'class': 'copy', 'from': 'att_att', 'n_out': value_dim * num_heads},
    other_subnet_layers=network, rtol=3*1e-3)


def test_lsh_attention_optimize_out():
  _test_lsh_attention_optimize_out(chunk_size=1, chunks_before=0, chunks_after=0)
  _test_lsh_attention_optimize_out(chunk_size=7, chunks_before=0, chunks_after=0)
  _test_lsh_attention_optimize_out(chunk_size=7, chunks_before=0, chunks_after=0, num_hashes=2)
  _test_lsh_attention_optimize_out(chunk_size=4, chunks_before=1, chunks_after=0)
  _test_lsh_attention_optimize_out(chunk_size=4, chunks_before=1, chunks_after=0, num_hashes=2)
  _test_lsh_attention_optimize_out(chunk_size=4, chunks_before=1, chunks_after=1, chunk_align='search_bounds_centered')


def _test_lsh_self_attention_no_mask_different_hashes(
    n_time, past_only, mask_current, chunk_size, chunks_before, chunks_after, duplicates, n_batch=3,
    num_heads=2, key_dim=3, value_dim=3, num_hashes=26, chunk_align='identity'):
  print(
    'Testing n_time =', n_time, 'past_only =', past_only, 'mask_current =', mask_current, 'chunk_size =', chunk_size,
    'chunks_before =', chunks_before, 'chunks_after =', chunks_after, 'allow_duplicate_attention =', duplicates)
  with make_scope() as session:
    total_chunks = chunks_before + 1 + chunks_after
    assert n_time <= chunk_size * (chunks_before + 1 + chunks_after), (
      'if chunk size is too small, vanilla attention != lsh attention')
    assert not duplicates or n_time >= chunk_size * (total_chunks - 1), (
      'if chunk size is too big, we might attend multiple times')
    net_dict = {
      "lsh_out": {"class": "copy", "from": "lsh_att", "is_output_layer": True},  # [B,T1,F]
      "vanilla_out": {"class": "copy", "from": "vanilla_att", "is_output_layer": True},  # [B,T1,F]

      "output": {"class": "copy", "from": ["lsh_att"]}}  # [B,T,F]
    add_lsh_self_attention_layer(
      net_dict, 'data', 'lsh', inside_rec_layer=False, past_only=past_only,
      num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, num_hashes=num_hashes,
      chunk_size=chunk_size, chunks_before=chunks_before, chunks_after=chunks_after,
      mask_current=mask_current, mask_different_hashes=False, allow_duplicate_attention=duplicates,
      chunk_alignment=chunk_align, debug_print=False)
    add_vanilla_self_attention_layer(
      net_dict, 'data', 'vanilla', inside_rec_layer=False, past_only=past_only,
      num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, share_key_query=True,
      mask_current=mask_current
    )
    net_dict["vanilla_qv0"]["reuse_params"] = "lsh_qv0"

    config = Config({"debug_print_layer_output_template": True, "debug_add_check_numerics_ops": True})
    config.update(dict(num_inputs=num_heads*key_dim, num_outputs=num_heads*value_dim))
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)

    assert_equal(network.get_layer("vanilla_out").output.shape, (None, num_heads * value_dim))
    assert_equal(network.get_layer("lsh_out").output.shape, (None, num_heads * value_dim))
    feed_dict = make_feed_dict(network.extern_data.data.values(), same_time=True, n_batch=n_batch, n_time=n_time)
    session.run(tf_compat.v1.global_variables_initializer())

    input_out = network.get_layer("data").output
    vanilla_out = network.get_layer("vanilla_out").output
    lsh_out = network.get_layer("lsh_out").output
    vanilla, lsh, sizes, vanilla_sizes, lsh_sizes = session.run(
      [vanilla_out.placeholder, lsh_out.placeholder,
        input_out.size_placeholder[input_out.time_dim_axis_excluding_batch],
        vanilla_out.size_placeholder[vanilla_out.time_dim_axis_excluding_batch],
        lsh_out.size_placeholder[lsh_out.time_dim_axis_excluding_batch]],
      feed_dict=feed_dict)
    numpy.testing.assert_equal(vanilla_sizes, sizes)
    numpy.testing.assert_equal(lsh_sizes, sizes)
    # take into account different seq lengths
    assert vanilla_out.batch_dim_axis == lsh_out.batch_dim_axis == 0
    assert vanilla_out.time_dim_axis == lsh_out.time_dim_axis == 1
    mask = (numpy.arange(numpy.shape(vanilla)[1]).reshape([1,-1,1]) < sizes.reshape([-1,1,1]))
    vanilla = vanilla * mask
    lsh = lsh * mask
    print('seq lengths:', sizes)
    print('vanilla out:  - ', vanilla_out)
    pprint(vanilla)
    print('lsh out:  -', lsh_out)
    pprint(lsh)
    numpy.testing.assert_almost_equal(vanilla, lsh, decimal=5)
    print('They are equal!')


def test_lsh_self_attention_no_mask_different_hashes():
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=False, mask_current=False, chunk_size=5, chunks_before=1, chunks_after=1, duplicates=True)
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=False, mask_current=True, chunk_size=5, chunks_before=1, chunks_after=1, duplicates=True)
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=True, mask_current=False, chunk_size=7, chunks_before=1, chunks_after=0, duplicates=True)
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=True, mask_current=True, chunk_size=7, chunks_before=1, chunks_after=0, duplicates=True)
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=False, mask_current=False, chunk_size=5, chunks_before=1, chunks_after=1, duplicates=True,
    chunk_align='search_bounds_centered')
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=False, mask_current=True, chunk_size=5, chunks_before=1, chunks_after=1, duplicates=True,
    chunk_align='search_bounds_centered')


def test_lsh_self_attention_no_mask_different_hashes_no_duplicates():
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=6, past_only=False, mask_current=False, chunk_size=4, chunks_before=1, chunks_after=0, duplicates=False,
    num_heads=1)
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=2, past_only=False, mask_current=False, chunk_size=1, chunks_before=2, chunks_after=0, duplicates=False,
    n_batch=1, num_heads=1)
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=2, past_only=False, mask_current=False, chunk_size=1, chunks_before=2, chunks_after=0, duplicates=False)
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=False, mask_current=False, chunk_size=5, chunks_before=2, chunks_after=0, duplicates=False)
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=False, mask_current=False, chunk_size=5, chunks_before=3, chunks_after=0, duplicates=False)
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=True, mask_current=False, chunk_size=5, chunks_before=3, chunks_after=0, duplicates=False)

  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=False, mask_current=False, chunk_size=5, chunks_before=1, chunks_after=1, duplicates=False)
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=False, mask_current=True, chunk_size=5, chunks_before=1, chunks_after=1, duplicates=False)
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=True, mask_current=False, chunk_size=7, chunks_before=1, chunks_after=0, duplicates=False)
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=True, mask_current=True, chunk_size=7, chunks_before=1, chunks_after=0, duplicates=False)
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=False, mask_current=False, chunk_size=5, chunks_before=1, chunks_after=1, duplicates=False,
    chunk_align='search_bounds_centered')
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=False, mask_current=True, chunk_size=5, chunks_before=1, chunks_after=1, duplicates=False,
    chunk_align='search_bounds_centered')


def _test_lsh_self_attention_hashing(
    hash_sequence, chunk_size, chunks_before, chunks_after, past_only=False, chunk_align='identity', shuffle_kv=False):
  """
  :param np.ndarray hash_sequence: shape [batch, heads, rounds, time], dtype int32, with hash classes
  :return:
  """
  # For input position i, set key = value = i-th unit vector.
  # Distance between all query-key pairs is equal this way.
  # Set chunk size large enough s.t. only different hash classes will cause pruning.
  import numpy as np
  with make_scope() as session:
    print(
      '------ Testing with chunk_size =', chunk_size, 'chunks_before =', chunks_before, 'chunks_after =', chunks_after,
      'chunk_align =', chunk_align, 'shuffle_kv =', shuffle_kv)
    hash_sequence = np.asarray(hash_sequence, dtype='int32')
    if len(hash_sequence.shape) == 3:
      # hash_sequence is [batch, heads, time]
      hash_sequence = np.reshape(hash_sequence, hash_sequence.shape[:2] + (1,) + hash_sequence.shape[2:])
    assert len(hash_sequence.shape) == 4
    n_batch, num_heads, num_rounds, n_time = hash_sequence.shape
    num_hashes = 42
    key_dim, value_dim = n_time, n_time

    kqv_sequence = np.zeros((n_batch, n_time, num_heads, key_dim), dtype='float32')
    for query_t in range(n_time):
      kqv_sequence[:,query_t,:,query_t] = 1

    net_dict = {"output": {"class": "copy", "from": ["lsh_att"]}}
    add_lsh_self_attention_layer(
      net_dict, 'data', 'lsh', inside_rec_layer=False, past_only=past_only, num_heads=num_heads, num_rounds=num_rounds,
      key_dim=key_dim, value_dim=value_dim, num_hashes=num_hashes, chunk_size=chunk_size, chunks_before=chunks_before,
      chunks_after=chunks_after, share_key_query=True,
      mask_current=True, mask_different_hashes=True, allow_duplicate_attention=False, chunk_alignment=chunk_align,
      shuffle_kv=shuffle_kv, debug_print=False)
    # Now we override the keys/queries, lsh_value and lsh_kq_hash with our own inputs

    def get_kqv_sequence(self, source):
      assert source(0, as_data=True).shape == (None, num_heads, key_dim)
      return tf.constant(kqv_sequence)

    def get_hash_sequence(self, source):
      assert source(0, as_data=True).shape == (num_heads, num_rounds, None)
      return tf.constant(hash_sequence)

    assert "lsh_query" in net_dict and "lsh_queries_hashed" in net_dict and "lsh_queries_hashed_neg_mask"
    net_dict["unittest_kq_ignored_orig"] = net_dict["lsh_query"]  # keep a copy of the original value for the data shape
    net_dict["lsh_query"] = {"class": "eval", "from": "unittest_kq_ignored_orig", "eval": get_kqv_sequence}
    # lsh_key is just copied from lsh_query because share_kq=True
    assert "lsh_queries_hashed" in net_dict and "lsh_queries_hashed_neg_mask" in net_dict
    net_dict["unittest_q_hash_ignored_orig"] = net_dict[
      "lsh_queries_hashed"]  # keep a copy of the original value for the data shape
    net_dict["lsh_queries_hashed"] = {"class": "eval", "from": "unittest_q_hash_ignored_orig", "eval": get_hash_sequence}
    net_dict["lsh_queries_hashed_neg_mask"] = {"class": "copy", "from": "lsh_queries_hashed"}

    assert "lsh_value" in net_dict
    net_dict["unittest_v_ignored_orig"] = net_dict["lsh_value"]
    net_dict["lsh_value"] = {"class": "eval", "from": "unittest_v_ignored_orig", "eval": get_kqv_sequence}

    assert "lsh_keys_hashed" in net_dict and "lsh_keys_hashed_neg_mask" in net_dict
    net_dict["lsh_keys_hashed"] = {"class": "copy", "from": "lsh_queries_hashed"}
    net_dict["lsh_keys_hashed_neg_mask"] = {"class": "copy", "from": "lsh_queries_hashed"}

    config = Config({"debug_print_layer_output_template": True, "debug_add_check_numerics_ops": True})
    config.update(dict(num_inputs=num_heads * key_dim, num_outputs=num_heads * value_dim))
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)

    assert_equal(network.get_layer("lsh_query").output.shape, (None, num_heads, key_dim))  # [B,T,H,F]
    assert_equal(network.get_layer("lsh_value").output.shape, (None, num_heads, value_dim))  # [B,T,H,F]
    assert_equal(network.get_layer("lsh_queries_hashed").output.shape, (num_heads, num_rounds, None))  # [B,H,R,T]
    assert_equal(network.get_layer("lsh_output").output.shape, (num_heads, None, value_dim))  # [B,H,T,F]
    session.run(tf_compat.v1.global_variables_initializer())
    feed_dict = {
      network.extern_data.data["data"].placeholder: np.zeros((n_batch, n_time, num_heads * key_dim)),
      network.extern_data.data["data"].size_placeholder[0]: [n_time] * n_batch}
    fetch_output = session.run(
      network.get_layer("lsh_output").output.placeholder, feed_dict=feed_dict)  # type: np.ndarray
    assert fetch_output.shape == (n_batch, num_heads, n_time, value_dim)

    with np.printoptions(precision=2, suppress=True):
      print('input hash vector [B,H,R,T]:')
      pprint(hash_sequence)
      print('output context vector [B,H,R,T,F] (with past_only=%s):' % past_only)
      pprint(fetch_output)
    for b in range(n_batch):
      for h in range(num_heads):
        for query_t in range(n_time):
          output_vector = fetch_output[b,h,query_t,:]
          query_hash = hash_sequence[b,h,:,query_t]  # [R]
          matching_keys = [
            key_t for key_t in range(n_time) if
            any(hash_sequence[b,h,r,key_t] == query_hash[r] for r in range(num_rounds))
            and key_t != query_t
            and (not past_only or key_t <= query_t)]
          if len(matching_keys) == 0:
            matching_keys = [query_t]
          should = np.zeros((value_dim,))
          for matching_key in matching_keys:
            should[matching_key] = 1 / len(matching_keys)
          np.testing.assert_almost_equal(output_vector, should, decimal=5)
    print('Matches!')


def _test_lsh_self_attention_hashing_all(hash_sequence, chunk_size, chunks_before, chunks_after):
  for past_only in [False, True]:
    for shuffle_kv in [False, True]:
      _test_lsh_self_attention_hashing(
        hash_sequence, chunk_size=chunk_size, chunks_before=chunks_before, chunks_after=chunks_after,
        past_only=past_only, chunk_align='identity', shuffle_kv=shuffle_kv)
      # with chunk_align='search_bounds_centered' these tests should always still work
      if chunks_before == chunks_after:
        _test_lsh_self_attention_hashing(
          hash_sequence, chunk_size=chunk_size, chunks_before=chunks_before, chunks_after=chunks_after,
          past_only=past_only, chunk_align='search_bounds_centered', shuffle_kv=shuffle_kv)


def test_lsh_self_attention_hashing():
  import numpy as np
  _test_lsh_self_attention_hashing_all([[[1,1,1,2,2,2,3,3,3]]], chunk_size=10, chunks_before=0, chunks_after=0)
  _test_lsh_self_attention_hashing_all([[[1,1,1,2,2,2,3,3,3]]], chunk_size=3, chunks_before=1, chunks_after=1)
  _test_lsh_self_attention_hashing_all([[[1,1,1,2,2,2,3,3,3]]], chunk_size=3, chunks_before=0, chunks_after=0)
  _test_lsh_self_attention_hashing_all([[[1,2,3,4,5,6,6,6,7,8,8,8,9]]], chunk_size=15, chunks_before=0, chunks_after=0)
  _test_lsh_self_attention_hashing_all(
    [[[1,1,1,2,2,2,3,3,3,4,4,4,4],[1,2,3,4,5,6,6,6,7,8,8,8,9]]], chunk_size=15, chunks_before=0, chunks_after=0)
  _test_lsh_self_attention_hashing_all(
    [[[1,1,1,2,2,2,3,3,3,4,4,4,4]],[[1,2,3,4,5,6,6,6,7,8,8,8,9]]], chunk_size=15, chunks_before=0, chunks_after=0)
  _test_lsh_self_attention_hashing_all([[[2,2,1,1,1]]], chunk_size=3, chunks_before=0, chunks_after=0)
  _test_lsh_self_attention_hashing_all([[[1,2,3,1,2,3]]], chunk_size=6, chunks_before=0, chunks_after=0)

  np.random.seed(0)
  _test_lsh_self_attention_hashing_all(
    np.random.randint(low=0, high=2, size=(1,1,1,5), dtype='int32'), chunk_size=3, chunks_before=1, chunks_after=0)
  np.random.seed(0)
  _test_lsh_self_attention_hashing_all(
    np.random.randint(low=0, high=30, size=(3,4,1,13), dtype='int32'), chunk_size=7, chunks_before=1, chunks_after=0)
  # technically, the chunk size is too small. but it is very unlikely that more than 3 keys have the same hash.
  np.random.seed(0)
  random_hashes = np.random.randint(low=0, high=26, size=(3,4,1,34), dtype='int32')
  _test_lsh_self_attention_hashing(random_hashes, chunk_size=3, chunks_before=1, chunks_after=1, past_only=False)
  _test_lsh_self_attention_hashing(random_hashes, chunk_size=3, chunks_before=1, chunks_after=0, past_only=True)
  _test_lsh_self_attention_hashing(
    random_hashes, chunk_size=3, chunks_before=1, chunks_after=1, past_only=False, chunk_align='search_bounds_centered')
  _test_lsh_self_attention_hashing(
    random_hashes, chunk_size=3, chunks_before=1, chunks_after=1, past_only=False, shuffle_kv=True)
  _test_lsh_self_attention_hashing(
    random_hashes, chunk_size=3, chunks_before=1, chunks_after=1, past_only=False, chunk_align='search_bounds_centered',
    shuffle_kv=True)


def test_lsh_self_attention_hashing_multi_round():
  # hash rounds do not help here, as the hash classes for all rounds are equal
  _test_lsh_self_attention_hashing_all(
    [[[[1,1,1,2,2,2,3,3,3], [2,2,2,3,3,3,4,4,4]]]], chunk_size=10, chunks_before=0, chunks_after=0)
  _test_lsh_self_attention_hashing_all(
    [[[[1,1,1,2,2,2,3,3,3], [4,4,4,2,2,2,3,3,3]]]], chunk_size=10, chunks_before=0, chunks_after=0)
  # hash rounds should now increase the effective window, but keys of different hash rounds are disjoint.
  _test_lsh_self_attention_hashing_all(
    [[[[1,1,1,2,2,2,3,3,3], [1,2,3,4,1,2,3,4,5]]]], chunk_size=10, chunks_before=0, chunks_after=0)
  # hash rounds increase the effective window, but also select keys twice some times.
  _test_lsh_self_attention_hashing_all(
    [[[[1,2,2,2], [5,5,5,6]]]], chunk_size=4, chunks_before=0, chunks_after=0)
  _test_lsh_self_attention_hashing_all(
    [[[[1,1,1,2,2,2,3,3,3], [1,1,2,2,3,3,4,4,5]]]], chunk_size=10, chunks_before=0, chunks_after=0)
  # now try a bigger test
  import numpy as np
  np.random.seed(0)
  _test_lsh_self_attention_hashing_all(
    np.random.randint(low=0, high=2, size=(1,1,3,5), dtype='int32'), chunk_size=3, chunks_before=1, chunks_after=0)
  np.random.seed(0)
  _test_lsh_self_attention_hashing_all(
    np.random.randint(low=0, high=30, size=(3,4,5,13), dtype='int32'), chunk_size=7, chunks_before=1, chunks_after=0)
  # # technically, the chunk size is too small. but it is very unlikely that more than 5 keys have the same hash.
  # # (even for eight hash rounds). Also see test_lsh_self_attention_hashing, where we use a little bit lower count.
  np.random.seed(0)
  random_hashes = np.random.randint(low=0, high=26, size=(3,4,8,34), dtype='int32')
  _test_lsh_self_attention_hashing(random_hashes, chunk_size=5, chunks_before=1, chunks_after=1, past_only=False)
  _test_lsh_self_attention_hashing(random_hashes, chunk_size=5, chunks_before=1, chunks_after=0, past_only=True)
  _test_lsh_self_attention_hashing(
    random_hashes, chunk_size=5, chunks_before=1, chunks_after=1, past_only=False, shuffle_kv=True)
  # search_bounds_centered does not work well:
  # e.g. chunk_size=5 does not pass, chunk_size=6 passes, chunk_size=8 does not, chunk_size=10 again passes...
  _test_lsh_self_attention_hashing(
    random_hashes, chunk_size=10, chunks_before=1, chunks_after=1, past_only=False,
    chunk_align='search_bounds_centered')
  _test_lsh_self_attention_hashing(
    random_hashes, chunk_size=3, chunks_before=2, chunks_after=2, past_only=False,
    chunk_align='search_bounds_centered')
  _test_lsh_self_attention_hashing(
    random_hashes, chunk_size=2, chunks_before=3, chunks_after=3, past_only=False,
    chunk_align='search_bounds_centered')
  _test_lsh_self_attention_hashing(
    random_hashes, chunk_size=1, chunks_before=3, chunks_after=3, past_only=False,
    chunk_align='search_bounds_centered')
  _test_lsh_self_attention_hashing(
    random_hashes, chunk_size=10, chunks_before=1, chunks_after=1, past_only=False,
    chunk_align='search_bounds_centered', shuffle_kv=True)
  _test_lsh_self_attention_hashing(
    random_hashes, chunk_size=2, chunks_before=3, chunks_after=3, past_only=False,
    chunk_align='search_bounds_centered', shuffle_kv=True)


def test_vanilla_self_attention_equal_to_SelfAttentionLayer():
  for past_only in [False, True]:
    with make_scope() as session:
      print('Testing past_only=%s' % past_only)
      n_time = 13
      num_heads, key_dim, value_dim = 2, 3, 3
      net_dict = {
        "single_layer_att": {
          "class": "self_attention", "from": "data", "num_heads": num_heads, "total_key_dim": num_heads * key_dim,
          "n_out": num_heads * value_dim, "attention_left_only": past_only, 'is_output_layer': True},  # [B,T,F]
        "multi_layer_att": None  # [B,T,F], added below.
      }
      add_vanilla_self_attention_layer(
        net_dict, 'data', 'multi_layer', inside_rec_layer=False, past_only=past_only,
        num_heads=num_heads, key_dim=key_dim, value_dim=value_dim)
      net_dict["multi_layer_att"]["is_output_layer"] = True
      def custom(reuse_layer, *args, **kwargs):
        return tf.identity(reuse_layer.params['QKV'])
      net_dict["multi_layer_qkv0"]["reuse_params"] = {
        "auto_create_missing": False, "map": {"W": {"reuse_layer": "single_layer_att", "custom": custom}}}

      config = Config({"debug_print_layer_output_template": True, "debug_add_check_numerics_ops": True})
      config.update(dict(num_inputs=num_heads*key_dim, num_outputs=num_heads*value_dim))
      network = TFNetwork(config=config, train_flag=True)
      network.construct_from_dict(net_dict)

      single_out = network.get_layer("single_layer_att").output
      multi_out = network.get_layer("multi_layer_att").output
      assert_equal(single_out.shape, (None, num_heads * value_dim))
      assert_equal(multi_out.shape, (None, num_heads * value_dim))
      feed_dict = make_feed_dict(network.extern_data.data.values(), same_time=True, n_time=n_time)
      session.run(tf_compat.v1.global_variables_initializer())

      single, multi = session.run([single_out.placeholder, multi_out.placeholder], feed_dict=feed_dict)
      print('single layer output:')
      pprint(single)
      print('multi layer output:')
      pprint(multi)
      numpy.testing.assert_almost_equal(single, multi, decimal=5)
      print('They are equal!')


def test_vanilla_self_attention_optimize_out():
  num_heads, key_dim, value_dim = 2, 3, 3
  network = {}
  add_vanilla_self_attention_layer(
    network, 'data:source', 'att', inside_rec_layer=True, past_only=True, time_axis='stag:extern_data:data',
    num_heads=num_heads, key_dim=key_dim, value_dim=value_dim)

  check_reclayer_optimize_out(
    {'class': 'copy', 'from': 'att_att', 'n_out': value_dim * num_heads},
    other_subnet_layers=network)


def test_full_lsh_cross_attention_construct():
  num_heads, key_dim, value_dim = 2, 3, 3
  net_dict = {
    'encoder': {'class': 'linear', 'n_out': 5, 'activation': None, 'from': 'data'},
    'output': {
      'class': 'rec',
      'target': 'classes',
      'from': [],
      'max_seq_len': 'max_len_from("base:encoder") * 3',
      'unit': {
        'embed': {'class': 'linear', 'activation': None, 'from': ['prev:output'], "n_out": 7},
        'att_att': None,
        'output_prob': {'class': 'softmax', 'from': 'att_att', 'target': 'classes'},
        'output': {
          'class': 'choice', 'beam_size': 4, 'target': 'classes', 'from': ['output_prob'], 'initial_output': 'zeros',
          'loss': 'ce', 'is_output_layer': True},
        "end": {'class': 'compare', 'from': ['output'], 'value': 0},
      }
    },
    'decision': {'class': 'decide', 'from': ['output'], 'loss': 'edit_distance', 'loss_opts': {}, 'target': 'classes'}
  }

  add_full_lsh_cross_attention_layer(
    d=net_dict['output']['unit'], db=net_dict, input='embed', keys_input='base:encoder', output='att',
    num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, num_hashes=6)
  pprint(net_dict)

  with make_scope():
    config = Config({"debug_print_layer_output_template": True, "debug_add_check_numerics_ops": True})
    config.set('extern_data', {
      "data": {"dim": 7, "sparse": True},
      "classes": {"dim": 6, "sparse": True, "available_for_inference": False}})
    net = TFNetwork(config=config, search_flag=False, train_flag=True, eval_flag=False)
    net.construct_from_dict(net_dict)
    print(net.layers)


def _test_lsh_cross_attention_equals_full_lsh_cross_attention(
    enc_time, dec_time, chunk_size, chunks_before, chunks_after, num_heads=2, num_hashes=6, chunk_align='identity',
    shuffle_kv=False):
  key_dim, value_dim = 3, 3
  net_dict = {
    'encoder': {'class': 'linear', 'n_out': 5, 'activation': None},
    'output': {
      'class': 'rec',
      'target': 'classes',
      'max_seq_len': dec_time,
      'from': [],
      'unit': {
        'embed': {'class': 'linear', 'activation': None, 'from': ['prev:output'], "n_out": 7},
        'chunked_att_att': None,
        'full_att_att': None,
        'output_prob': {'class': 'softmax', 'from': ['full_att_att', 'chunked_att_att'], 'target': 'classes'},
        'output': {
          'class': 'choice', 'beam_size': 4, 'target': 'classes', 'from': ['output_prob'], 'initial_output': 'zeros',
          'loss': 'ce', 'is_output_layer': True},
        "end": {'class': 'compare', 'from': ['output'], 'value': -1},  # should always be False.
      }
    },
    'chunked_att': {'class': 'copy', 'from': 'output/chunked_att_att', 'is_output_layer': True},
    'full_att': {'class': 'copy', 'from': 'output/full_att_att', 'is_output_layer': True},
    'decision': {'class': 'decide', 'from': ['output'], 'loss': 'edit_distance', 'loss_opts': {}, 'target': 'classes'}
  }  # type: dict[str, Any]

  add_full_lsh_cross_attention_layer(
    d=net_dict['output']['unit'], db=net_dict, input='embed', keys_input='base:encoder', output='full_att',
    num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, num_hashes=num_hashes, debug_print=False)
  add_lsh_cross_attention_layer(
    d=net_dict['output']['unit'], db=net_dict, input='embed', keys_input='base:encoder',
    output='chunked_att', num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, num_hashes=num_hashes,
    key_chunk_size=chunk_size, query_chunk_size=chunk_size, key_chunks_before=chunks_before,
    key_chunks_after=chunks_after, debug_print=False, chunk_alignment=chunk_align, shuffle_kv=shuffle_kv)
  net_dict['output']['unit']['chunked_att_att']['is_output_layer'] = True
  net_dict['output']['unit']['full_att_att']['is_output_layer'] = True
  net_dict['output']['unit']['chunked_att_query0']['reuse_params'] = 'full_att_query0'
  net_dict['chunked_att_key0']['reuse_params'] = 'full_att_key0'
  net_dict['chunked_att_value0']['reuse_params'] = 'full_att_value0'
  assert 'chunked_att_hash_gen_top_unnamed' in net_dict['output']['unit']
  assert 'full_att_hash_gen_top_unnamed' in net_dict
  net_dict['output']['unit']['chunked_att_hash_gen_top_unnamed'] = {
    'class': 'copy', 'from': 'base:full_att_hash_gen_top_unnamed'}

  with make_scope() as session:
    config = Config({"debug_print_layer_output_template": True, "debug_add_check_numerics_ops": True})
    config.set('extern_data', {
      "data": {"dim": 7, "sparse": True},
      "classes": {"dim": 6, "sparse": True, "available_for_inference": False}})
    net = TFNetwork(config=config, train_flag=True, search_flag=False, eval_flag=False)

    feed_dict = make_feed_dict(net.extern_data.data.values(), n_time=enc_time)
    net.construct_from_dict(net_dict)
    full_att_layer, chunked_att_layer = net.layers['full_att'], net.layers['chunked_att']
    assert full_att_layer.output.shape == chunked_att_layer.output.shape
    assert full_att_layer.output.get_time_dim_tag().is_equal(chunked_att_layer.output.get_time_dim_tag())
    session.run(tf_compat.v1.global_variables_initializer())
    assert full_att_layer.output.time_dim_axis == chunked_att_layer.output.time_dim_axis
    time_axis = full_att_layer.output.time_dim_axis

    full_att, chunked_att, full_time, chunked_time = session.run(
      [full_att_layer.output.placeholder, chunked_att_layer.output.placeholder,
        full_att_layer.output.get_dynamic_size(time_axis), chunked_att_layer.output.get_dynamic_size(time_axis)
      ], feed_dict=feed_dict)

    # mask away things out of seq length
    assert full_att_layer.output.batch_dim_axis == chunked_att_layer.output.batch_dim_axis == 0
    assert full_att_layer.output.time_dim_axis == chunked_att_layer.output.time_dim_axis == 1
    mask = numpy.arange(numpy.shape(full_att)[1]).reshape([1,-1,1]) < full_time.reshape([-1,1,1])
    full_att = full_att * mask
    chunked_att = chunked_att * mask

    print('Full LSH attention context vector:', full_att_layer.output)
    pprint(full_att)
    print('with seq lengths:', full_time)
    print('Chunked LSH attention context vector:', chunked_att_layer.output)
    pprint(chunked_att)
    print('with seq lengths:', chunked_time)

    from numpy.testing import assert_almost_equal
    assert_almost_equal(full_time, chunked_time)
    assert(not numpy.any(numpy.isnan(chunked_att)))

    assert_almost_equal(full_att, chunked_att, decimal=3)
    print("Attention context vectors are equal!")


def test_lsh_cross_attention_equals_full_lsh_cross_attention():
  _test_lsh_cross_attention_equals_full_lsh_cross_attention(
    enc_time=5, dec_time=1, chunk_size=6, chunks_before=0, chunks_after=0, num_heads=1, num_hashes=4)
  _test_lsh_cross_attention_equals_full_lsh_cross_attention(
    enc_time=15, dec_time=10, chunk_size=5, chunks_before=1, chunks_after=1)
  _test_lsh_cross_attention_equals_full_lsh_cross_attention(
    enc_time=5, dec_time=1, chunk_size=6, chunks_before=0, chunks_after=0, num_heads=1, num_hashes=4,
    chunk_align='search_bounds_centered')
  _test_lsh_cross_attention_equals_full_lsh_cross_attention(
    enc_time=15, dec_time=10, chunk_size=5, chunks_before=1, chunks_after=1, chunk_align='search_bounds_centered')


def test_lsh_cross_attention_equals_full_lsh_cross_attention_shuffle_kv():
  _test_lsh_cross_attention_equals_full_lsh_cross_attention(
    enc_time=5, dec_time=1, chunk_size=5, chunks_before=0, chunks_after=0, num_heads=1, num_hashes=4,
    chunk_align='identity', shuffle_kv=True)
  _test_lsh_cross_attention_equals_full_lsh_cross_attention(
    enc_time=5, dec_time=1, chunk_size=6, chunks_before=0, chunks_after=0, num_heads=1, num_hashes=4,
    chunk_align='search_bounds_centered', shuffle_kv=True)
  _test_lsh_cross_attention_equals_full_lsh_cross_attention(
    enc_time=15, dec_time=10, chunk_size=5, chunks_before=1, chunks_after=1, chunk_align='search_bounds_centered',
    shuffle_kv=True)


def _test_lsh_cross_attention_no_mask_different_hashes(
    n_time, mask_current, chunk_size, chunks_before, chunks_after, duplicates, n_batch=3,
    num_heads=2, key_dim=3, value_dim=3, num_hashes=26, chunk_align='identity', shuffle_kv=False):
  print(
    'Testing n_time =', n_time, 'mask_current =', mask_current, 'chunk_size =', chunk_size,
    'chunks_before =', chunks_before, 'chunks_after =', chunks_after, 'allow_duplicate_attention =', duplicates,
    'chunk_align =', chunk_align)
  with make_scope() as session:
    assert n_time <= chunk_size * (chunks_before + 1 + chunks_after), (
      'if chunk size is too small, vanilla attention != lsh attention')
    net_dict = {
      'encoder': {'class': 'linear', 'n_out': 5, 'activation': None},
      'output': {
        'class': 'rec',
        'target': 'classes',
        'max_seq_len': 'max_len_from("base:encoder") * 3',
        'from': [],
        'unit': {
          'embed': {'class': 'linear', 'activation': None, 'from': ['prev:output'], "n_out": 7},
          'lsh_att': None,
          'vanilla_att': None,
          'output_prob': {'class': 'softmax', 'from': ['lsh_att', 'vanilla_att'], 'target': 'classes'},
          'output': {
            'class': 'choice', 'beam_size': 4, 'target': 'classes', 'from': ['output_prob'], 'initial_output': 'zeros',
            'loss': 'ce', 'is_output_layer': True},
          "end": {'class': 'compare', 'from': ['output'], 'value': -1},  # should always be False.
        }
      },
      'vanilla_out': {'class': 'copy', 'from': 'output/vanilla_att', 'is_output_layer': True},
      'lsh_out': {'class': 'copy', 'from': 'output/lsh_att', 'is_output_layer': True},
      'decision': {'class': 'decide', 'from': ['output'], 'loss': 'edit_distance', 'loss_opts': {}, 'target': 'classes'}
    }  # type: dict[str, Any]
    add_lsh_cross_attention_layer(
      net_dict['output']['unit'], net_dict, input='embed', keys_input='base:encoder', output='lsh',
      num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, num_hashes=num_hashes,
      key_chunk_size=chunk_size, query_chunk_size=chunk_size, key_chunks_before=chunks_before,
      key_chunks_after=chunks_after, mask_different_hashes=False,
      allow_duplicate_attention=duplicates, chunk_alignment=chunk_align, shuffle_kv=shuffle_kv, debug_print=False)
    add_vanilla_cross_attention_layer(
      net_dict['output']['unit'], net_dict, input='embed', keys_input='base:encoder', output='vanilla',
      num_heads=num_heads, key_dim=key_dim, value_dim=value_dim)
    net_dict['output']['unit']['lsh_att']['is_output_layer'] = True
    net_dict['output']['unit']['vanilla_att']['is_output_layer'] = True
    net_dict['output']['unit']['lsh_query0']['reuse_params'] = 'vanilla_query0'
    net_dict['lsh_key0']['reuse_params'] = 'vanilla_key0'
    net_dict['lsh_value0']['reuse_params'] = 'vanilla_value0'

    config = Config({"debug_print_layer_output_template": True, "debug_add_check_numerics_ops": True})
    config.update(dict(num_inputs=num_heads*key_dim, num_outputs=num_heads*value_dim))
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)

    assert_equal(network.get_layer("vanilla_out").output.shape, (None, num_heads * value_dim))
    assert_equal(network.get_layer("lsh_out").output.shape, (None, num_heads * value_dim))
    feed_dict = make_feed_dict(network.extern_data.data.values(), same_time=True, n_batch=n_batch, n_time=n_time)
    session.run(tf_compat.v1.global_variables_initializer())

    input_out = network.get_layer("data").output
    vanilla_out = network.get_layer("vanilla_out").output
    lsh_out = network.get_layer("lsh_out").output
    vanilla, lsh, sizes, vanilla_sizes, lsh_sizes = session.run(
      [vanilla_out.placeholder, lsh_out.placeholder,
        input_out.size_placeholder[input_out.time_dim_axis_excluding_batch],
        vanilla_out.size_placeholder[vanilla_out.time_dim_axis_excluding_batch],
        lsh_out.size_placeholder[lsh_out.time_dim_axis_excluding_batch]],
      feed_dict=feed_dict)
    numpy.testing.assert_equal(vanilla_sizes, sizes)
    numpy.testing.assert_equal(lsh_sizes, sizes)
    # take into account different seq lengths
    assert vanilla_out.batch_dim_axis == lsh_out.batch_dim_axis == 0
    assert vanilla_out.time_dim_axis == lsh_out.time_dim_axis == 1
    mask = (numpy.arange(numpy.shape(vanilla)[1]).reshape([1,-1,1]) < sizes.reshape([-1,1,1]))
    vanilla = vanilla * mask
    lsh = lsh * mask
    print('seq lengths:', sizes)
    print('vanilla out:  - ', vanilla_out)
    pprint(vanilla)
    print('lsh out:  -', lsh_out)
    pprint(lsh)
    numpy.testing.assert_almost_equal(vanilla, lsh, decimal=5)
    print('They are equal!')


def test_lsh_cross_attention_no_mask_different_hashes():
  _test_lsh_cross_attention_no_mask_different_hashes(
    n_time=13, mask_current=False, chunk_size=5, chunks_before=1, chunks_after=1, duplicates=True)
  _test_lsh_cross_attention_no_mask_different_hashes(
    n_time=13, mask_current=True, chunk_size=5, chunks_before=1, chunks_after=1, duplicates=True)
  _test_lsh_cross_attention_no_mask_different_hashes(
    n_time=13, mask_current=False, chunk_size=5, chunks_before=1, chunks_after=1, duplicates=True, shuffle_kv=True)
  _test_lsh_cross_attention_no_mask_different_hashes(
    n_time=13, mask_current=True, chunk_size=5, chunks_before=1, chunks_after=1, duplicates=True, shuffle_kv=True)
