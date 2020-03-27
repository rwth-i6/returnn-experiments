import tensorflow as tf
from TFNetworkLayer import _ConcatInputLayer, Loss
import numpy as np


class HMMFactorization(_ConcatInputLayer):
  """
  Layer to use attention explicitly in the posterior distribution.
  """

  layer_class = "hmm_factorization"

  def __init__(self, attention_weights, base_encoder_transformed, prev_state, prev_outputs, n_out, debug=False,
               attention_location=None, transpose_and_average_att_weights=False, top_k=None,
               window_size=None, first_order_alignments=False, first_order_k=None, window_factor=1.0,
               tie_embedding_weights=None, prev_prev_state=None, sample_softmax=None,
               first_order_approx=False, sample_method="uniform", sample_use_bias=False, **kwargs):
    """
    Layer to use attention explicitly in the posterior distribution.
    :param LayerBase attention_weights: Attention weights layer.
    :param LayerBase base_encoder_transformed: Encoder layer.
    :param LayerBase prev_state: Previous (i.e. i-1, so current time-step) decoder state
    :param LayerBase prev_outputs: Previous embedded output data
    :param int n_out: Full vocabulary size.
    :param bool debug: Debug mode.
    :param str,None attention_location: Posterior attention weight saving.
    :param bool transpose_and_average_att_weights: Whether to apply appropriate modifications to the attention weights
    when coming from Transformer (True) or not (False).
    :param int,str,None top_k: topK setting for optimization. Can be int, or a string which calculates the number
    dynamically. Look at eval layer for usage examples.
    :param int window_size: Window width, should be even. window_size/2 is the size of window on each side.
    :param bool first_order_alignments: Whether to use posterior attention or not.
    :param bool first_order_approx: Whether to use a mixture of encoder hidden states (True) or as explicit calculation
    in the posterior (False).
    :param int,None first_order_k: First order topK optimization. If set to None, then uses top_k setting.
    :param int window_factor: Factor with which target step i is multiplied to center the window on the source side.
    :param LayerBase tie_embedding_weights: Layer for weight tying.
    :param LayerBase prev_prev_state: Layer for (i-2) state.
    :param int,None sample_softmax: If not None, then uses this number as to sample the softmax. Requires setting
    "hmm_factorization_sampled_loss" as the loss, and the same sampling setting there.
    :param str sample_method: Sample method, in {"uniform", "log_uniform", "learned_unigram"}
    :param bool sample_use_bias: Whether to use a bias term in the sampled softmax (True) or not (False).
    :param kwargs: kwargs.
    """

    super(HMMFactorization, self).__init__(**kwargs)

    # Define basic class instance variables
    self.iteration = 0
    self.batch_iteration = 0
    self.debug = debug
    self.in_loop = True if len(prev_state.output.shape) == 1 else False

    # Process input so that it is standardized and saved in the class instance
    self._process_input(attention_weights, base_encoder_transformed, prev_state, prev_outputs, prev_prev_state,
                        transpose_and_average_att_weights)

    # top_k management
    top_k, first_order_k = self._topk_preprocess(top_k, first_order_k)

    # Posterior attention management
    if first_order_alignments is True:
      self._first_order_processing(base_encoder_transformed, first_order_k, first_order_approx, attention_location)

    # Use only top_k from self.attention_weights
    if top_k is not None:
      top_indices = self._process_topk(window_size, top_k, window_factor)  # [(I,) B, 1, top_k]

    # Get size data
    attention_weights_shape = tf.shape(self.attention_weights)
    if self.in_loop is False:
      time_i = attention_weights_shape[0]
      batch_size = attention_weights_shape[2]
      time_j = attention_weights_shape[1]
    else:
      batch_size = attention_weights_shape[1]
      time_j = attention_weights_shape[0]
      time_i = None

    # Convert base_encoder_transformed, prev_state and prev_outputs to correct shape
    self._tile_dependencies(time_j, time_i)

    # Fix self.base_encoder_transformed if in top_k
    if top_k is not None:
      self._post_process_topk(top_indices, top_k, batch_size, time_i)

    if self.debug:
      self.base_encoder_transformed = tf.Print(self.base_encoder_transformed, [tf.shape(self.base_encoder_transformed),
                                                                               tf.shape(self.prev_state),
                                                                               tf.shape(self.prev_outputs)],
                                               message='Shapes of base encoder, prev_state '
                                                       'and prev_outputs post shaping: ',
                                               summarize=100)

    # Permute attention weights correctly
    perm_att_weights = [1, 2, 0] if self.in_loop else [0, 2, 3, 1]
    self.attention_weights = tf.transpose(self.attention_weights, perm=perm_att_weights)  # Now [(I,) B, 1, J]

    if self.debug:
      self.attention_weights = tf.Print(self.attention_weights, [tf.shape(self.attention_weights)],
                                        message='attention_weights shape transposed: ',
                                        summarize=100)

      self.base_encoder_transformed = tf.Print(self.base_encoder_transformed,
                                               [tf.shape(self.base_encoder_transformed + self.prev_outputs + self.prev_state)],
                                               message='Pre lex logits shape: ', summarize=100)

    if sample_softmax:
      lexicon_model = self._process_sampled_softmax(sample_use_bias, n_out, base_encoder_transformed, sample_softmax,
                                                    sample_method)  # [(I,) B, J, vocab_size]
    else:
      lexicon_model = self._process_lexicon_model(tie_embedding_weights, n_out)  # [(I,) B, J, vocab_size]

    if self.debug:
      lexicon_model = tf.Print(lexicon_model, [tf.shape(lexicon_model)], message='lexicon_model shape: ', summarize=100)

    # Multiply for final logits, [(I,) B, 1, J] x [(I,) B, J, vocab_size] ----> [(I,) B, 1, vocab]
    final_output = tf.matmul(self.attention_weights, lexicon_model)

    if self.debug:
      final_output = tf.Print(final_output, [tf.shape(final_output)], message='final_output shape: ', summarize=100)

    final_output = tf.squeeze(final_output, axis=-2)  # [(I,) B, vocab]

    if self.debug:
      final_output = tf.Print(final_output, [tf.shape(final_output)], message='final_output post squeeze shape: ',
                              summarize=100)

    self.output.placeholder = final_output

    # Add all trainable params
    with self.var_creation_scope() as scope:
      self._add_all_trainable_params(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name))

  def _process_lexicon_model(self, tie_embedding_weights, n_out):
    """
    Process the lexicon model in normal/weight tying case.
    :param TFNetworkLayer.LayerBase tie_embedding_weights: Layer for weight tying.
    :param int n_out: Vocab size.
    :return: tf.Tensor Processed lexicon model of shape [(I,) B, J, vocab_size]
    """

    # Get logits, now [(I,) J, B, vocab_size]
    if tie_embedding_weights is None:
      # When calculating logits directly
      lexicon_logits = tf.layers.dense(self.base_encoder_transformed + self.prev_outputs + self.prev_state,
                                       units=n_out,
                                       activation=None,
                                       use_bias=False)
    else:
      # When using weight tying
      lexicon_weight = tf.get_default_graph().get_tensor_by_name(
        tie_embedding_weights.get_base_absolute_name_scope_prefix() + "W:0")  # [vocab, emb]
      lexicon_weight = tf.transpose(lexicon_weight, perm=[1, 0])  # [emb, vocab]
      lexicon_logits = self.linear(x=self.base_encoder_transformed + self.prev_outputs + self.prev_state,
                                   weight=lexicon_weight,
                                   units=lexicon_weight.shape[1])  # [(I,) J, B, vocab_size]

    if self.debug:
      lexicon_logits = tf.Print(lexicon_logits, [tf.shape(lexicon_logits)], message='Post lex logits shape: ',
                                summarize=100)

    perm_logits = [1, 0, 2] if self.in_loop else [0, 2, 1, 3]
    lexicon_logits = tf.transpose(lexicon_logits, perm=perm_logits)  # Now [(I,) B, J, vocab_size]

    # Now [(I,) B, J, vocab_size], Perform softmax on last layer
    lexicon_model = tf.nn.softmax(lexicon_logits)
    return lexicon_model

  def _process_sampled_softmax(self, sample_use_bias, n_out, base_encoder_transformed, sample_softmax, sample_method):
    """
    Process lexicon model in sampled softmax case.
    :param bool sample_use_bias: Whether to add bias term.
    :param int n_out: Full vocab size.
    :param tf.Tensor base_encoder_transformed: Raw encoder states of shape [(I,) J, B, f]
    :param int sample_softmax: Sample size.
    :param str sample_method: Sample method, in {"uniform", "log_uniform", "learned_unigram"}
    :return: tf.Tensor Processed lexicon model of shape [(I,) B, J, vocab_size]
    """

    # [vocab_size, emb]
    lexicon_weight = tf.get_variable("sampled_lexicon_weight", shape=[n_out, base_encoder_transformed.output.shape[-1]])

    if sample_use_bias:
      lexicon_bias = tf.get_variable("sampled_lexicon_bias", shape=[n_out])
    else:
      lexicon_bias = None

    # Get targets
    targets = None
    targets_flat = None
    if "classes" in self.network.extern_data.data:
      if self.in_loop:
        targets_flat = self.network.extern_data.data['classes'].placeholder
        targets = targets_flat
      else:
        targets = self.network.extern_data.data['classes'].placeholder
        targets_flat = self._flatten_or_merge(targets,
                                              seq_lens=self.network.extern_data.data['classes'].get_sequence_lengths(),
                                              time_major=False)
    # Input is of shape [(I,) J, B, f]
    lexicon_model = self.sampled_softmax(inp=self.base_encoder_transformed + self.prev_outputs + self.prev_state,
                                         weight=lexicon_weight,
                                         num_samples=sample_softmax,
                                         full_vocab_size=n_out,
                                         targets=targets,
                                         targets_flat=targets_flat,
                                         full_softmax=self.network.search_flag,
                                         bias=lexicon_bias,
                                         sample_method=sample_method,
                                         )  # [(I,) B, J, vocab_size]

    return lexicon_model

  def _post_process_topk(self, top_indices, top_k, batch_size, time_i):
    """
    Post process operation to extract corresponding encoder states from topK
    :param tf.Tensor top_indices: From topK, [(I,) B, 1, top_k].
    :param int top_k: topK size.
    :param tf.Tensor[int] batch_size: Dynamic batch size.
    :param tf.Tensor[int] time_i: Dynamic target time size.
    """

    # Process inputs into correct shapes
    perm_enc_1 = [1, 0, 2] if self.in_loop else [0, 2, 1, 3]
    self.base_encoder_transformed = tf.transpose(self.base_encoder_transformed, perm=perm_enc_1)  # Now [(I,) B, J, f]
    top_indices = tf.squeeze(top_indices, axis=-2)

    # Generate appropriate indices
    if self.in_loop is False:
      ii, jj, _ = tf.meshgrid(tf.range(time_i), tf.range(batch_size), tf.range(top_k), indexing='ij')  # [I, B, k]
      # Stack complete index
      index = tf.stack([ii, jj, top_indices], axis=-1)  # [I, B, k, 3]
    else:
      jj, _ = tf.meshgrid(tf.range(batch_size), tf.range(top_k), indexing='ij')
      # Stack complete index
      index = tf.stack([jj, top_indices], axis=-1)

    # Extract the corresponding encoder data from the topK op
    self.base_encoder_transformed = tf.gather_nd(self.base_encoder_transformed, index)

    # Transpose into correct shape
    perm_enc_2 = [1, 0, 2] if self.in_loop else [0, 2, 1, 3]
    self.base_encoder_transformed = tf.transpose(self.base_encoder_transformed, perm=perm_enc_2)  # [(I,) J, B, f]

  def _tile_dependencies(self, time_j, time_i):
    """
    Tile the internal states so that they're of correct shapes.
    :param tf.Tensor[int] time_j: Dynamic source time size.
    :param tf.Tensor[int] time_i: Dynamic target time size.
    """

    # Convert base_encoder_transformed, prev_state and prev_outputs to correct shape
    if self.in_loop is False:
      self.base_encoder_transformed = tf.tile(tf.expand_dims(self.base_encoder_transformed, axis=0),
                                              [time_i, 1, 1, 1])  # [I, J, B, intermediate_size]

      self.prev_state = tf.tile(tf.expand_dims(self.prev_state, axis=1),
                                [1, time_j, 1, 1])  # [I, J, B, intermediate_size]

      self.prev_outputs = tf.tile(tf.expand_dims(self.prev_outputs, axis=1),
                                  [1, time_j, 1, 1])  # [I, J, B, intermediate_size]
    else:
      self.base_encoder_transformed = tf.transpose(self.base_encoder_transformed,
                                                   perm=[1, 0, 2])  # [J, B, f]

      self.prev_state = tf.tile(tf.expand_dims(self.prev_state, axis=0),
                                [time_j, 1, 1])  # [J, B, f]
      self.prev_outputs = tf.tile(tf.expand_dims(self.prev_outputs, axis=0),
                                  [time_j, 1, 1])  # [J, B, f]

  def _process_topk(self, window_size, top_k, window_factor):
    """
    Extracts the topK most relevant source indices, modifies the attention weights and applies windowing if needed.
    :param int window_size: Window width, should be even. window_size/2 is the size of window on each side.
    :param tf.Tensor[int] top_k: K size
    :param int window_factor: Factor with which target step i is multiplied to center the window on the source side.
    :return: tf.Tensor The corresponding topK indices of shape [(I,) B, 1, top_k]
    """

    perm_att_1 = [1, 2, 0] if self.in_loop else [0, 2, 3, 1]
    temp_attention_weights = tf.transpose(self.attention_weights, perm=perm_att_1)  # Now [(I,) B, 1, J]

    if window_size is not None:
      temp_attention_weights = self._process_window(window_size, top_k, temp_attention_weights, window_factor)

    top_values, top_indices = tf.nn.top_k(temp_attention_weights, k=top_k)  # top_values and indices [(I,) B, 1, top_k]

    if self.debug:
      top_indices = tf.Print(top_indices, [top_indices], message="top_indices eg", summarize=20)

    perm_att_2 = [2, 0, 1] if self.in_loop else [0, 3, 1, 2]
    self.attention_weights = tf.transpose(top_values, perm=perm_att_2)  # Now [I, J=top_k, B, 1]

    if self.debug:
      self.attention_weights = tf.Print(self.attention_weights, [tf.shape(self.attention_weights)],
                                        message='Top K Attention weight shape: ', summarize=100)

    return top_indices

  def _process_window(self, window_size, top_k, temp_attention_weights, window_factor):
    """
    Applied a window mask on the attention weights.
    :param int window_size: Window width, should be even. window_size/2 is the size of window on each side.
    :param tf.Tensor[int] top_k: K size.
    :param tf.Tensor temp_attention_weights: The current attention weights [(I,) B, 1, J].
    :param int window_factor: Factor with which target step i is multiplied to center the window on the source side.
    :return: tf.Tensor Masked attention weights of shape [(I,) B, 1, J].
    """

    assert window_size % 2 == 0, "HMM Factorization: Window size has to be divisible by 2!"
    if isinstance(top_k, int):
      assert top_k <= window_size, "HMM Factorization: top_k can be maximally as large as window_size!"

    if self.in_loop is False:
      # Example:
      # window_size = 2, I=J=4:
      # [0, 0] = 1, [0, 1] = 1, [0, 2] = 0, [0, 3] = 0
      # [1, 0] = 1, [1, 1] = 1, [1, 2] = 1, [1, 3] = 0
      # [2, 0] = 0, [2, 1] = 1, [2, 2] = 1, [2, 3] = 1
      # [3, 0] = 0, [3, 1] = 0, [3, 2] = 1, [3, 3] = 1

      # Get shapes
      sh = tf.shape(temp_attention_weights)
      ti = sh[0]
      tj = sh[3]
      b = sh[1]

      # mask = tf.matrix_band_part(tf.ones([ti, tj], dtype=tf.bool), int(window_size/2), int(window_size/2))  # [I, J]

      # Shifting with window_factor, quite hacky using a precomputed mask. If using longer sequences set max_i to
      # the length of your longest target sequence.
      def static_mask_graph(max_i=500):
        # make static mask, is only done once during graph creation
        max_j = np.int64(np.floor(max_i * window_factor))
        matrices = []
        for i in range(max_i):
          i = np.int64(np.floor(i * window_factor))
          new_m = np.concatenate([np.zeros([np.maximum(i - int(window_size / 2), 0)], dtype=np.bool),
                                  np.ones([window_size + 1 - np.maximum(-i + int(window_size / 2), 0) - np.maximum(
                                    (i + int(window_size / 2)) - (max_j - 1), 0)], dtype=np.bool),
                                  np.zeros([np.maximum(np.minimum(max_j - i - int(window_size / 2), max_j) - 1, 0)],
                                           dtype=np.bool)],
                                 axis=0)
          matrices.append(new_m)
        mask = np.stack(matrices)
        return mask

      mask = static_mask_graph()

      # The following lines are to make the behaviour of in loop and outer loop match
      f = tf.cast(tf.floor(tf.cast(tj, tf.float32) / window_factor), tf.int32)
      mask_1 = tf.slice(mask, [0, 0], [f, tj])
      mask_2 = tf.concat([tf.zeros([tj - int(window_size / 2), ti - f], dtype=tf.bool),
                          tf.ones([int(window_size / 2), ti - f], dtype=tf.bool)], axis=0)
      mask_2 = tf.transpose(mask_2, perm=[1, 0])
      mask = tf.concat([mask_1, mask_2], axis=0)

      # Then use expand_dims to make it 4D
      mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)  # Now [I, 1, 1, J]
      mask = tf.tile(mask, [1, b, 1, 1])  # Now [I, B, 1, J]

      # Mask temp_attention_weights to 0
      temp_attention_weights = tf.where(mask, temp_attention_weights, tf.zeros_like(temp_attention_weights))
    else:
      sh = tf.shape(temp_attention_weights)
      tj = sh[2]
      b = sh[0]

      i = tf.get_default_graph().get_tensor_by_name('output/rec/while/Identity:0')  # super hacky, get current step
      i = tf.cast(tf.floor(tf.cast(i, tf.float32) * window_factor), tf.int32)  # Shift by window_factor
      i = tf.minimum(i, tj)

      mask = tf.concat([tf.zeros([tf.maximum(i - int(window_size / 2), 0)], dtype=tf.bool),
                        tf.ones([window_size + 1 - tf.maximum(-i + int(window_size / 2), 0) - tf.maximum(
                          (i + int(window_size / 2)) - (tj - 1), 0)], dtype=tf.bool),
                        tf.zeros([tf.maximum(tf.minimum(tj - i - int(window_size / 2), tj) - 1, 0)], dtype=tf.bool)],
                       axis=0)  # Shape [J]

      mask = tf.expand_dims(tf.expand_dims(mask, axis=0), axis=0)  # Now [1, 1, J]
      mask = tf.tile(mask, [b, 1, 1])

      # mask is now [B, 1, J], having true only in J where j=i +/- window_size
      # Mask temp_attention_weights to 0
      temp_attention_weights = tf.where(mask, temp_attention_weights, tf.zeros_like(temp_attention_weights))

    return temp_attention_weights

  def _first_order_processing(self, base_encoder_transformed, first_order_k, first_order_approx, attention_location):
    """
    Processes the posterior attention modifications.
    :param tf.Tensor base_encoder_transformed: Raw encoder states of shape [(I,) J, B, f]
    :param tf.Tensor[int] first_order_k: The posterior attention topK setting.
    :param bool first_order_approx: Whether to use a mixture of encoder hidden states (True) or as explicit calculation
    in the posterior (False).
    :param str attention_location: Used for saving posterior attention weights in debugging.
    """

    # Get shape data
    s = tf.shape(self.base_encoder_transformed)
    batch_size = s[0] if self.in_loop else s[1]
    time_i = None if self.in_loop else tf.shape(self.prev_state)[0]

    # Posterior attention
    prev_prev_output_and_decoder = self.prev_prev_state + self.prev_outputs  # [(I,) B, f]
    prev_prev_output_and_decoder_exp = tf.expand_dims(prev_prev_output_and_decoder, axis=-3)  # [(I,) 1, B, f]
    # Note: we use the auto-broadcast function of "+" to make the tiling ops

    encoder_tr = self.base_encoder_transformed  # [J, B, f] self.in_loop: [B, J, f]
    if self.in_loop is False:
      encoder_h1 = tf.expand_dims(encoder_tr, axis=0)  # [1, J', B, f]
    else:
      encoder_tr = tf.transpose(encoder_tr, perm=[1, 0, 2])  # [J, B, f]
      encoder_h1 = encoder_tr  # [J', B, f]
    encoder_h = encoder_h1  # [(1,) J, B, f]

    post_attention = prev_prev_output_and_decoder_exp + encoder_h1  # [(I,) J', B, f]
    post_attention = tf.layers.dense(post_attention, units=base_encoder_transformed.output.shape[-1],
                                     activation=tf.nn.tanh, use_bias=False)  # TODO: check if this is how we want it
    post_attention = tf.layers.dense(post_attention, units=1, activation=None, use_bias=False,
                                     name="post_att")  # [(I,) J', B, 1]
    post_attention = tf.nn.softmax(post_attention, axis=-3, name="post_att_softmax")  # [(I,) J', B, 1]

    # topk on posterior attention
    if self.in_loop is False:
      post_attention_topk = tf.transpose(post_attention, perm=[0, 2, 3, 1])  # [I, B, 1, J']
      post_attention_topk, post_top_indices = tf.nn.top_k(post_attention_topk, k=first_order_k)  # Both [I, B, 1, top_k]
      post_attention_topk = tf.squeeze(post_attention_topk, axis=-2)  # [I, B, top_k=J']
      post_top_indices = tf.squeeze(post_top_indices, axis=2)  # [I, B, top_k]
      ii, bb, _ = tf.meshgrid(tf.range(time_i), tf.range(batch_size), tf.range(first_order_k),
                              indexing='ij')  # [I, B, k]
      post_indices = tf.stack([ii, bb, post_top_indices], axis=-1)  # [I, B, k, 3]

      if self.debug:
        post_indices = tf.Print(post_indices, [post_top_indices], message="post_top_indices", summarize=20)

      encoder_h2 = tf.tile(tf.expand_dims(tf.transpose(encoder_tr, perm=[1, 0, 2]), axis=0),
                           [time_i, 1, 1, 1])  # [I, B, J, f]
      encoder_h2_dir = tf.gather_nd(encoder_h2, post_indices)  # [I, B, top_k=J', f]
      encoder_h2 = tf.expand_dims(encoder_h2_dir, axis=1)  # [I, 1, B, top_k=J', f]
    else:
      post_attention_topk = tf.transpose(post_attention, perm=[1, 2, 0])  # [B, 1, J']
      post_attention_topk, post_top_indices = tf.nn.top_k(post_attention_topk, k=first_order_k)  # Both [B, 1, top_k]
      post_attention_topk = tf.squeeze(post_attention_topk, axis=-2)  # [B, top_k=J']
      post_top_indices = tf.squeeze(post_top_indices, axis=1)  # [B, top_k]
      bb, _ = tf.meshgrid(tf.range(batch_size), tf.range(first_order_k), indexing='ij')  # [B, k]
      post_indices = tf.stack([bb, post_top_indices], axis=-1)  # [B, k, 2]
      if self.debug:
        post_indices = tf.Print(post_indices, [post_indices, post_top_indices],
                                message="post_indices, post_top_indices", summarize=20)
      encoder_h2 = tf.transpose(encoder_tr, perm=[1, 0, 2])  # [B, J, f]
      encoder_h2_dir = tf.gather_nd(encoder_h2, post_indices)  # [B, top_k=J', f]
      encoder_h2 = tf.expand_dims(encoder_h2_dir, axis=0)  # [1, B, top_k=J', f]

    # First order attention
    prev_output_and_decoder = self.prev_state + self.prev_outputs  # [(I,) B, f]
    if first_order_approx:
      # first get c_i
      if self.in_loop:
        c = tf.einsum("bj,bjf->bf", post_attention_topk, encoder_h2_dir)  # Now [B, f]
      else:
        c = tf.einsum("ibj,ibjf->ibf", post_attention_topk, encoder_h2_dir)  # Now [I, B, f]

      # Merge with target e
      prev_output_and_decoder += c  # [(I,) B, f]
      prev_output_and_decoder = tf.expand_dims(prev_output_and_decoder, axis=-3)  # [(I,) 1, B, f]

      # add, FF, softmax
      att = encoder_h + prev_output_and_decoder  # [(I,) J, B, f]
      att = tf.layers.dense(att, units=1, activation=None, use_bias=False)  # [(I,) J, B, 1]
      att = tf.nn.softmax(att, axis=-3)  # [(I,) J, B, 1]

      # set
      self.attention_weights = att  # [(I,) J, B, 1]

    else:
      # First order attention
      prev_output_and_decoder_exp = tf.expand_dims(prev_output_and_decoder, axis=-3)  # [(I,) 1, B, f]

      first_order_att = prev_output_and_decoder_exp + encoder_h  # Additive attention  [(I,) J, B, f]
      first_order_att = tf.expand_dims(first_order_att, axis=-2)  # [(I,) J, B, 1, f]
      first_order_att = first_order_att + encoder_h2  # [(I,) J, B, top_k=J', f]
      first_order_att = tf.layers.dense(first_order_att, units=base_encoder_transformed.output.shape[-1],
                                        activation=tf.nn.tanh, use_bias=False)
      first_order_att = tf.layers.dense(first_order_att, units=1, activation=None,
                                        use_bias=False)  # [(I,) J, B, top_k=J', 1]
      first_order_att = tf.nn.softmax(first_order_att, axis=-4, name="fo_softmax")  # [(I,) J, B, top_k=J', 1]
      first_order_att = tf.squeeze(first_order_att, axis=-1)  # [(I,) J, B, top_k=J']

      # Combine together
      if self.in_loop is False:
        self.attention_weights = tf.einsum('ibk,ijbk->ijbk', post_attention_topk,
                                           first_order_att)  # [I, J, B, top_k=J']
      else:
        self.attention_weights = tf.einsum('bk,jbk->jbk', post_attention_topk,
                                           first_order_att)  # [J, B, top_k=J']

      self.attention_weights = tf.reduce_sum(self.attention_weights, axis=-1, keep_dims=True)  # [(I,) J, B, 1]

      if self.debug:
        if self.in_loop:
          self.attention_weights = \
            tf.Print(self.attention_weights, [tf.reduce_sum(tf.transpose(self.attention_weights, perm=[1, 0, 2]),
                                                           axis=-2)[0],
                     tf.transpose(self.attention_weights, perm=[1, 0, 2])[0]], summarize=1000,
                     message="self.attention_weights sum and eg")
        else:
          self.attention_weights = \
            tf.Print(self.attention_weights,
                                            [tf.reduce_sum(tf.transpose(self.attention_weights, perm=[0, 2, 1, 3]),
                                                           axis=-2)[0],
                                             tf.transpose(self.attention_weights, perm=[0, 2, 1, 3])[0, 0]],
                     summarize=1000,
                     message="self.attention_weights sum and eg")

      if attention_location is not None:
        if self.in_loop:
          i = tf.get_default_graph().get_tensor_by_name('output/rec/while/Identity:0')  # super hacky, get current step
        else:
          i = None

        if i is not None:
          self.attention_weights = tf.py_func(func=self.save_tensor, inp=[self.attention_weights, attention_location,
                                                                          self.network.global_train_step,
                                                                          post_attention, i],
                                              Tout=tf.float32, stateful=True)
        else:
          self.attention_weights = tf.py_func(func=self.save_tensor, inp=[self.attention_weights, attention_location,
                                                                          self.network.global_train_step,
                                                                          post_attention],
                                              Tout=tf.float32, stateful=True)

  def _topk_preprocess(self, top_k, first_order_k):
    """
    Applies preprocessing steps to standardize both the topK and first_order_k operations.
    :param top_k: topK setting
    :param first_order_k: Posterior attention topK setting.
    :return: Modified top_K, first_order_k
    """

    assert isinstance(top_k, int) or isinstance(top_k, str), "HMM factorization: top_k of wrong format"

    if first_order_k is None and top_k is not None:
      first_order_k = top_k

    # if we want dynamic top_k
    if isinstance(top_k, str):
      import TFUtil
      vs = vars(TFUtil).copy()
      vs.update({"tf": tf, "self": self})
      top_k = eval(top_k, vs)
      if self.debug:
        top_k = tf.Print(top_k, [top_k], message="Dynamically calculated top k (may be overriden): ")

    # max cut top_k
    top_k = tf.minimum(top_k, tf.shape(self.base_encoder_transformed)[1 if self.in_loop else 0])

    # if we want dynamic top_k
    if isinstance(first_order_k, str):
      import TFUtil
      vs = vars(TFUtil).copy()
      vs.update({"tf": tf, "self": self})
      first_order_k = eval(first_order_k, vs)
      if self.debug:
        first_order_k = tf.Print(first_order_k, [first_order_k],
                                 message="Dynamically calculated top k (may be overriden): ")

    # max cut top_k
    first_order_k = tf.minimum(first_order_k, tf.shape(self.base_encoder_transformed)[1 if self.in_loop else 0])

    return top_k, first_order_k

  def _process_input(self, attention_weights, base_encoder_transformed, prev_state, prev_outputs, prev_prev_state,
                     transpose_and_average_att_weights):
    """
    Processes the input data to be of correct format.
    :param TFNetworkLayer.LayerBase attention_weights: Attention weights layer.
    :param TFNetworkLayer.LayerBase base_encoder_transformed:  Encoder layer.
    :param TFNetworkLayer.LayerBase prev_state: Previous (current, i-1) states layer.
    :param TFNetworkLayer.LayerBase prev_outputs: Previous outputs layer.
    :param TFNetworkLayer.LayerBase prev_prev_state: i-2 states layer
    :param bool transpose_and_average_att_weights: Whether to apply averaging for Transformer.
    """

    # Get data
    if self.in_loop is False:
      self.attention_weights = \
        attention_weights.output.get_placeholder_as_time_major()  # [J, B, H/1, I]/for rnn: [I, B, H/1, J]
      self.base_encoder_transformed = base_encoder_transformed.output.get_placeholder_as_time_major()  # [J, B, f]
      self.prev_state = prev_state.output.get_placeholder_as_time_major()  # [I, B, f]
      self.prev_outputs = prev_outputs.output.get_placeholder_as_time_major()  # [I, B, f]
      if prev_prev_state:
        self.prev_prev_state = prev_prev_state.output.get_placeholder_as_time_major()  # [I, B, f]
    else:
      self.attention_weights = attention_weights.output.get_placeholder_as_batch_major()  # [B, (H,) 1, J]
      self.base_encoder_transformed = base_encoder_transformed.output.get_placeholder_as_batch_major()  # [B, J, f]
      self.prev_state = prev_state.output.get_placeholder_as_batch_major()  # [B, intermediate_size]
      self.prev_outputs = prev_outputs.output.get_placeholder_as_batch_major()  # [B, intermediate_size]
      if prev_prev_state:
        self.prev_prev_state = prev_prev_state.output.get_placeholder_as_batch_major()  # [B, f]

    if self.debug:
      self.attention_weights = tf.Print(self.attention_weights, [tf.shape(self.attention_weights)],
                                        message='Attention weight shape: ', summarize=100)

    if self.debug:
      self.base_encoder_transformed = tf.Print(self.base_encoder_transformed, [tf.shape(self.base_encoder_transformed),
                                                                               tf.shape(self.prev_state),
                                                                               tf.shape(self.prev_outputs)],
                                               message='Shapes of base encoder, prev_state and prev_outputs pre shaping: ',
                                               summarize=100)

    # Transpose and average out attention weights (for when we use transformer architecture)
    if transpose_and_average_att_weights is True:
      if self.in_loop is False:
        # attention_weights is [J, B, H, I]
        self.attention_weights = tf.transpose(self.attention_weights, perm=[3, 0, 1, 2])  # Now it is [I, J, B, H]
        self.attention_weights = tf.reduce_mean(self.attention_weights, keep_dims=True, axis=3)  # Now [I, J, B, 1]
      else:
        # attention_weights is [B, H, 1, J]   (old: [J, B, H, 1?] )
        self.attention_weights = tf.squeeze(self.attention_weights, axis=-2)  # [B, H, J]
        self.attention_weights = tf.reduce_mean(self.attention_weights, keep_dims=True, axis=1)  # [B, 1, J]
        self.attention_weights = tf.transpose(self.attention_weights, perm=[2, 0, 1])  # Now it is [J, B, 1]
    else:
      if self.in_loop is True:
        # Edge case of attention weights
        self.attention_weights = tf.transpose(self.attention_weights, perm=[2, 0, 1])  # Now [J, B, 1]
      else:
        self.attention_weights = tf.transpose(self.attention_weights, perm=[0, 3, 1, 2])  # [I, J, B, 1]

    if self.debug:
      self.attention_weights = tf.Print(self.attention_weights, [tf.shape(self.attention_weights)],
                                        message='Attention weight shape after processing: ', summarize=100)

  def linear(self, x, units, inp_dim=None, weight=None, bias=False):
    """
    An optimized GPU linear layer.
    :param tf.Tensor x: Input
    :param int units: Output feature dim.
    :param int inp_dim: If weight not provided, then the input dimension
    :param tf.Tensor weight: If set, then use this weight to multiply, else will generate custom trainable weight.
    :param bool,tf.Tensor bias: Provide a tensor to use as bias or set to true for auto generated trainable bias.
    :return: Output tensor.
    """

    in_shape = tf.shape(x)
    inp = tf.reshape(x, [-1, in_shape[-1]])
    if weight is None:
      weight = tf.get_variable("lin_weight", trainable=True, shape=[inp_dim, units], dtype=tf.float32)
    out = tf.matmul(inp, weight)
    if bias is True or bias is not None:
      if bias is True:
        bias = tf.get_variable("lin_bias", trainable=True, initializer=tf.zeros_initializer, shape=[units],
                               dtype=tf.float32)
      out = out + bias
    out_shape = tf.concat([in_shape[:-1], [units]], axis=0)
    out = tf.reshape(out, out_shape)
    return out

  def sampled_softmax(self, inp, weight, num_samples, full_vocab_size, bias=None, full_softmax=False, sample_method="uniform",
                      targets=None, targets_flat=None):
    """
    Applies sampled softmax.
    :param tf.Tensor inp: Input tensors.
    :param tf.Tensor weight: Weight matrix of output layer.
    :param int num_samples: Number of samples to select.
    :param int full_vocab_size: Full vocab size.
    :param tf.Tensor bias: Bias tensor to use.
    :param bool full_softmax: Whether to compute the full softmax (True), or to use samples approach (False)
    :param str sample_method: Sample method, in {"uniform", "log_uniform", "learned_unigram"}
    :param tf.Tensor targets: Tensor containing target indices.
    :param tf.Tensor targets_flat: Targets correctly flattened out.
    :return: Appropriate posterior, of shape [I, B, J, num_samples].
    """

    if full_softmax:
      # full softmax version
      logits = self.linear(x=inp, units=full_vocab_size, weight=tf.transpose(weight, perm=[1, 0]), bias=bias)
      posterior = tf.nn.softmax(logits, axis=-1)
      return tf.transpose(posterior, perm=[1, 0, 2] if self.in_loop else [0, 2, 1, 3])
    else:
      # flatten inp
      weight_shape = tf.shape(weight)  # [vocab_size, emb]
      input_shape = tf.shape(inp)  # [I, J=k, B, emb]
      input_shape = tf.gather(input_shape, [1, 0, 2] if self.in_loop else [0, 2, 1, 3])
      inp = tf.reshape(inp, shape=[-1, weight_shape[1]])  # [B * I * J, emb]

      # get sample
      sampler_dic = {"uniform": tf.nn.uniform_candidate_sampler,
                     "log_uniform": tf.nn.log_uniform_candidate_sampler,
                     "learned_unigram": tf.nn.learned_unigram_candidate_sampler,}
      sampler = sampler_dic[sample_method]

      targets = tf.reshape(targets, shape=[-1, 1])  # [(T *) B, 1]
      targets = tf.cast(targets, tf.int64)
      sampled, _, _ = sampler(true_classes=targets,
                              num_true=1,
                              num_sampled=num_samples,
                              unique=True,
                              range_max=full_vocab_size,
                              )

      if targets_flat is not None:
        targets_flat, _ = tf.unique(targets_flat)
        targets_flat = tf.cast(targets_flat, tf.int64)
        sampled = tf.concat([sampled, targets_flat], axis=0)
        num_samples = tf.shape(sampled)[0]

      if self.debug:
        sampled = tf.Print(sampled, [num_samples], message="Sampled softmax: number of samples taken: ")

      # get weight for samples
      weight = tf.nn.embedding_lookup(weight, sampled)  # [num_samples, emb]
      if bias is not None:
        bias = tf.nn.embedding_lookup(bias, sampled)

      # [B * I * J, emb] x [emb, num_samples] -> [B * I * J, num_samples]
      logits = tf.matmul(inp, weight, transpose_b=True)
      if bias is not None:
        logits += bias

      # reshape back
      output_shape = tf.concat([input_shape[:-1], [num_samples]], axis=0)
      logits = tf.reshape(logits, shape=output_shape)  # [I, B, J,  num_samples]

      # normalize with softmax
      distribution = tf.nn.softmax(logits, axis=-1)

      # [I, B, J, num_samples]
      return distribution

  def _add_all_trainable_params(self, tf_vars):
    """
    Adds all trainable tensorflow variables to save.
    :param list[tf.Variable] tf_vars: Variables to save.
    :return:
    """
    for var in tf_vars:
      self.add_param(param=var, trainable=True, saveable=True)

  def _flatten_or_merge(self, x, seq_lens, time_major):
    """
    Copy of the function from Loss
    :param tf.Tensor x: (B,T,...) or (T,B,...)
    :param tf.Tensor seq_lens: (B,)
    :param bool time_major:
    :return: (B*T|T*B|B',...)
    :rtype: tf.Tensor
    """
    from TFUtil import flatten_with_seq_len_mask
    return flatten_with_seq_len_mask(x, seq_lens, time_major=time_major)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    d["from"] = d["prev_state"]
    if "threshold" in d:
      del d["threshold"]

    super(HMMFactorization, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["attention_weights"] = get_layer(d["attention_weights"])
    d["base_encoder_transformed"] = get_layer(d["base_encoder_transformed"])
    d["prev_state"] = get_layer(d["prev_state"])
    if "prev_prev_state" in d:
      d["prev_prev_state"] = get_layer(d["prev_prev_state"])
    d["prev_outputs"] = get_layer(d["prev_outputs"])
    if "tie_embedding_weights" in d:
      d["tie_embedding_weights"] = get_layer(d["tie_embedding_weights"])

  def save_tensor(self, attention_tensor, location, global_train_step, posterior_attention=None, i_step=None):
    """
    Hacky way to save tensor.
    :param attention_tensor: Attention tensor.
    :param location: Location to save.
    :param global_train_step: Global training save.
    :param posterior_attention: Posterior attention tensor.
    :param i_step: Current sub step, can be None.
    """
    # save tensor to file location
    d = {}
    d["i_step"] = i_step
    d["global_train_step"] = global_train_step
    d["shape"] = attention_tensor.shape
    d["attention_tensor"] = attention_tensor
    d["posterior_attention"] = posterior_attention

    if i_step is not None:
      if i_step == 0:
        self.batch_iteration += 1
    else:
      self.batch_iteration += 1

    np.save(str(location.decode("utf-8")) + '/' + str(self.batch_iteration) + "_" + str(i_step) +'_attention.npy', d)

    self.iteration += 1
    return attention_tensor


class HMMFactorizationSampledSoftmaxLoss(Loss):
  """
  Loss which should be used when using the sampled softmax configuration of the hmm_factorization layer.
  """

  class_name = "hmm_factorization_sampled_loss"

  def __init__(self, num_sampled, debug=False, safe_log_opts=None, **kwargs):
    """
    Loss which should be used when using the sampled softmax configuration of the hmm_factorization layer.
    :param int num_sampled: The amount sampled. Should be set to the same value as the setting in hmm_factorization
    layer.
    :param bool debug: Whether to enable debug mode.
    :param dict[str] safe_log_opts: passed to :func:`safe_log`
    """
    super(HMMFactorizationSampledSoftmaxLoss, self).__init__(**kwargs)
    self.num_sampled = num_sampled
    self.debug = debug
    self.safe_log_opts = safe_log_opts or {}

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    assert "num_sampled" in d, "hmm_factorization_sampled_loss: num_sampled parameter should be set to the same value" \
                               "as in the configuration of the hmm_factorization layer!"

  def _flatten_or_merge(self, x, seq_lens, time_major):
    """
    :param tf.Tensor x: (B,T,...) or (T,B,...)
    :param tf.Tensor seq_lens: (B,)
    :param bool time_major:
    :return: (B*T|T*B|B',...)
    :rtype: tf.Tensor
    """
    from TFUtil import flatten_with_seq_len_mask
    return flatten_with_seq_len_mask(x, seq_lens, time_major=time_major)

  def get_value(self):
    """
    Processes the loss.
    :return: Unnormzlized loss.
    """
    from TFUtil import safe_log
    output = self.output_flat

    output = output[:, self.num_sampled:]  # We assume that the targets are in the last part of the feature dims

    # remove duplicates
    target_flat_unique, target_flat_idx = tf.unique(self.target_flat)
    target_flat_idx = tf.stack([tf.range(tf.shape(target_flat_idx)[0]), target_flat_idx], axis=1)

    output = tf.gather_nd(output, target_flat_idx)

    out = -safe_log(output, **self.safe_log_opts)

    return self.reduce_func(out)

  def get_error(self):
    """
    Processes the error. Note, that generally this is lower than the true error.
    :return: Unnormzlized error.
    """
    output = self.output_flat

    output = output[:, self.num_sampled:]  # We assume that the targets are in the last part of the feature dims

    # remove duplicates
    target_flat_unique, target_flat_idx = tf.unique(self.target_flat)

    # get argmax
    argmax_output = tf.argmax(output, axis=-1, output_type=tf.int32)

    # compare how often they are not equal
    not_equal = tf.not_equal(argmax_output, target_flat_idx)

    return self.reduce_func(tf.cast(not_equal, tf.float32))


class GeometricNormalization(_ConcatInputLayer):
  """
  Use normalization based on l2 norm between target embedding and true embedding. Note: has not been proven to work.
  Also, there is a possible optimization to make the l2 calculation quicker, ask Ringo for details. Needs to use
  geometric_normalization_loss or geometric_ce_loss as the loss function.
  """

  layer_class = "geometric_normalization"

  def __init__(self, target_embedding_layer, **kwargs):
    """
    Use normalization based on l2 norm between target embedding and true embedding. Note: has not been proven to work.
    Also, there is a possible optimization to make the l2 calculation quicker, ask Ringo for details. Needs to use
  geometric_normalization_loss as the loss function.
    :param LayerBase target_embedding_layer: Layer of target embeddings.
    """

    super(GeometricNormalization, self).__init__(**kwargs)

    # TODO: add asserts
    decoder_output_dis = self.input_data.placeholder  # of shape [<?>,...,<?>, embedding_size]
    decoder_output_dis = decoder_output_dis  # Remove nans in input_data?

    self.word_embeddings = tf.get_default_graph().get_tensor_by_name(
                        target_embedding_layer.get_base_absolute_name_scope_prefix() +
                        "W:0")  # [vocab_size, embedding_size]

    # set shaping info correctly
    for d in range(len(decoder_output_dis.shape) - 1):  # -1 due to not wanting to add feature dim
      self.word_embeddings = tf.expand_dims(self.word_embeddings, axis=0)

    decoder_output_dis = tf.expand_dims(decoder_output_dis, axis=-2)  # [..., 1, embedding_size]

    distances = self.word_embeddings - decoder_output_dis  # [..., vocab_size, embedding_size]
    distances = tf.pow(distances, 2)  # [..., vocab_size, embedding_size]
    distances = tf.reduce_sum(distances, axis=-1)  # [..., vocab_size]
    max_distances = tf.reduce_max(distances, axis=-1, keepdims=True)  # [..., 1]
    distances = max_distances - distances  # [..., vocab_size]
    normalization_constant = 1 / tf.reduce_sum(distances, axis=-1, keepdims=True)  # [..., 1]
    output_geometric = tf.multiply(distances, normalization_constant)  # [..., vocab_size]

    if self.network.search_flag is False:
      from TFUtil import OutputWithActivation
      # Actually not doing activation here
      self.output_before_activation = OutputWithActivation(self.input_data.placeholder)
    self.output.placeholder = output_geometric

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    super(GeometricNormalization, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["target_embedding_layer"] = get_layer(d["target_embedding_layer"])


class GeometricNormalizationLoss(Loss):
  """
  Loss function for the geometric_normalization layer. Uses regression for training.
  This needs to be used for geometric_normalization. Again, can be further optimized.
  """
  class_name = "geometric_normalization_loss"

  def __init__(self, target_embedding_layer, min_regularizer=0.0, max_regularizer=0.0, debug=False, **kwargs):
    """

    :param target_embedding_layer:
    :param min_regularizer:
    :param max_regularizer:
    :param debug:
    :param kwargs:
    """
    super(GeometricNormalizationLoss, self).__init__(**kwargs)
    # Get embedding weights
    self.embedding_weights = None
    self.target_embedding_layer = target_embedding_layer
    self.min_regularizer = min_regularizer
    self.max_regularizer = max_regularizer
    self.debug = debug

  def get_value(self):
    assert self.target.sparse, "GeometricNormalizationLoss: Supporting only sparse targets"

    self.embedding_weights = tf.get_default_graph().get_tensor_by_name(
                      self.target_embedding_layer.get_base_absolute_name_scope_prefix() + "W:0")  # [vocab_size, embedding_size]

    output_embs = self.output_with_activation.y

    if self.target_flat is not None:
      targets = self.target.placeholder
      output_embs = tf.transpose(output_embs, perm=[1, 0, 2])  # TODO: make this generic
    else:
      targets = self.target.copy_as_time_major().placeholder

    target_embeddings = tf.nn.embedding_lookup(self.embedding_weights, ids=targets)  # [B, I, embedding_size]
    out = tf.squared_difference(output_embs, target_embeddings)

    if self.target_flat is not None:
      #out = self.reduce_func(tf.reduce_mean(out, axis=1))
      out = self.reduce_func(out)
    else:
      out = self.reduce_func(out)

    # TODO: maybe use stop_gradients instead?
    if self.min_regularizer > 0.0:
      assert self.min_regularizer <= self.max_regularizer, "Geo soft: Check min/max reg setting!"

      # TODO: make batch_axis dynamic
      reg = self._regularizer(target_embds=target_embeddings, batch_axis=0)
      reg_factor = tf.reduce_sum(out)/tf.reduce_sum(reg)
      reg_factor = tf.Print(reg_factor, [reg_factor], "Raw reg_factor")
      reg_factor = tf.clip_by_value(reg_factor, clip_value_min=self.min_regularizer,
                                    clip_value_max=self.max_regularizer)
      if self.debug:
        out = tf.Print(out, [out, reg, reg_factor], message="Out, Regularizer, reg_factor: ")
      out = out - reg_factor * reg

    if self.debug:
      out = tf.Print(out, [tf.reduce_sum(target_embeddings)], message="Target emb sum: ")

    return out

  def _regularizer(self, target_embds, batch_axis=0):
    # Get average vector over batch
    target_avg = tf.reduce_mean(target_embds, axis=batch_axis)  # [1, I, embedding_size]
    dist = target_embds - target_avg  # [B, I, embedding_size]
    dist = tf.pow(dist, 2)  # [B, I, embedding_size]
    return tf.reduce_sum(dist)


  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    super(GeometricNormalizationLoss, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["target_embedding_layer"] = get_layer(d["target_embedding_layer"])

  def get_error(self):
    # TODO: only get_error when actually needed
    return tf.constant(0.0)


class GeometricCrossEntropy(Loss):
  """
  Loss function for the geometric_normalization layer. Uses cross entropy for training.
  This needs to be used for geometric_normalization. Again, can
  be further optimized.
  """
  class_name = "geometric_ce_loss"

  def __init__(self, target_embedding_layer, full_vocab_size, vocab_sample_size=10, debug=False, **kwargs):
    super(GeometricCrossEntropy, self).__init__(**kwargs)
    # Get embedding weights
    self.embedding_weights = None
    self.target_embedding_layer = target_embedding_layer
    self.vocab_sample_size = vocab_sample_size
    self.debug = debug
    self.full_vocab_size = full_vocab_size

  def get_value(self):

    assert self.target.sparse, "GeometricNormalizationLoss: Supporting only sparse targets"

    # TODO: scopes
    # TODO: make less hacky
    self.embedding_weights = tf.get_default_graph().get_tensor_by_name(
                      self.target_embedding_layer.get_base_absolute_name_scope_prefix() + "W:0")  # [vocab_size, embedding_size]

    output_embs = self.output_with_activation.y

    if self.target_flat is not None:
      targets = self.target.placeholder
      output_embs = tf.transpose(output_embs, perm=[1, 0, 2])  # TODO: make this generic
    else:
      targets = self.target.copy_as_time_major().placeholder  # [B, I]

    output_embs = tf.expand_dims(output_embs, axis=-2)  # [B, I, 1, emb]

    # reshape targets to get to sampled vocab size
    targets = tf.expand_dims(targets, axis=-1)  # [B, I, 1]
    # sample shape
    sample_shape = tf.concat([tf.shape(targets)[:-1], [self.vocab_sample_size]], axis=0)

    # sample distribution to get random samples
    neg_sample = tf.random_uniform(shape=sample_shape, minval=0, maxval=self.full_vocab_size, dtype=tf.int32)
    # TODO: maybe force remove targets from sample

    # concat
    targets = tf.concat([targets, neg_sample], axis=-1)  # [B, I, vocab_sample_size + 1]

    target_embeddings = tf.nn.embedding_lookup(self.embedding_weights, ids=targets)  # [B, I, vocab_sample_size + 1, embedding_size]

    distances = tf.squared_difference(target_embeddings, output_embs)  # [..., vocab_size, embedding_size]
    distances = tf.reduce_sum(distances, axis=-1)  # [..., vocab_size]
    max_distances = tf.reduce_max(distances, axis=-1, keepdims=True)  # [..., 1]
    distances = max_distances - distances  # [..., vocab_size]

    # use sparse ce with logits
    new_targets = tf.zeros(shape=tf.shape(distances)[:-1], dtype=tf.int32)
    out = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=new_targets, logits=distances)

    if self.target_flat is not None:
      #out = self.reduce_func(tf.reduce_mean(out, axis=1))
      out = self.reduce_func(out)
    else:
      out = self.reduce_func(out)

    if self.debug:
      out = tf.Print(out, [out, tf.reduce_sum(target_embeddings)], message="out, Target emb sum: ")

    return out

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    super(GeometricCrossEntropy, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["target_embedding_layer"] = get_layer(d["target_embedding_layer"])

  def get_error(self):
    # TODO: only get_error when actually needed
    return tf.constant(0.0)

