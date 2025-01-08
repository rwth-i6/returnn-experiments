class FastBaumWelchFactoredOp(NativeOpGenBase):
  # noinspection PyUnresolvedReferences
  """
  inputs:
    :param am_scores: scores in -log space. 3d (time,batch,dim)
    :param edges: edges of the graph (from,to,center_idx,sequence_idx,left_idx,right_idx)
    :param weights: weights of the edges
  outputs:
    :param output: Baum-Welch alignment, scores in -log space. 3d (time,batch,dim), like am_scores
  """
  in_info = (
    {"name": "am_scores_left", "ndim": 3, "shape": (None, None, None),
     "need_contiguous": True, "gradient": "disconnected"},
    {"name": "am_scores_center",      "ndim": 3, "shape": (None, None, None),
     "need_contiguous": True, "gradient": "disconnected"},
    {"name": "am_scores_right",       "ndim": 3, "shape": (None, None, None),
     "need_contiguous": True, "gradient": "disconnected"},
    {"name": "edges",            "ndim": 2, "shape": (None,   None), "dtype": "int32",
     "need_contiguous": True, "gradient": "disconnected"},
    {"name": "weights",          "ndim": 1, "shape": (None,),
     "need_contiguous": True, "gradient": "disconnected"},
    {"name": "start_end_states", "ndim": 2, "shape": (2,      None), "dtype": "int32",
     "need_contiguous": True, "gradient": "disconnected"},
    {"name": "index",            "ndim": 2, "shape": ((0, 0), (0, 1)),
     "need_contiguous": True, "gradient": "disconnected"},
    {"name": "state_buffer",     "ndim": 2, "shape": (2,      None),
     "need_contiguous": True, "gradient": "disconnected"}
  )
  out_info = (
    {"name": "out_left", "ndim": 3, "shape": ((0, 0), (0, 1), (0, 2)), "need_contiguous": True},
    {"name": "out_center", "ndim": 3, "shape": ((1, 0), (1, 1), (1, 2)), "need_contiguous": True},
    {"name": "out_right", "ndim": 3, "shape": ((2, 0), (2, 1), (2, 2)), "need_contiguous": True},
    {"name": "sums",   "ndim": 2, "shape": ((0, 0), (0, 1)),         "need_contiguous": True},
  )

  c_extra_support_code = copy.copy(common_fast_bw_kernels)
  c_extra_support_code.update({
    "100_init_bwd_state_buffer": """
      DEF_KERNEL
      void init_bwd_state_buffer(
          float* states, unsigned* end_states, unsigned t, unsigned max_t, float* index, unsigned index_stride) {
        unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (index[t * index_stride + idx] == 1.0 && (t == max_t || index[(t + 1) * index_stride + idx] == 0.0)) {
          unsigned state_idx = end_states[idx];
          states[state_idx] = 0.0;
        }
      }
    """,
    "101_next_frame": """
      DEF_KERNEL
      void next_frame(bool fwd, unsigned num_edges, unsigned  num_left, unsigned  num_center, unsigned  num_right,
                      unsigned* sequence_idxs, unsigned* from_buffer, unsigned* to_buffer, float* weight_buffer,
                      unsigned* left_idxs, unsigned* center_idxs, unsigned* right_idxs,
                      float* prev_frame, float* next_frame,
                      float* am_scores_left, float* am_scores_center, float* am_scores_right, float* edge_buffer) {
        unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_edges) {
          return;
        }

        unsigned from     = from_buffer  [idx];
        float    prev_val = prev_frame[from];
        if (isinf(prev_val)) {
          edge_buffer[idx] = INF_F;
          return;
        }

        unsigned to           = to_buffer[idx];
        unsigned l_idx        = left_idxs[idx];
        unsigned c_idx        = center_idxs[idx];
        unsigned r_idx        = right_idxs[idx];
        float    edge_weight  = weight_buffer[idx];
        unsigned sequence_idx = sequence_idxs[idx];

        float val = prev_val + edge_weight
                    + am_scores_left[sequence_idx * num_left + l_idx]
                    + am_scores_center[sequence_idx * num_center + c_idx]
                    + am_scores_right[sequence_idx * num_right + r_idx];

        if (fwd) {
          edge_buffer[idx] += val;
        }
        else {
          edge_buffer[idx] += prev_val;
        }
        atomic_prob_add(next_frame + to, val);
      }
    """,
    "102_normalize": """
      DEF_KERNEL
      void normalize(float* buffer, unsigned* sequence_idxs, unsigned num_edges, unsigned num_seqs, float* sum_output) {
        DEF_SHARED(float, sum);

        buffer += blockIdx.x * num_edges;

        for (unsigned s = 0u; s < num_seqs; s++) {
          sum[s] = INF_F;
        }

        for (unsigned e = 0u; e < num_edges; e++) {
          unsigned s = sequence_idxs[e];
          sum[s] = prob_add(sum[s], buffer[e]);
        }

        for (unsigned s = 0ul; s < num_seqs; s++) {
          if (isinf(sum[s])) {
            // if the frame is empty (happens due to batching of seqs with unequal length), set it to 0
            sum_output[blockIdx.x * num_seqs + s] = 0.0;
          }
          else {
            sum_output[blockIdx.x * num_seqs + s] = sum[s];
          }
        }

        for (unsigned e = 0u; e < num_edges; e++) {
          unsigned s = sequence_idxs[e];
          buffer[e] -= sum[s];
        }
      }
    """,
    "103_compute_result": """
      DEF_KERNEL
      void compute_result(float* edge_buffer, float* out_left, float* out_center, float* out_right,
                          unsigned* left_idxs, unsigned* center_idxs, unsigned* right_idxs,
                          unsigned* sequence_idxs,
                          unsigned frame_stride_left, unsigned frame_stride_center, unsigned frame_stride_right,
                          unsigned seq_stride_left, unsigned seq_stride_center, unsigned seq_stride_right,
                          unsigned num_frames, unsigned num_seqs, unsigned num_edges) {
        unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_frames * num_edges) {
          return;
        }

        unsigned e_idx        = idx % num_edges;
        unsigned frame        = idx / num_edges;
        unsigned seq_idx      = sequence_idxs[e_idx];

        unsigned left_idx   = left_idxs[e_idx];
        unsigned center_idx = center_idxs[e_idx];
        unsigned right_idx  = right_idxs[e_idx];

        float    score        = edge_buffer[idx];

        atomic_prob_add(out_left + frame * frame_stride_left + seq_idx * seq_stride_left + left_idx, score);
        atomic_prob_add(out_center + frame * frame_stride_center + seq_idx * seq_stride_center + center_idx, score);
        atomic_prob_add(out_right + frame * frame_stride_left + seq_idx * seq_stride_right + right_idx, score);

      }
    """,

  })

  c_fw_code = """
    // am_scores, edges, weights, start_end_states, index, state_buffer* = input_names (*: inplace)
    // output = output_names
    assert(n_inputs  == 8);
    assert(n_outputs == 4);

    Ndarray* am_scores_left    = inputs[0];
    Ndarray* am_scores_center  = inputs[1];
    Ndarray* am_scores_right   = inputs[2];
    Ndarray* edges             = inputs[3];
    Ndarray* weights           = inputs[4];
    Ndarray* start_end_states  = inputs[5];
    Ndarray* index             = inputs[6];
    Ndarray* state_buffer      = inputs[7];

    Ndarray* out_left          = *outputs[0];
    Ndarray* out_center        = *outputs[1];
    Ndarray* out_right         = *outputs[2];

    Ndarray* sum_output        = *outputs[3];



    //debug_print(context, am_scores_left, "am_scores_left");
    //debug_print(context, am_scores_center, "am_scores_center");
    //debug_print(context, am_scores_right, "am_scores_right");
    //debug_print(context, edges, "edges");
    //debug_print(context, weights, "weights");
    //debug_print(context, start_end_states, "start_end_states");
    //debug_print(context, index, "index");
    //debug_print(context, state_buffer, "state_buffer");


    //assert_cmp(Ndarray_DIMS(am_scores)[0], ==, Ndarray_DIMS(out)[0]);
    //assert_cmp(Ndarray_DIMS(am_scores)[1], ==, Ndarray_DIMS(out)[1]);
    //assert_cmp(Ndarray_DIMS(am_scores)[2], ==, Ndarray_DIMS(out)[2]);
    //assert_cmp(Ndarray_DIMS(am_scores)[1], ==, Ndarray_DIMS(start_end_states)[1]);

    //assert_cmp(Ndarray_DIMS(sum_output)[0], ==, Ndarray_DIMS(am_scores)[0]);
    //assert_cmp(Ndarray_DIMS(sum_output)[1], ==, Ndarray_DIMS(am_scores)[1]);

    bool            dump_alignment = false;
    bool            dump_output    = false;
    unsigned        dump_every     = 40u;
    static unsigned batch_idx      = 0u;
    float           pruning        = 10.f;

    unsigned* d_from = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges)
      + 0 * Ndarray_STRIDE(edges, 0));
    unsigned* d_to = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges)
      + 1 * Ndarray_STRIDE(edges, 0));
    unsigned* d_emission_idxs = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges)
      + 2 * Ndarray_STRIDE(edges, 0));
    unsigned* d_sequence_idxs = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges)
      + 3 * Ndarray_STRIDE(edges, 0));
    unsigned* d_left_idxs = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges)
      + 4 * Ndarray_STRIDE(edges, 0));
    unsigned* d_center_idxs = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges)
      + 5 * Ndarray_STRIDE(edges, 0));
    unsigned* d_right_idxs = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(edges)
      + 6 * Ndarray_STRIDE(edges, 0));

    float*    d_weights = Ndarray_DEV_DATA(weights);

    float*    d_am_scores_left = Ndarray_DEV_DATA(am_scores_left);
    float*    d_am_scores_center = Ndarray_DEV_DATA(am_scores_center);
    float*    d_am_scores_right = Ndarray_DEV_DATA(am_scores_right);

    unsigned* d_start_states = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(start_end_states)
      + 0 * Ndarray_STRIDE(start_end_states, 0));
    unsigned* d_end_states = reinterpret_cast<unsigned*>(Ndarray_DEV_DATA_int32(start_end_states)
      + 1 * Ndarray_STRIDE(start_end_states, 0));
    float*    d_index             = Ndarray_DEV_DATA(index);
    float*    d_state_buffer_prev = Ndarray_DEV_DATA(state_buffer) + 0 * Ndarray_STRIDE(state_buffer, 0);
    float*    d_state_buffer_next = Ndarray_DEV_DATA(state_buffer) + 1 * Ndarray_STRIDE(state_buffer, 0);

    float*    d_out_left          = Ndarray_DEV_DATA(out_left);
    float*    d_out_center        = Ndarray_DEV_DATA(out_center);
    float*    d_out_right         = Ndarray_DEV_DATA(out_right);

    float*    d_sum_output        = Ndarray_DEV_DATA(sum_output);

    unsigned n_frames    = Ndarray_DIMS(am_scores_center)[0];
    unsigned n_seqs      = Ndarray_DIMS(am_scores_center)[1];

    unsigned n_lefts = Ndarray_DIMS(am_scores_left)[2];
    unsigned n_centers = Ndarray_DIMS(am_scores_center)[2];
    unsigned n_rights = Ndarray_DIMS(am_scores_right)[2];

    unsigned n_states    = Ndarray_DIMS(state_buffer)[1];
    unsigned n_edges     = Ndarray_DIMS(edges)[1];
    unsigned n_threads   = 1024u;
    unsigned n_blocks    = (n_edges + n_threads - 1) / n_threads;

    unsigned frame_stride_left     = Ndarray_STRIDE(am_scores_left, 0);
    unsigned frame_stride_center   = Ndarray_STRIDE(am_scores_center, 0);
    unsigned frame_stride_right    = Ndarray_STRIDE(am_scores_right, 0);

    unsigned sequence_stride_left  = Ndarray_STRIDE(am_scores_left, 1);
    unsigned sequence_stride_center= Ndarray_STRIDE(am_scores_center, 1);
    unsigned sequence_stride_right = Ndarray_STRIDE(am_scores_right, 1);

    unsigned index_stride    = Ndarray_STRIDE(index, 0);

    assert(n_frames > 0);

    //std::cerr << "n_centers: "    << n_lefts    << std::endl;
    //std::cerr << "n_centers: "    << n_centers    << std::endl;
    //std::cerr << "n_centers: "    << n_rights    << std::endl;
    //std::cerr << "n_seqs: "      << n_seqs      << std::endl;
    //std::cerr << "n_emissions: " << n_emissions << std::endl;
    //std::cerr << "n_states: "    << n_states    << std::endl;
    //std::cerr << "n_edges: "     << n_edges     << std::endl;
    //std::cerr << "n_threads: "   << n_threads   << std::endl;
    //std::cerr << "n_blocks: "    << n_blocks    << std::endl;

    //std::cerr << "frame_stride: "     << frame_stride    << std::endl;
    //std::cerr << "sequnence_stride: " << sequence_stride << std::endl;
    //std::cerr << "index_stride: "     << index_stride    << std::endl;

    // initialize edge buffer
    float* d_edge_buffer = reinterpret_cast<float*>(device_malloc(n_edges * n_frames * sizeof(float)));
    if(!d_edge_buffer) { HANDLE_LAST_ERROR(); abort(); }  // error should have been set in device_malloc
    unsigned n_fill_blocks = (n_edges * n_frames + n_threads - 1u) / n_threads;
    start_dev_kernel2(fill_array, n_fill_blocks, n_threads, 0, (d_edge_buffer, 0.0, n_edges * n_frames));
    HANDLE_LAST_ERROR();

    // initialize the state buffer
    n_fill_blocks = (n_states + n_threads - 1u) / n_threads;
    start_dev_kernel2(
      fill_array, n_fill_blocks, n_threads, 0,
      (d_state_buffer_prev, std::numeric_limits<float>::infinity(), n_states));
    HANDLE_LAST_ERROR();
    start_dev_kernel2(set_start_states, 1, n_seqs, 0, (d_state_buffer_prev, d_start_states));
    HANDLE_LAST_ERROR();

    // initialize full state buffer (only used to dump the alignment)
    float* d_state_buffer_all = NULL;
    if (dump_alignment && batch_idx %% dump_every == 0) {
      d_state_buffer_all = reinterpret_cast<float*>(device_malloc(n_states * (n_frames + 1u) * sizeof(float)));
      if(!d_state_buffer_all) { HANDLE_LAST_ERROR(); abort(); }  // error should have been set in device_malloc
      Ndarray_memcpy(d_state_buffer_all, d_state_buffer_prev, n_states * sizeof(float));
      HANDLE_LAST_ERROR();
    }

    // fwd pass
    for (unsigned t = 0u; t < n_frames; t++) {
      start_dev_kernel2(
        fill_array, n_fill_blocks, n_threads, 0,
        (d_state_buffer_next, std::numeric_limits<float>::infinity(), n_states));
      HANDLE_LAST_ERROR();
      start_dev_kernel2(next_frame, n_blocks, n_threads, 0,
        (true, n_edges, sequence_stride_left, sequence_stride_center, sequence_stride_right,
         d_sequence_idxs, d_from, d_to, d_weights,
         d_left_idxs, d_center_idxs, d_right_idxs,
         d_state_buffer_prev, d_state_buffer_next,
         d_am_scores_left + t * frame_stride_left, d_am_scores_center + t * frame_stride_center, d_am_scores_right + t * frame_stride_right,
         d_edge_buffer + t * n_edges));
      HANDLE_LAST_ERROR();
      std::swap(d_state_buffer_prev, d_state_buffer_next);
    }

    // bwd pass
    start_dev_kernel2(
      fill_array, n_fill_blocks, n_threads, 0,
      (d_state_buffer_prev, std::numeric_limits<float>::infinity(), n_states));
    HANDLE_LAST_ERROR();
    for (unsigned t = n_frames; t > 0; t--) {
      start_dev_kernel2(init_bwd_state_buffer, 1, n_seqs, 0,
        (d_state_buffer_prev, d_end_states, t - 1, n_frames - 1, d_index, index_stride));
      HANDLE_LAST_ERROR();
      if (dump_alignment && batch_idx %% dump_every == 0) {
        float alpha = 1.0f;
        //HANDLE_ERROR(cublasSaxpy(
        //  handle, n_states, &alpha, d_state_buffer_prev, 1, d_state_buffer_all + t * n_states, 1));
      }
      start_dev_kernel2(
        fill_array, n_fill_blocks, n_threads, 0,
        (d_state_buffer_next, std::numeric_limits<float>::infinity(), n_states));
      HANDLE_LAST_ERROR();
      start_dev_kernel2(next_frame, n_blocks, n_threads, 0,
        (false, n_edges, sequence_stride_left, sequence_stride_center, sequence_stride_right,
         d_sequence_idxs, d_to, d_from, d_weights,
         d_left_idxs, d_center_idxs, d_right_idxs,
         d_state_buffer_prev, d_state_buffer_next,
         d_am_scores_left + (t - 1) * frame_stride_left, d_am_scores_center + (t - 1) * frame_stride_center, d_am_scores_right + (t - 1) * frame_stride_right,
         d_edge_buffer + (t - 1) * n_edges));
      HANDLE_LAST_ERROR();
      std::swap(d_state_buffer_prev, d_state_buffer_next);
    }

    // normalize at each time frame
    start_dev_kernel2(normalize, n_frames, 1, n_seqs * sizeof(float),
      (d_edge_buffer, d_sequence_idxs, n_edges, n_seqs, d_sum_output));
    HANDLE_LAST_ERROR();


    unsigned n_fill_blocks_left = (n_frames * n_seqs * n_lefts + n_threads - 1u) / n_threads;
    unsigned n_fill_blocks_center = (n_frames * n_seqs * n_centers + n_threads - 1u) / n_threads;
    unsigned n_fill_blocks_right = (n_frames * n_seqs * n_rights + n_threads - 1u) / n_threads;

    start_dev_kernel2(
      fill_array, n_fill_blocks_left, n_threads, 0,
      (d_out_left, std::numeric_limits<float>::infinity(), n_frames * n_seqs * n_lefts));
    HANDLE_LAST_ERROR();

    start_dev_kernel2(
      fill_array, n_fill_blocks_center, n_threads, 0,
      (d_out_center, std::numeric_limits<float>::infinity(), n_frames * n_seqs * n_centers));
    HANDLE_LAST_ERROR();

    start_dev_kernel2(
      fill_array, n_fill_blocks_right, n_threads, 0,
      (d_out_right, std::numeric_limits<float>::infinity(), n_frames * n_seqs * n_rights));
    HANDLE_LAST_ERROR();

    frame_stride_left      = Ndarray_STRIDE(out_left, 0);
    sequence_stride_left   = Ndarray_STRIDE(out_left, 1);
    frame_stride_center    = Ndarray_STRIDE(out_center, 0);
    sequence_stride_center = Ndarray_STRIDE(out_center, 1);
    frame_stride_right     = Ndarray_STRIDE(out_right, 0);
    sequence_stride_right  = Ndarray_STRIDE(out_right, 1);
    n_blocks               = (n_frames * n_edges + n_threads - 1u) / n_threads;

    start_dev_kernel2(compute_result, n_blocks, n_threads, 0,
      (d_edge_buffer, d_out_left, d_out_center, d_out_right,
       d_left_idxs, d_center_idxs, d_right_idxs, d_sequence_idxs,
       frame_stride_left, frame_stride_center, frame_stride_right,
       sequence_stride_left, sequence_stride_center, sequence_stride_right, n_frames, n_seqs, n_edges));
    HANDLE_LAST_ERROR();

    #if TENSORFLOW
    // Certain TensorFlow code doesn't like inf, even if it is just the CheckNumerics,
    // which is helpful for debugging.
    // We replace it by a very high number, so that tf.exp(-out) will still result in 0.0.

    unsigned n_blocks_left = (n_frames * n_seqs * n_lefts + n_threads - 1u) / n_threads;
    unsigned n_blocks_center = (n_frames * n_seqs * n_centers + n_threads - 1u) / n_threads;
    unsigned n_blocks_right = (n_frames * n_seqs * n_rights + n_threads - 1u) / n_threads;

    start_dev_kernel2(remove_inf, n_blocks_left, n_threads, 0, (d_out_left, n_frames * n_seqs * n_lefts));
    start_dev_kernel2(remove_inf, n_blocks_center, n_threads, 0, (d_out_center, n_frames * n_seqs * n_centers));
    start_dev_kernel2(remove_inf, n_blocks_right, n_threads, 0, (d_out_right, n_frames * n_seqs * n_rights));
    #endif

    device_free(d_edge_buffer);
    if (d_state_buffer_all != NULL) {
      device_free(d_state_buffer_all);
    }
    batch_idx++;
  """

  c_bw_code = None
