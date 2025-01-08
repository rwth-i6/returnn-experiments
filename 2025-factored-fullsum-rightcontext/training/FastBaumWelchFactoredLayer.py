class FastBaumWelchFactoredLayer(_ConcatInputLayer):
    """
    Calls :func:`fast_baum_welch` or :func:`fast_baum_welch_by_sprint_automata`.
    We expect that our input are +log scores, e.g. use log-softmax.
    """

    layer_class = "fast_bw_factored"
    recurrent = True

    def __init__(
        self,
        align_target,
        align_target_key=None,
        ctc_opts=None,
        hmm_opts=None,
        sprint_opts=None,
        input_type="log_prob",
        tdp_scale=1.0,
        am_scale=1.0,
        min_prob=0.0,
        **kwargs,
    ):
        """
        :param str align_target: e.g. "monophone-hmm", "ctc"
        :param str|None align_target_key: e.g. "classes", used for e.g. align_target "ctc"
        :param dict[str] ctc_opts: used for align_target "ctc"
        :param dict[str] ctc_opts: used for align_target "X-hmm"
        :param dict[str] sprint_opts: used for Sprint (RASR) for align_target "sprint"
        :param str input_type: "log_prob" or "prob"
        :param float tdp_scale:
        :param float am_scale:
        :param float min_prob: clips the minimum prob (value in [0,1])
        :param LayerBase|None staircase_seq_len_source:
        """

        def get_labels_from_hmm_dense(labels, num_contexts):
            factored_labels = {}

            factored_labels["r"] = labels % num_contexts
            result = labels // num_contexts

            factored_labels["l"] = result % num_contexts
            factored_labels["c"] = result // num_contexts

            return factored_labels

        super(FastBaumWelchFactoredLayer, self).__init__(**kwargs)

        # only for seq ids
        from returnn.tf.util.basic import sequence_mask_time_major

        data = self.sources[1].output.copy_as_time_major()
        seq_tags = self.network.get_seq_tags()
        seq_mask = sequence_mask_time_major(data.get_sequence_lengths())

        # We want the scores in -log space.
        am_scores = []
        for input in self.sources:
            am_s = self.get_am_scores_from_input(input, input_type)
            if min_prob > 0:
                am_s = tf.minimum(am_s, -numpy.log(min_prob))
                # in -log space
            if am_scale != 1.0:
                am_s *= am_scale
            am_scores.append(am_s)


        from returnn.tf.native_op import fast_baum_welch_factored

        if align_target == "hmm-monophone":
            from returnn.tf.sprint import get_sprint_automata_for_batch_op
            assert hmm_opts is not None and "num_contexts" in hmm_opts, "Pls. provide number of context labels"
            edges, weights, start_end_states = get_sprint_automata_for_batch_op(sprint_opts=sprint_opts, tags=seq_tags)
            #edges[2] has the dense labels
            labels = get_labels_from_hmm_dense(labels=edges[2], num_contexts=hmm_opts["num_contexts"])
            extended_edges = tf.cast(
                tf.stack([edges[0], edges[1], edges[2], edges[3], labels["l"], labels["c"], labels["r"]]), dtype="int32"
            )

        elif align_target == "ctc":
            from returnn.tf.sprint import get_sprint_factored_ctc_automata_for_batch_op
            assert ctc_opts is not None and "num_contexts" in ctc_opts, "Pls. provide the number of context"
            assert ctc_opts is not None and "max_non_end_of_word_index" in ctc_opts, "Pls. provide the last index for non end-of-word"
            # edges[2] has the ctc-like via rasr definition
            edges, weights, start_end_states, left, right = get_sprint_factored_ctc_automata_for_batch_op(
                sprint_opts=sprint_opts, ctc_opts=ctc_opts, tags=seq_tags)
            extended_edges = tf.cast(
                tf.stack([edges[0], edges[1], edges[2], edges[3], left, edges[2], right]), dtype="int32"
            )
        else:
            raise Exception("%s: invalid align_target %r" % (self, align_target))

        if tdp_scale != 1.0:
            if tdp_scale == 0.0:
                weights = tf.zeros_like(weights)
            else:
                weights *= tdp_scale

        fwdbwd_left, fwdbwd_center, fwdbwd_right, obs_scores = fast_baum_welch_factored(
            am_scores_left=am_scores[0],
            am_scores_center=am_scores[1],
            am_scores_right=am_scores[2],
            float_idx=seq_mask,
            edges=extended_edges,
            weights=weights,
            start_end_states=start_end_states,
        )

        loss = obs_scores[0]
        self.output_loss = loss

        bw_left = tf.exp(-fwdbwd_left)
        bw_center = tf.exp(-fwdbwd_center)
        bw_right = tf.exp(-fwdbwd_right)

        shape_left = bw_left.get_shape().as_list()
        shape_center = bw_center.get_shape().as_list()
        shape_right = bw_right.get_shape().as_list()

        # possible solution with sub_layers
        bw_left_data = Data(shape=shape_left[-2:], placeholder=bw_left, name="left", batch_dim_axis=1, dtype="float32")
        bw_center_data = Data(
            shape=shape_center[-2:], placeholder=bw_center, name="center", batch_dim_axis=1, dtype="float32"
        )
        bw_right_data = Data(
            shape=shape_right[-2:], placeholder=bw_right, name="right", batch_dim_axis=1, dtype="float32"
        )

        gamma_left_layer = InternalLayer(name=f"left", network=self.network, output=bw_left_data)
        gamma_left_layer.output_loss = loss
        gamma_left_layer.output.size_placeholder = self.sources[0].output.copy_as_time_major().size_placeholder.copy()

        gamma_center_layer = InternalLayer(name="center", network=self.network, output=bw_center_data)
        gamma_center_layer.output_loss = loss
        gamma_center_layer.output.size_placeholder = self.sources[1].output.copy_as_time_major().size_placeholder.copy()

        gamma_right_layer = InternalLayer(name="right", network=self.network, output=bw_right_data)
        gamma_right_layer.output_loss = loss
        gamma_right_layer.output.size_placeholder = self.sources[2].output.copy_as_time_major().size_placeholder.copy()

        self._sub_layers = {"left": gamma_left_layer, "center": gamma_center_layer, "right": gamma_right_layer}

        self.output.placeholder = tf.concat([bw_left, bw_center, bw_right], axis=2)
        self.output.size_placeholder = data.size_placeholder.copy()

    @classmethod
    def transform_config_dict(cls, d, network, get_layer):
        """
        :param dict[str] d:
        :param TFNetwork.TFNetwork network:
        :param get_layer:
        """
        super(FastBaumWelchFactoredLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
        """
        #### from old implementation, not sure if it is still needed
        if d.get("phone_idcs_layer"):
            d["phone_idcs_layer"] = get_layer(d["phone_idcs_layer"])
        """

    def get_sub_layer(self, layer_name):
        """
        :param str layer_name:
        :rtype: LayerBase|None
        """
        return self._sub_layers.get(layer_name, None)

    def get_am_scores_from_input(cls, input, input_type="log_prob"):
        data = input.output.copy_as_time_major()
        if input_type == "log_prob":
            am_scores = -data.placeholder
        elif input_type == "prob":
            if input.output_before_activation:
                am_scores = -input.output_before_activation.get_log_output()
                if input.output.is_batch_major:
                    from returnn.tf.util.basic import swapaxes

                    am_scores = swapaxes(am_scores, 0, 1)
            else:
                from returnn.tf.util.basic import safe_log

                am_scores = -safe_log(data.placeholder)
        else:
            raise Exception("%s: invalid input_type %r" % (self, input_type))
        return am_scores

