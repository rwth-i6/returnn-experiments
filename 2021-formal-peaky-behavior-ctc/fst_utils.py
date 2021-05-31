#!/usr/bin/env python3

"""
Some symbolic computations on FSAs, e.g. counting all paths,
or all paths with some label in a specific frame, or so.
This script is used for the symbolic computations in the paper.

For symbolic mathematics, these are probably the relevant possible Python options:
- https://www.sympy.org/ (the one I use here)
- https://www.sagemath.org/ (too big maybe)
- https://github.com/diofant/diofant (too less mature maybe)

Relevant functions:

- count_all_paths, for C(T)
- count_all_paths_with_label_in_frame, for C(s,t,T)
- count_paths_with_label, for C_T
- count_all_paths_with_label_avg, for C(s,T)
- count_all_paths_with_label_seq, for FFNN and generative model

"""

import sympy_utils
import os
import sys
from collections import defaultdict
import sympy
import numpy
import typing
from pprint import pprint


class Arc:
  def __init__(self, source_state: int, target_state: int, label: str):
    self.source_state = source_state
    self.target_state = target_state
    self.label = label

  def short_str(self, target_is_final_mark: bool = False):
    return "%i -%s-> %i%s" % (
      self.source_state, self.label, self.target_state, "." if target_is_final_mark else "")

  def __repr__(self):
    return "Arc{%s}" % self.short_str()


class Fsa:
  """
  Finite state automaton.
  """

  def __init__(self):
    self.states = {0}  # type: typing.Set[int]
    self.start_state = 0
    self.final_states = set()  # type: typing.Set[int]
    self.arcs = set()  # type: typing.Set[Arc]
    self.arcs_by_source_state = {}  # type: typing.Dict[int,typing.Set[Arc]]  # src-state -> Arc

  def add_arc(self, source_state: int, target_state: int, label: str):
    self.states.add(source_state)
    self.states.add(target_state)
    arc = Arc(source_state=source_state, target_state=target_state, label=label)
    self.arcs.add(arc)
    self.arcs_by_source_state.setdefault(source_state, set()).add(arc)

  def add_final_state(self, state: int):
    self.final_states.add(state)

  def get_labels(self):
    labels = set()
    for arc in self.arcs:
      labels.add(arc.label)
    return sorted(labels)

  def is_deterministic_by_label(self):
    for _, arcs in self.arcs_by_source_state.items():
      labels = set()
      for arc in arcs:
        if arc.label in labels:
          return False
        labels.add(arc.label)
    return True

  def str(self):
    collected_arcs_set = set()
    collected_arcs_list = []
    visited_states = set()
    state_queue = [self.start_state]
    while state_queue:
      state = state_queue.pop(0)
      if state in visited_states:
        continue
      visited_states.add(state)
      for arc in self.arcs_by_source_state[state]:
        if arc in collected_arcs_set:
          continue
        collected_arcs_set.add(arc)
        collected_arcs_list.append(arc.short_str(target_is_final_mark=arc.target_state in self.final_states))
        if arc.target_state not in visited_states:
          state_queue.append(arc.target_state)
    return ", ".join(collected_arcs_list)

  def __repr__(self):
    return "Fsa{%s}" % self.str()

  def copy_ref_new_final_states(self, final_states: typing.Set[int]):
    fsa = Fsa()
    fsa.states = self.states
    fsa.start_state = self.start_state
    fsa.arcs = self.arcs
    fsa.arcs_by_source_state = self.arcs_by_source_state
    fsa.final_states = final_states
    return fsa

  def get_deterministic_source_to_target_arc(self, source_state, target_state):
    """
    :param int source_state:
    :param int target_state:
    :rtype: Arc|None
    """
    assert source_state in self.states and target_state in self.states
    possible_arcs = []
    for arc in self.arcs_by_source_state[source_state]:
      if arc.target_state == target_state:
        possible_arcs.append(arc)
    assert len(possible_arcs) <= 1
    if not possible_arcs:
      return None
    return possible_arcs[0]

  def get_deterministic_source_to_target_prob(self, source_state, target_state, probs_by_label):
    """
    :param int source_state:
    :param int target_state:
    :param dict[str,T] probs_by_label:
    :rtype: T|int
    """
    arc = self.get_deterministic_source_to_target_arc(source_state=source_state, target_state=target_state)
    if arc:
      return probs_by_label[arc.label]
    return 0

  def get_edges_weights_start_end_states(self):
    """
    :return: edges, weights, start_end_states;
      edges is (4,num_edges), int32, edges of the graph (from,to,emission_idx,sequence_idx).
      weights is (num_edges,), float32. all zero.
      start_end_states is (2,batch), int32, (start,end) state idx in FSA.
    :rtype: (numpy.ndarray,numpy.ndarray,numpy.ndarray)
    """
    assert len(self.final_states) == 1  # API does not allow otherwise...
    # TODO ...

  def tf_get_full_sum(self, logits, idx_by_label=None, unroll=False, max_approx=False):
    """
    :param tf.Tensor logits: [T,B,D], normalized as you want
    :param dict[str,int]|None idx_by_label:
    :param bool unroll:
    :param bool max_approx: max instead of sum (Viterbi instead of Baum-Welch)
    :return: [B], scores in +log space
    :rtype: tf.Tensor
    """
    import tensorflow as tf
    tf = tf.compat.v1
    assert isinstance(logits, tf.Tensor)
    num_states = max(self.states) + 1
    states = range(num_states)
    labels = self.get_labels()
    num_labels = len(labels)
    logits.set_shape((None, None, num_labels))
    if idx_by_label is None:
      idx_by_label = {}
      idx = 0
      for label in labels:
        if label != BlankLabel:
          idx_by_label[label] = idx
          idx += 1
      if BlankLabel in labels:
        idx_by_label[BlankLabel] = idx
    assert all([0 <= idx_by_label[label] < num_labels for label in labels])
    num_frames, num_batch, _ = logits.shape.as_list()
    if num_frames is None:
      num_frames = tf.shape(logits)[0]
    if num_batch is None:
      num_batch = tf.shape(logits)[1]
    logits_ta = tf.TensorArray(tf.float32, size=num_frames, element_shape=(num_labels, None))
    logits_ta = logits_ta.unstack(tf.transpose(logits, [0, 2, 1]))  # (dim,B) each element
    initial_scores = [float("-inf")] * num_states
    initial_scores[self.start_state] = 0.
    scores = tf.tile(tf.expand_dims(tf.constant(initial_scores), axis=1), [1, num_batch])  # (states,B)

    def combine_scores(ss_, axis):
      if max_approx:
        # Do not use reduce_max directly, we want that the gradient follows really only a single path.
        arg_ = tf.argmax(ss_, axis=axis)  # [B]
        # Ugly usage of batch_gather...
        if axis == 0:
          ss_ = tf.transpose(ss_, [1, 0])  # [B,dim]
        assert arg_.shape.as_list() == [None]
        ss_ = tf.squeeze(tf.batch_gather(ss_, indices=tf.expand_dims(arg_, axis=-1)), axis=-1)  # [B]
        assert ss_.shape.as_list() == [None]
        return ss_
        # return tf.reduce_max(ss_, axis=axis)
      return tf.reduce_logsumexp(ss_, axis=axis)

    def add_scores(s_, y_):
      # Need to use this, to avoid inf/nan gradients.
      return tf.where(tf.is_finite(s_), s_ + y_, s_)

    def body(t_, scores_):
      """
      :param tf.Tensor t_: scalar
      :param tf.Tensor scores_: (states,B)
      """
      logits_cur = logits_ta.read(t_)  # (dim,B)
      logits_cur_by_label_idx = tf.unstack(logits_cur, axis=0)
      assert len(logits_cur_by_label_idx) == num_labels
      scores_by_state_ = tf.unstack(scores_, axis=0)
      assert len(scores_by_state_) == num_states

      next_scores_parts = [[] for _ in range(num_states)]
      for src_state in states:
        arcs = sorted(  # make it deterministic by sorting
          self.arcs_by_source_state[src_state],
          key=lambda arc_: (arc_.target_state, idx_by_label[arc_.label]))
        for arc in arcs:
          next_scores_parts[arc.target_state].append(
            add_scores(scores_by_state_[src_state], logits_cur_by_label_idx[idx_by_label[arc.label]]))

      next_scores = [float("-inf")] * num_states
      for i, parts in enumerate(next_scores_parts):
        if parts:
          if len(parts) == 1:
            next_scores[i] = parts[0]
          else:
            next_scores[i] = combine_scores(parts, axis=0)
      return t_ + 1, tf.convert_to_tensor(next_scores)

    if unroll:
      assert isinstance(num_frames, int)
      for t in range(num_frames):
        with tf.name_scope("frame_t%i" % t):
          _, scores = body(t, scores)
    else:
      _, scores = tf.while_loop(
        cond=lambda t_, scores_: tf.less(t_, num_frames),
        body=body,
        loop_vars=(0, scores),
        shape_invariants=(tf.TensorShape(()), tf.TensorShape((num_states, None))))
    scores_by_state = tf.unstack(scores, axis=0)
    assert len(scores_by_state) == num_states
    final_scores = combine_scores([scores_by_state[i] for i in self.final_states], axis=0)
    return final_scores

  def tf_get_best_alignment(self, logits):
    """
    :param tf.Tensor logits: [T,B,D], before softmax
    :return: (alignment, scores), alignment is (time, batch), scores is (batch,), in +log space
    :rtype: (tf.Tensor, tf.Tensor)
    """
    import tensorflow as tf
    scores = self.tf_get_full_sum(logits=logits, max_approx=True)
    # Somewhat hacky to get alignment that way, but should work...
    logits_grad, = tf.gradients(tf.reduce_sum(scores), logits)
    # logits_grad is non-zero only for the right label. Shape [T,B,D].
    # logits_grad = tf.Print(logits_grad, ["logits_grad", logits_grad], summarize=100)
    alignment = tf.argmax(logits_grad, axis=-1)  # [T,B]
    return alignment, scores


def iterate_all_paths(fsa: Fsa, num_frames: int, state: typing.Union[None, int] = None) -> (
      typing.Generator[typing.List[Arc], None, None]):
  if state is None:
    state = fsa.start_state
  if num_frames == 0:
    if state in fsa.final_states:
      yield []
    return
  assert num_frames > 0
  for arc in fsa.arcs_by_source_state[state]:
    for sub_path in iterate_all_paths(fsa=fsa, state=arc.target_state, num_frames=num_frames - 1):
      yield [arc] + sub_path


def count_all_paths_inefficient(fsa: Fsa, num_frames: int) -> int:
  return len(list(iterate_all_paths(fsa=fsa, num_frames=num_frames)))


def count_all_paths_with_label_in_frame_inefficient(fsa: Fsa, num_frames: int, frame_idx: int, label: str) -> int:
  return len([path for path in iterate_all_paths(fsa=fsa, num_frames=num_frames) if path[frame_idx].label == label])


@sympy.cacheit
def count_all_paths(fsa: Fsa, state: typing.Union[None, int] = None) -> (sympy.Symbol, sympy.Expr):
  """
  :return: (num_frames, count).
    num_frames is a symbolic var,
    count is the count of all unique paths from the given state (or start state) to any final state.
  """
  if state is None:
    return count_all_paths(fsa=fsa, state=fsa.start_state)
  # Currently assume only loops in current state, or forward.
  num_frames = sympy.Symbol("num_frames_%i" % state, integer=True, nonnegative=True)
  if state in fsa.final_states:
    count = sympy.Piecewise((1, sympy.Equality(num_frames, 0)), (0, True))
  else:
    count = sympy.sympify(0)

  have_loop = False
  for arc in fsa.arcs_by_source_state[state]:
    if arc.target_state == state:
      have_loop = True
    else:
      num_frames_, count_ = count_all_paths(fsa=fsa, state=arc.target_state)
      count += sympy.Piecewise(
        (count_.subs(num_frames_, num_frames - 1), sympy.Ge(num_frames, 1)),
        (0, True))

  count = count.simplify()

  assert isinstance(count, sympy.Expr)
  while have_loop:  # this loop just to simplify the logic. only one iteration
    # If there is a loop, it means we have to sum over all possible num_frames.
    # SymPy does not automatically simplify the expression currently, mostly because of Piecewise inside Sum,
    # so we try to catch all relevant cases ourselves and do the simplification manually.
    # We still have the code for the general case. See below.
    if count == sympy.sympify(0):
      break  # sum over 0 is 0. nothing to do.
    true_value_wild = sympy.Wild("true_value")
    false_value_wild = sympy.Wild("false_value")
    cond_wild = sympy.Wild("cond")
    m = count.match(sympy.Piecewise((true_value_wild, cond_wild), (false_value_wild, True)))
    if m:
      # Many of the assertions below are just because it is not implemented otherwise.
      true_value = m[true_value_wild]
      false_value = m[false_value_wild]
      n_wild = sympy.Wild("n")
      cond = m[cond_wild]
      assert isinstance(cond, (sympy.Expr, sympy.Basic))
      if cond.match(sympy.Eq(num_frames, 0) | sympy.Ge(num_frames, 1)) == {}:
        cond = sympy.Ge(num_frames, 0)
      cond_match = cond.match(sympy.Eq(num_frames, n_wild))
      if cond_match:
        assert isinstance(true_value, sympy.Integer) and isinstance(false_value, sympy.Integer)
        assert cond_match[n_wild] == sympy.sympify(0)
        count = true_value + false_value * num_frames
        break
      if cond.match(True) == {}:
        assert isinstance(true_value, sympy.Integer)
        count = true_value * (num_frames + 1)
        break
      cond_match = cond.match(sympy.Ge(num_frames, n_wild))
      if cond_match:
        n = cond_match[n_wild]
        assert isinstance(n, sympy.Integer)
        num_frames_loop = sympy.Symbol("num_frames_%i_looped" % state, integer=True, nonnegative=True)
        count = sympy.Sum(true_value, (num_frames, n, num_frames_loop))
        if n > 0:
          count += sympy.Sum(false_value, (num_frames, 0, n - 1))
        num_frames = num_frames_loop
        count = count.simplify().simplify()
        break
      # This exception can be removed once SymPy can automatically simplify the expression.
      # The whole code actually can be removed then...
      raise Exception("no simplify rule for piecewise %r" % m)

    # general case...
    num_frames_loop = sympy.Symbol("num_frames_%i_looped" % state, integer=True, nonnegative=True)
    count = sympy.Sum(count, (num_frames, 0, num_frames_loop))
    num_frames = num_frames_loop
    break

  return num_frames, count


@sympy.cacheit
def count_all_paths_with_label_in_frame(fsa: Fsa, label: str) -> (
      sympy.Symbol, sympy.Symbol, sympy.Expr):
  """
  :return: (num_frames, frame_idx, count)
  """
  num_frames = sympy.Symbol("num_frames", integer=True, nonnegative=True)
  frame_idx = sympy.Symbol("frame_idx", integer=True, nonnegative=True)  # 0-indexed
  count = 0
  for arc in fsa.arcs:
    if arc.label != label:
      continue

    before_num, before_count = count_all_paths(fsa=fsa.copy_ref_new_final_states({arc.source_state}))
    before_count = before_count.subs(before_num, frame_idx)
    after_num, after_count = count_all_paths(fsa=fsa, state=arc.target_state)
    after_count = after_count.subs(after_num, num_frames - frame_idx - 1)

    count += before_count * after_count

  count = sympy.sympify(count).simplify().simplify()
  return num_frames, frame_idx, count


def count_all_paths_with_label_avg(fsa: Fsa, label: str, num_frames: typing.Optional[int] = None):
  num_frames_, frame_idx, count = count_all_paths_with_label_in_frame(fsa=fsa, label=label)
  count_sum = sympy.Sum(count, (frame_idx, 0, num_frames_ - 1))
  for _ in range(4):
    count_sum = count_sum.simplify()
  print("sum over counts with l=%s in any frame:" % label, count_sum)
  n, count_all = count_all_paths(fsa=fsa)
  count_all = count_all.subs(n, num_frames_)
  avg = count_sum / (num_frames_ * count_all)
  for _ in range(4):
    avg = avg.simplify()
  print("avg counts with l=%s in any frame:" % label, avg)
  if num_frames is not None:
    for t_ in range(1, num_frames + 1):
      print("  with T=%i ->" % t_, avg.subs(num_frames_, t_), "(%f)" % avg.subs(num_frames_, t_))
  print()


def count_all_paths_with_label_seq(fsa: Fsa, label_seq_template: str):
  """
  :param Fsa fsa:
  :param str label_seq_template: example "baab". this will get upsampled for num_frames, e.g. "bbaaaabb"
  """
  n = sympy.Symbol("n", integer=True, nonnegative=True)
  num_frames = len(label_seq_template) * n
  frame_idx = sympy.Symbol("frame_idx", integer=True, nonnegative=True)
  labels = set(label_seq_template)
  count_by_label = {}  # label -> count
  for output_label in sorted(labels):
    num_frames_, frame_idx_, count = count_all_paths_with_label_in_frame(fsa=fsa, label=output_label)
    count = count.subs(num_frames_, num_frames)
    count = count.subs(frame_idx_, frame_idx)
    count_by_label[output_label] = count

  num_frames_, count_all = count_all_paths(fsa=fsa)
  count_all = count_all.subs(num_frames_, num_frames)

  for output_label in sorted(labels):  # error signal for `label_`
    sum_for_output = 0
    sum_by_input = {}  # input_label -> count_sum
    for input_label in sorted(labels):  # combined error signal for frames with `label`
      count_sum = 0
      count_label_frames = 0  # e.g. 2n for "baab"
      for i in range(len(label_seq_template)):
        if label_seq_template[i] == input_label:
          count_sum += sympy.Sum(count_by_label[output_label], (frame_idx, n * i, n * i + n - 1))
          count_label_frames += n
      for _ in range(4):
        count_sum = count_sum.simplify()
      sum_by_input[input_label] = count_sum
      sum_for_output += count_sum
      print(
        "sum over counts with (frame)l=%s, (output)l=%s for seq template %r:" % (
          input_label, output_label, label_seq_template),
        count_sum)
      avg = count_sum / (count_label_frames * count_all)
      for _ in range(5):
        avg = avg.simplify()
      print("avg counts with l=%s,%s:" % (input_label, output_label), avg)
      for n_ in range(1, 11):
        avg_ = avg.subs(n, n_)
        print(
          "avg counts with (frame)l=%s, (output)l=%s, n=%i, T=%i:" % (
            input_label, output_label, n_, num_frames.subs(n, n_)),
          avg_, "(%f)" % avg_)

    for input_label in sorted(labels):
      avg_per_input = sum_by_input[input_label] / sum_for_output
      for _ in range(5):
        avg_per_input = avg_per_input.simplify()
      print(
        "avg (per input) with (frame)l=%s, (output)l=%s:" % (input_label, output_label),
        avg_per_input)
      for n_ in range(1, 11):
        avg_ = avg_per_input.subs(n, n_)
        print(
          "avg (per input) with (frame)l=%s, (output)l=%s, n=%i, T=%i:" % (
            input_label, output_label, n_, num_frames.subs(n, n_)),
          avg_, "(%f)" % avg_)
  print()


def count_all_paths_with_label_seq_partly_dominated(
    fsa: Fsa, label_seq_template: str, dom_label: str,
    n: typing.Union[int, sympy.Symbol], factor: typing.Union[int, float, sympy.Symbol],
    fixed_factor_power: typing.Optional[typing.Union[sympy.Symbol, sympy.Expr]] = None) -> (
      typing.Dict[typing.Tuple[str, str], typing.Dict[str, sympy.Expr]]):
  """
  Example label_seq_template = "BaaB".
  Case 1: l=B dominating in x=a. Count how much l=B on avg in x=B.
  Case 2: l=B dominating in x=B. Count how much l=B on avg in x=a.

  :param Fsa fsa:
  :param str label_seq_template:
  :param str dom_label:
  :param n: should be integer. positive.
  :param factor: float>1.0 or just 1. in any case positive.
  :param fixed_factor_power: see code...
  :return: dict input label (with prob_dom), other input label (uniform) -> label -> float expr (normalized or not)
  """
  labels = fsa.get_labels()
  assert dom_label in labels
  input_labels = set(label_seq_template)
  # num_frames = n * len(label_seq_template)
  res = {}

  for input_label in input_labels:
    count_frames_in_template = 0
    for i, input_label_ in enumerate(label_seq_template):
      if input_label_ == input_label:
        count_frames_in_template += 1

    # Now count all paths which have exactly count_frames_dom_label dom labels in the seq at the input label.
    # To do that, find relevant arcs for the input label.
    # This generic code is incomplete... Below we just hardcode it for our relevant case.

    parts = []
    parts_by_state = {}

    class Part:
      def __init__(self, state: int):
        self.start_state = state
        self.end_state = state
        self.arcs = set()
        self.loops_in_states = set()

      def add_arc(self, arc_: Arc):
        assert self.start_state <= arc_.source_state <= self.end_state
        self.arcs.add(arc_)
        self.end_state = max(self.end_state, arc_.target_state)
        if arc_.source_state == arc_.target_state:
          self.loops_in_states.add(arc_.source_state)

      def have_loop(self):
        return bool(self.loops_in_states)

      def __repr__(self):
        return "FsaPart{%s}" % self.arcs

    rem_part = None
    for arc in sorted(fsa.arcs, key=lambda arc_: (arc_.source_state, arc_.target_state, arc_.label)):
      if arc.label == dom_label:
        if arc.source_state not in parts_by_state:
          part = Part(state=arc.source_state)
          parts.append(part)
          parts_by_state[arc.source_state] = part
        else:
          part = parts_by_state[arc.source_state]
        part.add_arc(arc)
        parts_by_state[arc.target_state] = part
      else:
        if rem_part is None:
          rem_part = Part(state=arc.source_state)
        rem_part.add_arc(arc)

    assert len(parts) == 2  # not implemented otherwise
    assert all([part.have_loop() for part in parts])  # just not implemented otherwise
    assert parts[0].start_state == 0  # just not implemented otherwise
    assert parts[-1].start_state in fsa.final_states and parts[-1].end_state in fsa.final_states  # not implemented otw

    # Need to iterate through all possible cases where we have count_frames_dom_label in the input label.
    # 0 .. count_frames_in_template * n are possible.

    # Implement it using a similar algorithm like in count_all_paths_with_label_in_frame.

    # The code is somewhat specific to the label seq template "BaaB", consisting of 4 parts, 2 inputs (B and a).
    assert label_seq_template == "BaaB"  # not implemented otherwise (make the code more generic below...)

    for input_label_ in input_labels:
      if input_label_ != input_label:  # e.g. input_label == Blank, input_label_ == Label1.
        res_ = {label: 0 for label in labels}  # label -> sum expr. counted in input_label_.
        res[(input_label, input_label_)] = res_

        def _add():
          if input_label == BlankLabel:
            blank_num_frames_input = blank_num_frames_p1 + blank_num_frames_p4
          else:
            blank_num_frames_input = blank_num_frames_p23
          if input_label_ == BlankLabel:
            blank_num_frames_input_ = blank_num_frames_p1 + blank_num_frames_p4
          else:
            blank_num_frames_input_ = blank_num_frames_p23
          label1_num_frames_input_ = 2 * n - blank_num_frames_input_

          if fixed_factor_power is not None:
            blank_num_frames_input = sympy.sympify(blank_num_frames_input)
            syms = set(blank_num_frames_input.free_symbols)
            if syms.issubset({n}):
              eq = sympy.Eq(fixed_factor_power, blank_num_frames_input)
              sum_blank = sympy.Sum(
                sympy.Sum(blank_num_frames_input_, label1_end_frame_range),
                label1_start_frame_range)
              sum_label = sympy.Sum(
                sympy.Sum(label1_num_frames_input_, label1_end_frame_range),
                label1_start_frame_range)
              if eq == sympy.sympify(True):
                res_[BlankLabel] += sum_blank
                res_[Label1] += sum_label
              else:
                res_[BlankLabel] += sympy.Piecewise((sum_blank, eq), (0, True))
                res_[Label1] += sympy.Piecewise((sum_label, eq), (0, True))

            elif syms.issubset({n, label1_start_frame}):
              label1_start_frame_, = sympy.solve(blank_num_frames_input - fixed_factor_power, label1_start_frame)
              in_range = sympy.And(
                sympy.Ge(label1_start_frame_, label1_start_frame_range[1]),
                sympy.Le(label1_start_frame_, label1_start_frame_range[2])).simplify()
              sum_blank = sympy.Sum(blank_num_frames_input_, label1_end_frame_range)
              sum_label = sympy.Sum(label1_num_frames_input_, label1_end_frame_range)
              sum_blank = sum_blank.subs(label1_start_frame, label1_start_frame_)
              sum_label = sum_label.subs(label1_start_frame, label1_start_frame_)
              # print("in range s", in_range)
              res_[BlankLabel] += sympy.Piecewise((sum_blank, in_range), (0, True))
              res_[Label1] += sympy.Piecewise((sum_label, in_range), (0, True))

            elif syms.issubset({n, label1_end_frame}):
              label1_end_frame_, = sympy.solve(blank_num_frames_input - fixed_factor_power, label1_end_frame)
              assert set(label1_end_frame_.free_symbols).issubset({n, fixed_factor_power})
              in_range = sympy.And(
                sympy.Ge(label1_end_frame_, label1_end_frame_range[1]),
                sympy.Le(label1_end_frame_, label1_end_frame_range[2])).simplify()
              assert set(in_range.free_symbols).issubset({n, fixed_factor_power})
              sum_blank = sympy.Sum(
                blank_num_frames_input_.subs(label1_end_frame, label1_end_frame_),
                label1_start_frame_range)
              sum_label = sympy.Sum(
                label1_num_frames_input_.subs(label1_end_frame, label1_end_frame_),
                label1_start_frame_range)
              # print("in range e", in_range)
              res_[BlankLabel] += sympy.Piecewise((sum_blank, in_range), (0, True))
              res_[Label1] += sympy.Piecewise((sum_label, in_range), (0, True))

            else:
              assert syms.issubset({n, label1_start_frame, label1_end_frame})
              # Not implemented otherwise...
              assert set(sympy.sympify(blank_num_frames_input_).free_symbols).issubset({n})
              assert set(sympy.sympify(label1_num_frames_input_).free_symbols).issubset({n})

              label1_end_frame_, = sympy.solve(blank_num_frames_input - fixed_factor_power, label1_end_frame)
              assert set(label1_end_frame_.free_symbols).issubset({n, fixed_factor_power, label1_start_frame})

              total_cond = True
              rs = []  # type: typing.List[sympy.Interval]
              r1 = sympy.Ge(label1_end_frame_, label1_end_frame_range[1]).simplify()
              r2 = sympy.Le(label1_end_frame_, label1_end_frame_range[2]).simplify()
              if label1_start_frame not in r1.free_symbols:
                total_cond = sympy.And(total_cond, r1)
              else:
                rs.append(sympy.solve_univariate_inequality(r1, label1_start_frame, relational=False))
              rs.append(sympy.solve_univariate_inequality(r2, label1_start_frame, relational=False))
              rs.append(sympy.Interval(label1_start_frame_range[1], label1_start_frame_range[2]))

              r3 = 0
              r4 = 4 * n - 1
              for r_interval in rs:
                assert isinstance(r_interval, sympy.Interval)
                r3 = -sympy.Min(-r3, -r_interval.start)  # -min for simpler simplify
                r4 = sympy.Min(r4, r_interval.end)

              _c = fixed_factor_power
              c = r4 - r3 + 1  # We check c >= 0 below.
              c = c.replace(sympy.Min(0, -_c + n - 1), sympy.Min(_c, n - 1) - _c)  # fix for missing simplify
              # It follows some specific code to get multiple cases in the outer piecewise,
              # to remove min/max in this count `c`.
              assert isinstance(c, sympy.Expr)
              assert c.count(sympy.Min) == 1 and c.count(sympy.Max) == 0  # currently not implemented otherwise
              q, = list(c.find(sympy.Min))
              assert isinstance(q, sympy.Min)
              assert len(q.args) == 2  # not implemented otherwise
              min_args = list(q.args)

              case_cond = [sympy.Le(min_args[0], min_args[1]), sympy.Ge(min_args[0], min_args[1])]
              cases_blank = []
              cases_label = []
              for i_ in range(2):
                c_ = c.replace(q, min_args[i_])
                assert c_.count(sympy.Min) == 0
                sum_blank = blank_num_frames_input_ * c_  # because no other free symbols
                sum_label = label1_num_frames_input_ * c_  # because no other free symbols
                cond = sympy.And(total_cond, case_cond[i_], sympy.Ge(c_, 0)).simplify()
                # Some specific replacement rules because SymPy does not simplify those automatically...
                cond = cond.replace(((_c - 2*n >= -1) & (_c - 2*n <= -1)), sympy.Eq(_c, 2*n - 1))
                cond = cond.replace((sympy.Eq(_c, 2 * n) & sympy.Eq(_c - 2 * n, -1)), False)
                cond = cond.simplify()
                # print("cond'", cond)
                if cond != sympy.sympify(False):
                  cases_blank.append((sum_blank, cond))
                  cases_label.append((sum_label, cond))
              cases_blank.append((0, True))
              cases_label.append((0, True))
              sum_blank = sympy.Piecewise(*cases_blank)
              sum_label = sympy.Piecewise(*cases_label)
              res_[BlankLabel] += sum_blank
              res_[Label1] += sum_label

          else:
            factor_ = sympy.Pow(factor, blank_num_frames_input)
            res_[BlankLabel] += sympy.Sum(
              sympy.Sum(blank_num_frames_input_ * factor_, label1_end_frame_range),
              label1_start_frame_range)
            res_[Label1] += sympy.Sum(
              sympy.Sum(label1_num_frames_input_ * factor_, label1_end_frame_range),
              label1_start_frame_range)

        label1_start_frame = sympy.Symbol("label1_start_frame", integer=True)  # 0-indexed
        label1_end_frame = sympy.Symbol("label1_end_frame", integer=True)  # inclusive, 0-indexed
        # Case 1: label 1 starts within part 1.
        if True:
          label1_start_frame_range = (label1_start_frame, 0, n - 1)
          # Case 1: label 1 ends within part 1.
          label1_end_frame_range = (label1_end_frame, label1_start_frame, n - 1)
          label1_num_frames_p1 = label1_end_frame - label1_start_frame + 1
          blank_num_frames_p1 = n - label1_num_frames_p1
          blank_num_frames_p23 = 2 * n
          blank_num_frames_p4 = n
          _add()
          # Case 2: label 1 ends within part 2 or 3.
          label1_end_frame_range = (label1_end_frame, n, 3 * n - 1)
          label1_num_frames_p1 = n - label1_start_frame
          blank_num_frames_p1 = n - label1_num_frames_p1
          label1_num_frames_p23 = label1_end_frame - n + 1
          blank_num_frames_p23 = 2 * n - label1_num_frames_p23
          blank_num_frames_p4 = n
          _add()
          # Case 3: label 1 ends within part 4.
          label1_end_frame_range = (label1_end_frame, 3 * n, 4 * n - 1)
          label1_num_frames_p1 = n - label1_start_frame
          blank_num_frames_p1 = n - label1_num_frames_p1
          blank_num_frames_p23 = 0
          label1_num_frames_p4 = label1_end_frame - 3 * n + 1
          blank_num_frames_p4 = n - label1_num_frames_p4
          _add()
        # Case 2: label 1 starts within part 2 or 3.
        if True:
          label1_start_frame_range = (label1_start_frame, n, 3 * n - 1)
          blank_num_frames_p1 = n
          # Case 1: label 1 ends within part 2 or 3.
          label1_end_frame_range = (label1_end_frame, label1_start_frame, 3 * n - 1)
          label1_num_frames_p23 = label1_end_frame - label1_start_frame + 1
          blank_num_frames_p23 = 2 * n - label1_num_frames_p23
          blank_num_frames_p4 = n
          _add()
          # Case 2: label 1 ends within part 4.
          label1_end_frame_range = (label1_end_frame, 3 * n, 4 * n - 1)
          blank_num_frames_p23 = label1_start_frame - n
          label1_num_frames_p4 = label1_end_frame - 3 * n + 1
          blank_num_frames_p4 = n - label1_num_frames_p4
          _add()
        # Case 3: label 1 starts within part 4.
        if True:
          label1_start_frame_range = (label1_start_frame, 3 * n, 4 * n - 1)
          blank_num_frames_p1 = n
          blank_num_frames_p23 = 2 * n
          # label 1 ends within part 4.
          label1_end_frame_range = (label1_end_frame, label1_start_frame, 4 * n - 1)
          label1_num_frames_p4 = label1_end_frame - label1_start_frame + 1
          blank_num_frames_p4 = n - label1_num_frames_p4
          _add()

        if isinstance(factor, int):
          for _ in range(5):
            for label in labels:
              x = res_[label]
              if fixed_factor_power is not None:
                # Some specific replacement rules because SymPy does not simplify those automatically...
                x = sympy_utils.simplify_and(x)
              x = x.simplify()
              res_[label] = x

  return res


def count_all_paths_with_label_seq_partly_dominated_inefficient(
    fsa: Fsa, label_seq_template: str, dom_label: str, n: int, prob_dom: float,
    normalized: bool = True, verbosity: int = 0) -> (
      typing.Dict[typing.Tuple[str, str], typing.Dict[str, float]]):
  """
  Same as :func:`count_all_paths_with_label_seq_partly_dominated`.
  For each input label, assume prob_dom and uniform in the other input labels, and then count.

  :param Fsa fsa: e.g. for 1 label B*a+B*
  :param str label_seq_template: e.g. "BaaB"
  :param str dom_label: e.g. B
  :param int n: multiplicator for seq template
  :param float prob_dom:
  :param bool normalized: return value normalized or not
  :param int verbosity: 0 is no output
  :return: input label (with prob_dom), other input label (uniform) -> label -> float (normalized or not)
  """
  labels = fsa.get_labels()
  assert dom_label in labels
  input_labels = set(label_seq_template)
  num_frames = n * len(label_seq_template)
  res = {}
  counts = defaultdict(int)
  counts_by_t = defaultdict(int)
  seqs = defaultdict(list)
  for path in iterate_all_paths(fsa=fsa, num_frames=num_frames):
    assert len(path) == num_frames
    for input_label in input_labels:
      count_frames_by_label = {label: 0 for label in labels}
      for i, input_label_ in enumerate(label_seq_template):
        if input_label_ == input_label:
          for j in range(i * n, i * n + n):
            count_frames_by_label[path[j].label] += 1

      count_frames_dom_label = count_frames_by_label[dom_label]
      counts[(input_label, count_frames_dom_label)] += 1
      seqs[(input_label, count_frames_dom_label)].append("".join([arc.label for arc in path]))
      for i, input_label_ in enumerate(label_seq_template):
        if input_label_ != input_label:
          for j in range(i * n, i * n + n):
            counts_by_t[(input_label, count_frames_dom_label, input_label_, path[j].label)] += 1

  v = {i: sys.stdout if i < verbosity else open(os.devnull, "w") for i in range(3)}
  print("Dom label: %s" % dom_label, file=v[1])
  for input_label in input_labels:
    print("Input %s in %s, n=%i, T=%i." % (input_label, label_seq_template, n, num_frames), file=v[0])
    rel_total_counts_by_label = defaultdict(int)
    counts_frames_dom_label = [c for input_label_, c in counts.keys() if input_label_ == input_label]
    max_count_frames_dom_label = max(counts_frames_dom_label)
    print(" Max count of dom label %s in label seq at this input:" % dom_label, max_count_frames_dom_label, file=v[0])
    for count_frames_dom_label in range(max_count_frames_dom_label + 1):
      print(" For count of dom label %i in input %s:" % (count_frames_dom_label, input_label), file=v[1])
      print("  Num seqs:", counts[input_label, count_frames_dom_label], file=v[1])
      print("  Seqs:", seqs[input_label, count_frames_dom_label], file=v[2])
      for input_label_ in input_labels:
        if input_label_ != input_label:
          count_all_frames = 0
          print("  Look at other input %s:" % input_label_, file=v[1])
          for i, input_label__ in enumerate(label_seq_template):
            if input_label__ == input_label_:
              count_all_frames += n
          for label in labels:
            a = counts_by_t[(input_label, count_frames_dom_label, input_label_, label)]
            b = counts[input_label, count_frames_dom_label] * count_all_frames
            print("   Count label %s: %i/%i (%f)" % (label, a, b, float(a) / b), file=v[1])
            if prob_dom == 0.5:
              rel_total_counts_by_label[(input_label_, label)] += a  # keep int
            else:
              rel_total_counts_by_label[(input_label_, label)] += (
                a * ((prob_dom / (1. - prob_dom)) ** count_frames_dom_label))
    for input_label_ in input_labels:
      if input_label_ != input_label:
        res_by_label = {}
        z = sum([rel_total_counts_by_label[input_label_, label] for label in labels])
        for label in labels:
          print(" (avg q(%s|x=%s)) Relative count for input %s, label %s, p(%s|x=%s)=%f: %f" % (
            label, input_label_,
            input_label_, label, dom_label, input_label, prob_dom,
            float(rel_total_counts_by_label[input_label_, label]) / z), file=v[0])
          res_by_label[label] = rel_total_counts_by_label[input_label_, label]
          if normalized:
            res_by_label[label] /= z
        res[(input_label, input_label_)] = res_by_label

  return res


def full_sum(fsa: Fsa, label_seq_template: str):
  n = sympy.Symbol("n", integer=True, nonnegative=True)
  # num_frames = len(label_seq_template) * n
  states = sorted(fsa.states)
  input_labels = sorted(set(label_seq_template))
  labels = fsa.get_labels()
  probs_by_label_by_input = {input_label: {} for input_label in input_labels}
  prob_vars = []
  for input_label in input_labels:
    s = 0
    for label in labels[:-1]:
      probs_by_label_by_input[input_label][label] = sympy.Symbol("prob_%s_in_%s" % (label, input_label))
      s += probs_by_label_by_input[input_label][label]
      prob_vars.append(probs_by_label_by_input[input_label][label])
    probs_by_label_by_input[input_label][labels[-1]] = 1 - s

  initial_vec = sympy.Matrix([[1, 0, 0]])
  final_vec = sympy.Matrix([[1] if state in fsa.final_states else [0] for state in states])
  v = initial_vec
  trans_mat_product = None
  for i in range(len(label_seq_template)):
    probs_by_label = probs_by_label_by_input[label_seq_template[i]]
    trans_mat = sympy.Matrix([
      [fsa.get_deterministic_source_to_target_prob(
        source_state=src_state, target_state=tgt_state, probs_by_label=probs_by_label)
       for tgt_state in states]
      for src_state in states])
    trans_mat = sympy.Pow(trans_mat, n)
    v *= trans_mat
    if trans_mat_product is None:
      trans_mat_product = trans_mat
    else:
      trans_mat_product *= trans_mat
  res = v * final_vec
  assert res.shape == (1, 1)
  res = res[0, 0]

  print("params:", prob_vars)
  for subs in [[(prob_vars[0], 1), (n, 4)]]:
    print("calc:")
    res_ = res
    for sub_var, sub_val in subs:
      res_ = res_.subs(sub_var, sub_val)
      print("sub", sub_var, "by", sub_val)
    for _ in range(4):
      res_ = res_.simplify()

    v = prob_vars[1]
    d = sympy.diff(res_, v)
    for _ in range(2):
      d = d.simplify()
    opts = sympy.solve(d, v)
    print("optima:", opts)
    print("max:", sympy.maximum(res_, v, sympy.Interval(0, 1)))
    for opt in opts:
      opt = opt.simplify()
      print(res_.subs(v, opt).simplify().doit())


BlankLabel = "B"
Label1 = "a"
Label2 = "b"
Label3 = "c"
Label4 = "d"
Label1StrTemplate = "BaaB"
Label1Str2TimesTemplate = "BBaaBaaB"
Label2StrTemplate = "BaaaBbbbB"


def get_std_fsa_1label():
  fsa = Fsa()
  # fsa for "b*a+b*"
  fsa.add_arc(0, 0, BlankLabel)
  fsa.add_arc(0, 1, Label1)
  fsa.add_arc(1, 1, Label1)
  fsa.add_arc(1, 2, BlankLabel)
  fsa.add_arc(2, 2, BlankLabel)
  fsa.add_final_state(1)
  fsa.add_final_state(2)
  return fsa


def get_std_fsa_1label_2times():
  fsa = Fsa()
  # fsa for "b*a+b+a+b*"
  fsa.add_arc(0, 0, BlankLabel)
  fsa.add_arc(0, 1, Label1)
  fsa.add_arc(1, 1, Label1)
  fsa.add_arc(1, 2, BlankLabel)
  fsa.add_arc(2, 2, BlankLabel)
  fsa.add_arc(2, 3, Label1)
  fsa.add_arc(3, 3, Label1)
  fsa.add_arc(3, 4, BlankLabel)
  fsa.add_arc(4, 4, BlankLabel)
  fsa.add_final_state(3)
  fsa.add_final_state(4)
  return fsa


def get_std_fsa_2label():
  fsa = Fsa()
  # fsa for "b*0+b*1+b*"
  fsa.add_arc(0, 0, BlankLabel)
  fsa.add_arc(0, 1, Label1)
  fsa.add_arc(1, 1, Label1)
  fsa.add_arc(1, 2, BlankLabel)
  fsa.add_arc(2, 2, BlankLabel)
  fsa.add_arc(1, 3, Label2)
  fsa.add_arc(2, 3, Label2)
  fsa.add_arc(3, 3, Label2)
  fsa.add_arc(3, 4, BlankLabel)
  fsa.add_arc(4, 4, BlankLabel)
  fsa.add_final_state(3)
  fsa.add_final_state(4)
  return fsa


def get_std_fsa_3label_blank():
  fsa = Fsa()
  # fsa for "b*0+b*1+b*2+b*"
  fsa.add_arc(0, 0, BlankLabel)
  fsa.add_arc(0, 1, Label1)
  fsa.add_arc(1, 1, Label1)
  fsa.add_arc(1, 2, BlankLabel)
  fsa.add_arc(2, 2, BlankLabel)
  fsa.add_arc(1, 3, Label2)
  fsa.add_arc(2, 3, Label2)
  fsa.add_arc(3, 3, Label2)
  fsa.add_arc(3, 4, BlankLabel)
  fsa.add_arc(4, 4, BlankLabel)
  fsa.add_arc(3, 5, Label3)
  fsa.add_arc(4, 5, Label3)
  fsa.add_arc(5, 5, Label3)
  fsa.add_arc(5, 6, BlankLabel)
  fsa.add_arc(6, 6, BlankLabel)
  fsa.add_final_state(5)
  fsa.add_final_state(6)
  return fsa


def get_std_fsa_3label_sil():
  fsa = Fsa()
  # fsa for "b*0+1+2+b*"
  fsa.add_arc(0, 0, BlankLabel)
  fsa.add_arc(0, 1, Label1)
  fsa.add_arc(1, 1, Label1)
  fsa.add_arc(1, 2, Label2)
  fsa.add_arc(2, 2, Label2)
  fsa.add_arc(2, 3, Label3)
  fsa.add_arc(3, 3, Label3)
  fsa.add_arc(3, 4, BlankLabel)
  fsa.add_arc(4, 4, BlankLabel)
  fsa.add_final_state(3)
  fsa.add_final_state(4)
  return fsa


def get_std_fsa_4label_2words_blank():
  fsa = Fsa()
  # fsa for "b*0+b*1+b*2+b*3+b*"
  fsa.add_arc(0, 0, BlankLabel)
  fsa.add_arc(0, 1, Label1)
  fsa.add_arc(1, 1, Label1)
  fsa.add_arc(1, 2, BlankLabel)
  fsa.add_arc(2, 2, BlankLabel)
  fsa.add_arc(1, 3, Label2)
  fsa.add_arc(2, 3, Label2)
  fsa.add_arc(3, 3, Label2)
  fsa.add_arc(3, 4, BlankLabel)
  fsa.add_arc(4, 4, BlankLabel)
  fsa.add_arc(3, 5, Label3)
  fsa.add_arc(4, 5, Label3)
  fsa.add_arc(5, 5, Label3)
  fsa.add_arc(5, 6, BlankLabel)
  fsa.add_arc(6, 6, BlankLabel)
  fsa.add_arc(5, 7, Label4)
  fsa.add_arc(6, 7, Label4)
  fsa.add_arc(7, 7, Label4)
  fsa.add_arc(7, 8, BlankLabel)
  fsa.add_arc(8, 8, BlankLabel)
  fsa.add_final_state(7)
  fsa.add_final_state(8)
  return fsa


def get_std_fsa_4label_2words_sil():
  fsa = Fsa()
  # fsa for "b*0+1+2+b*3+b*"
  fsa.add_arc(0, 0, BlankLabel)
  fsa.add_arc(0, 1, Label1)
  fsa.add_arc(1, 1, Label1)
  fsa.add_arc(1, 2, Label2)
  fsa.add_arc(2, 2, Label2)
  fsa.add_arc(2, 3, Label3)
  fsa.add_arc(3, 3, Label3)
  fsa.add_arc(3, 4, BlankLabel)
  fsa.add_arc(4, 4, BlankLabel)
  fsa.add_arc(3, 5, Label4)
  fsa.add_arc(4, 5, Label4)
  fsa.add_arc(5, 5, Label4)
  fsa.add_arc(5, 6, BlankLabel)
  fsa.add_arc(6, 6, BlankLabel)
  fsa.add_final_state(5)
  fsa.add_final_state(6)
  return fsa


def test_count_all_paths(fsa: Fsa, num_frames: int):
  c_ = count_all_paths_inefficient(fsa=fsa, num_frames=num_frames)
  print("count all paths for T=%i explicit:" % num_frames, c_)
  n, c = count_all_paths(fsa=fsa)
  print("count all paths symbolic:", n, "->", c)
  c__ = c.subs(n, num_frames).doit()
  print("count all paths for T=%i via symbolic:" % num_frames, c__)
  assert c_ == c__
  num_labels = len(fsa.get_labels())
  print("L with uniform distribution:", numpy.log(num_labels) * num_frames - numpy.log(c_))
  print("L with uniform distribution (inexact):", -numpy.log(((1.0 / num_labels) ** num_frames) * c_))
  print()


def test_count_all_paths_with_label_in_frame(fsa: Fsa, num_frames: int, frame_idx: int, label: str):
  c_ = count_all_paths_with_label_in_frame_inefficient(fsa=fsa, num_frames=num_frames, frame_idx=frame_idx, label=label)
  print("count all paths with t=%i, T=%i, l=%s explicit:" % (frame_idx, num_frames, label), c_)
  n, t, c = count_all_paths_with_label_in_frame(fsa=fsa, label=label)
  print("count all paths with l=%s symbolic:" % label, "(%s, %s) -> %s" % (n, t, c))
  n_t = sympy.Symbol("T", integer=True)  # just looks nicer
  t1 = sympy.Symbol("t", integer=True)
  print("count all paths with l=%s symbolic (t index 1):" % label, "(%s, %s) -> %s" % (
    n, t, c.subs(n, n_t).subs(t, t1 - 1).simplify()))
  c__ = c.subs(n, num_frames).subs(t, frame_idx).doit()
  print("count via symbolic:", c__,)
  assert c_ == c__
  n, count_all = count_all_paths(fsa=fsa)
  count_all = count_all.subs(n, num_frames)
  print("  / %i, fraction %f" % (count_all, c__ / count_all))
  print()


def count_paths_with_label(fsa: Fsa, num_frames: int, label: str):
  _n, _t, count_blank_sym = count_all_paths_with_label_in_frame(fsa=fsa, label=label)
  n_t = sympy.Symbol("T", integer=True)  # just looks nicer
  t1 = sympy.Symbol("t", integer=True)  # in 1..T, not 0..T-1
  count_blank_sym = count_blank_sym.subs(_n, n_t).subs(_t, t1 - 1).simplify()
  _n, count_all_sym = count_all_paths(fsa=fsa)
  count_all_sym = count_all_sym.subs(_n, n_t)
  for n_ in range(1, num_frames + 1):
    count_all = int(count_all_sym.subs(n_t, n_).doit())
    count_blank_dominating_frames = 0
    for t in range(1, n_ + 1):
      count_blank = int(count_blank_sym.subs(n_t, n_).subs(t1, t).doit())
      if count_blank * 2 > count_all:
        count_blank_dominating_frames += 1
    print(
      "T=%i, num frames dominated by label=%s:" % (n_, label), count_blank_dominating_frames,
      "(%f)" % (float(count_blank_dominating_frames) / n_))
  # Purely symbolic...
  # Show NumDom >= 3/4 T.
  expr_lhs = count_blank_sym * 2 - count_all_sym
  expr_lhs = expr_lhs.simplify()
  expr = sympy.GreaterThan(expr_lhs, 0)
  print(expr.simplify())
  assert expr_lhs.is_polynomial(t1)
  # solve_poly_inequality does not work yet, not implemented...
  t_real = sympy.Symbol("t_")
  expr_lhs = expr_lhs.subs(t1, t_real)
  zero1, zero2 = sympy.solve(expr_lhs, t_real)
  print("zeros:", [zero1, zero2])
  count_blank_dominating_frames_sym = sympy.ceiling(zero1 - 1) + sympy.ceiling(n_t - zero2)
  print("num frames dominated by label=%s symbolic:" % label, count_blank_dominating_frames_sym)
  count_blank_dominating_frames_small_lim_sym = zero1 - 1 + n_t - zero2
  rel_count_blank_dominating_frames_sym = count_blank_dominating_frames_small_lim_sym / n_t
  print("pessimistic relative count:", rel_count_blank_dominating_frames_sym)
  for n_ in range(1, num_frames + 1):
    print(
      "T=%i symbolic:" % n_,
      count_blank_dominating_frames_sym.subs(n_t, n_),
      float(count_blank_dominating_frames_small_lim_sym.subs(n_t, n_)),
      "relative:", float(rel_count_blank_dominating_frames_sym.subs(n_t, n_)))


def match(fsa: Fsa, input_seq: str) -> typing.Optional[typing.List[Arc]]:
  """
  Match the input_seq to the FSA.
  Assumes that the FSA is deterministic by label.
  """
  path = []
  state = fsa.start_state
  for label in input_seq:
    next_state = None
    for arc in fsa.arcs_by_source_state[state]:
      if arc.label == label:
        next_state = arc.target_state
        path.append(arc)
        break
    if next_state is None:
      return None
    state = next_state
  if state in fsa.final_states:
    return path
  return None


def bias_model(fsa: Fsa, num_frames: int):
  # Half symbolic, fixed T...
  print("Bias model with T=%i:" % num_frames)
  labels = fsa.get_labels()
  label_probs = {}  # label -> prob
  for label in labels[:-1]:
    label_probs[label] = sympy.Symbol("prob_%s" % label, real=True, nonnegative=True)
  label_probs[labels[-1]] = 1 - sum([label_probs[label] for label in labels[:-1]])
  prob_sum = 0
  for path in iterate_all_paths(fsa=fsa, num_frames=num_frames):
    prob_path = 1
    for label in labels:
      label_arcs = [arc for arc in path if arc.label == label]
      if label_arcs:
        prob_path *= sympy.Pow(label_probs[label], len(label_arcs))
    prob_sum += prob_path
  assert isinstance(prob_sum, sympy.Expr)
  print(prob_sum)
  label_prob0 = label_probs[labels[0]]
  for p in [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
    x = float(prob_sum.subs(label_prob0, p))
    print("p(%s)=%f ->" % (labels[0], p), x, -numpy.log(x))
  opts = sympy.solve(prob_sum.diff(label_prob0), label_prob0)
  opts = [opt.simplify() for opt in opts]
  print(opts)
  for p in opts:
    x = float(prob_sum.subs(label_prob0, p))
    print("p(%s)=%f ->" % (labels[0], p), x, -numpy.log(x))

  print()


def bias_model_1label(num_frames: int):
  # Half symbolic, fixed FSA...
  print("Bias model with fixed FSA.")
  labels = ["B", "a"]
  label_prob0 = sympy.Symbol("prob_B", real=True)
  i = sympy.Symbol("i", integer=True, positive=True)
  n = sympy.Symbol("T", integer=True, positive=True)
  prob_sum = sympy.Sum(
    i * sympy.Pow(label_prob0, i - 1) * sympy.Pow(1 - label_prob0, n - i + 1),
    (i, 2, n)) + sympy.Pow(1 - label_prob0, n)
  # prob_sum = prob_sum.simplify()
  assert isinstance(prob_sum, sympy.Expr)
  print(prob_sum)
  for p in [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
    print(
      "p(%s)=%f ->" % (labels[0], p),
      prob_sum.subs(label_prob0, p),
      "with T=%i:" % num_frames,
      prob_sum.subs(label_prob0, p).subs(n, num_frames).doit())

  d = prob_sum.diff(label_prob0)
  opts = sympy.solve(d, label_prob0)
  print(opts)  # seems wrong... has label_prob0 in it
  print()


def test_count_all_paths_with_label_seq_partly_dominated(recalc=False, check=False, check_with_factor=False):
  fsa = get_std_fsa_1label()
  n_ = 4
  n = sympy.Symbol("n", integer=True, positive=True)
  factor = sympy.Symbol("fact", real=True, positive=True)

  # Factor 1 <=> prob_dom 0.5. Will just yield normal counts.
  res = count_all_paths_with_label_seq_partly_dominated_inefficient(
    fsa=fsa, label_seq_template=Label1StrTemplate, dom_label=BlankLabel, n=n_,
    prob_dom=0.5, normalized=False, verbosity=2)
  print(res)
  if recalc:
    res_ = count_all_paths_with_label_seq_partly_dominated(
      fsa=fsa, label_seq_template=Label1StrTemplate, dom_label=BlankLabel, n=n_, factor=1)
  else:
    # It's a bit slow, so for simpler experiments, just copy it here.
    res_ = {
      ('B', 'a'): {
        'B': 2 * n * (13 * n ** 2 - 1) / 3,
        'a': 2 * n * (11 * n ** 2 + 6 * n + 1) / 3},
      ('a', 'B'): {
        'B': 2 * n * (19 * n ** 2 - 1) / 3,
        'a': 2 * n * (5 * n ** 2 + 6 * n + 1) / 3}}
  print(res_)
  for key, counts in res_.items():
    assert set(counts.keys()) == {BlankLabel, Label1}
    for label, c in counts.items():
      c__ = c.subs(n, n_).doit()
      print("%s, label %s ->" % (key, label), c, "=", c__)
      assert c__ == res[key][label]  # exactly the same counts

  c = fixed_factor_power = sympy.Symbol("c", integer=True, nonnegative=True)
  if recalc:
    res__ = count_all_paths_with_label_seq_partly_dominated(
      fsa=fsa, label_seq_template=Label1StrTemplate, dom_label=BlankLabel, n=n_, factor=1,
      fixed_factor_power=fixed_factor_power)
  else:
    # It's slow, so for simpler experiments, just copy it here.
    res__ = {
      ('B', 'a'): {
        'B': sympy.Piecewise(
          (2*n*(4*n**2 - 1)/3, sympy.Eq(c, 2*n)),
          (2*n*(c + 2*n), sympy.Eq(c - 2*n, -1)),
          (2*n*(2*c + 1), (c >= n) & (c - 2*n <= -1)),
          (4*n*(c - n + 1), (c - n >= -1) & (c - 2*n <= -1)),
          (0, True)),
        'a': sympy.Piecewise(
          (2*n*(2*n**2 + 3*n + 1)/3, sympy.Eq(c, 2*n)),
          (2*n*(-c + 4*n), (c >= n) & (c - 2*n <= -1)),
          (2*n*(c + 1), c - n <= -1),
          (2*n*(-c + 2*n - 1), (c - n >= -1) & (c - 2*n <= -1)),
          (0, True))},
      ('a', 'B'): {
        'B': sympy.Piecewise(
          (n*(n**2 + 2*n + 1), sympy.Eq(c, 0)),
          (n*(5*n**2 + 3*n - 2)/3, sympy.Eq(c, 2*n)),
          (n*(2*c + 3*n + 1), c - 2*n <= -1),
          (4*n**2, sympy.Eq(c - 2*n, -1)),
          (0, True)),
        'a': sympy.Piecewise(
          (n*(n**2 + 2*n + 1), sympy.Eq(c, 0)),
          (n*(n**2 + 3*n + 2)/3, sympy.Eq(c, 2*n)),
          (n*(n + 1), c - 2*n <= -1),
          (0, True))}}
  print(res__)
  for key, counts in res__.items():
    assert set(counts.keys()) == {BlankLabel, Label1}
    for label, c in counts.items():
      print("%s, label %s ->" % (key, label), c)
      if check:
        c_sum = 0
        for fixed_count in range(0, 2 * n_ + 1):
          c__ = c.subs({n: n_, fixed_factor_power: fixed_count}).doit()
          print("  c %i -> %i" % (fixed_count, c__))
          # We could also check each count individually here, by returning those numbers
          # in the inefficient explicit algo above.
          # However, for simplicity, just check the sum.
          c_sum += c__
        assert c_sum == res[key][label]  # exactly the same counts
    d = counts[BlankLabel] - counts[Label1]
    d = d.simplify()
    d = sympy_utils.simplify_and(d)
    print("diff:", d)
    # sympy.ceiling...
    #print(
    #  "first pos'",
    #  sympy_utils.sum_over_piecewise(
    #    d, fixed_factor_power, sympy.floor((4 * n - 1) / 3), sympy.ceiling((4 * n - 1) / 3)))
    # print("diff:", d)
    # print(" >0:", sympy.Gt(d, 0).simplify())
    for i in range(3):
      print("assume ((4 * n - 1 + i) / 3) natural number, i = %i." % i)
      print("  first pos i%i" % i, d.subs(fixed_factor_power, ((4 * n - 1 + i) / 3)).simplify())
      print("  last neg i%i" % i, d.subs(fixed_factor_power, ((4 * n - 1 + i) / 3) - 1).simplify())
      s_pos = sympy_utils.sum_over_piecewise(
        d, fixed_factor_power, ((4 * n - 1 + i) / 3), 2*n, extra_condition=sympy.Ge(n, 4))
      print("  diff sum c={(4n-1+%i)/3}^2n:" % i, s_pos)
      s_neg = sympy_utils.sum_over_piecewise(
        d, fixed_factor_power, 0, ((4 * n - 1 + i) / 3 - 1), extra_condition=sympy.Ge(n, 4))
      print("  diff sum c=0^{(4n-1+%i)/3-1}:" % i, s_neg)
      print("  tot:", (s_pos + s_neg).simplify())
    s = sympy_utils.sum_over_piecewise(d, fixed_factor_power, 0, 2*n)
    print("diff sum c=0^2n:", s)
    s = sympy_utils.sum_over_piecewise(d, fixed_factor_power, n, 2*n)
    print("diff sum c=n^2n:", s)
    s = sympy_utils.sum_over_piecewise(d, fixed_factor_power, 0, n - 1)
    print("diff sum c=0^{n-1}:", s)
    s = sympy_utils.sum_over_piecewise(counts[BlankLabel], fixed_factor_power, 0, 2*n)
    print("blank sum c=0^2n:", s)

  if check_with_factor:
    res = count_all_paths_with_label_seq_partly_dominated_inefficient(
      fsa=fsa, label_seq_template=Label1StrTemplate, dom_label=BlankLabel, n=n_, prob_dom=0.6, verbosity=1)
    print(res)
    res_ = count_all_paths_with_label_seq_partly_dominated(
      fsa=fsa, label_seq_template=Label1StrTemplate, dom_label=BlankLabel, n=n, factor=factor)
    print(res_)
    for key, counts in res_.items():
      print("key:", key)
      assert set(counts.keys()) == {BlankLabel, Label1}
      d = counts[BlankLabel] - counts[Label1]
      print(d)
      #d = d.simplify()
      #print(d)
    # Does not really simplify...


def gen_model_1label():
  """
  \\sum_{s:y} p(x|s),
  two possible inputs x1 (1,0) and x2 (0,1),
  two possible labels "a" and (blank) "B".
  Define p(x1|s=a) = theta_a, p(x2|s=a) = 1 - theta_a,
  p(x2|s=B) = theta_B, p(x1|s=B) = 1 - theta_B.

  For simplicity, fsa ^= a*B*, and the input be x1^{na},x2^{nB}, T = na + nB.
  Then we can just count. All alignments can be iterated through by t=0...T.
  Symmetric case...
  """
  na = sympy.Symbol("na", integer=True, positive=True)
  nb = sympy.Symbol("nb", integer=True, positive=True)
  theta_a = sympy.Symbol("theta_a", real=True, nonnegative=True)
  theta_b = sympy.Symbol("theta_b", real=True, nonnegative=True)
  t = sympy.Symbol("t", integer=True, nonnegative=True)
  # Make 2 parts of the sum, one t=0...na, another t=na..T.
  # Should get rid of the min/max cases, simplify it.
  p1 = sympy.Pow(theta_a, sympy.Min(t, na))
  p2 = sympy.Pow(1 - theta_a, sympy.Max(t - na, 0))
  p3 = sympy.Pow(theta_b, sympy.Min(na + nb - t, nb))  # exp = min(na - t, 0) + nb
  p4 = sympy.Pow(1 - theta_b, sympy.Max(na - t, 0))
  sum_ = sympy.Sum(p1 * p2 * p3 * p4, (t, 0, na + nb))
  # Case: t == 0.
  p1 = 1
  p2 = 1
  p3 = sympy.Pow(theta_b, nb)
  p4 = sympy_utils.polynomial_exp(1, -theta_b, na, expand=False)
  s0 = p1 * p2 * p3 * p4
  # Case: t < na.
  p1 = sympy.Pow(theta_a, t)
  p2 = 1
  p3 = sympy.Pow(theta_b, nb)
  p4 = sympy_utils.polynomial_exp(1, -theta_b, na - t, expand=False)
  sum1_ = sympy.Sum(p1 * p2 * p3 * p4, (t, 0, na))
  # Case: t == na.
  p1 = sympy.Pow(theta_a, na)
  p2 = 1
  p3 = sympy.Pow(theta_b, nb)
  p4 = 1
  s1 = p1 * p2 * p3 * p4
  # Case: t > na, t < na + nb
  p1 = sympy.Pow(theta_a, na)
  p2 = sympy_utils.polynomial_exp(1, -theta_a, t - na, expand=False)
  p3 = sympy.Pow(theta_b, na + nb - t)
  p4 = 1
  sum2_ = sympy.Sum(p1 * p2 * p3 * p4, (t, na + 1, na + nb))
  # Case: t == na + nb
  p1 = sympy.Pow(theta_a, na)
  p2 = sympy_utils.polynomial_exp(1, -theta_a, nb, expand=False)
  p3 = 1
  p4 = 1
  s2 = p1 * p2 * p3 * p4
  #sum_ = sum1_ + sum2_  # + s0 + s1 + s2
  #p1 = sympy.Pow(theta_a, t)
  #p4 = sympy.Pow(1 - theta_b, na - t)
  #sum_ = sympy.Sum(p1 * p4, (t, 1, na - 1))
  for _ in range(6):
    sum_ = sum_.simplify()
  print(sum_)

  sum__ = sum_.subs(na, 10).subs(nb, 10)
  xs = ys = numpy.linspace(0, 1., num=11)
  values = numpy.zeros((len(xs), len(ys)))
  for ix, x in enumerate(xs):
    for iy, y in enumerate(ys):
      value = sum__.subs(theta_a, x).subs(theta_b, y).doit()
      print("theta = (%f, %f) -> sum = %s" % (x, y, value))
      #values[ix, iy] = float(sum__.subs(theta_a, x).subs(theta_b, y).doit())
  #print(values)

  syms = (theta_a, theta_b)
  syms = (theta_a,)
  # sum_ = sum_.simplify()  -- makes it actually harder?
  sum_diff = sum_.diff(*syms)
  print("diff:", sum_diff)
  for _ in range(5):
    sum_diff = sum_diff.simplify()
    print(sum_diff)
  # sum_diff = sum_diff.simplify()  # -- also makes it harder?
  opts = sympy.solve(sum_diff, *syms)
  print("num opts:", len(opts))
  print("opts:", opts)


def gen_model_1label_bab():
  n = sympy.Symbol("n", integer=True, positive=True)
  t_end = n * 4
  theta_a = sympy.Symbol("theta_a", real=True, nonnegative=True)
  theta_b = sympy.Symbol("theta_b", real=True, nonnegative=True)
  # s_1^t1 === b
  t1 = sympy.Symbol("t1", integer=True, nonnegative=True)
  # s_{t1+1}^t2 === a
  t2 = sympy.Symbol("t2", integer=True, nonnegative=True)
  # s_{t2+1}^{n*4} === b

  # TODO ...
  # Iterate through num_correct_b, 0..2n.
  num_correct_b = sympy.Symbol("num_correct_b", integer=True, nonnegative=True)
  num_wrong_a = 2 * n - num_correct_b

  # Iterate through num_correct_b_left, max(num_correct_b-n,0)..min(num_correct_b,n).
  num_correct_b_left = sympy.Symbol("num_correct_b_left", integer=True, nonnegative=True)

  # Iterate through num_correct_a, 0..2n.
  num_correct_a = sympy.Symbol("num_correct_a", integer=True, nonnegative=True)

  # Iterate through num_correct_a_left, ...
  num_wrong_b = 0  # TODO ...

  p1 = sympy.Pow(theta_b, num_correct_b)
  p2 = sympy_utils.polynomial_exp(1, -theta_b, num_wrong_b)
  p3 = sympy.Pow(theta_a, num_correct_a)
  p4 = sympy_utils.polynomial_exp(1, -theta_b, num_wrong_a)
  p = p1 * p2 * p3 * p4

  p = sympy.Sum(p, (t2, t1 + 1, t_end))
  p = sympy.Sum(p, (t1, 0, t_end - 1))

  print(p)
  for _ in range(4):
    p = p.simplify()
    print(p)
  diff = p.diff(theta_a)
  print("diff:", diff)
  print(sympy.solve(diff, theta_a))


def gen_model_fsa_template_via_matrix(fsa: Fsa, label_seq_template: str):
  n = sympy.Symbol("n", integer=True, nonnegative=True)
  num_frames = len(label_seq_template) * n
  states = sorted(fsa.states)
  input_labels = sorted(set(label_seq_template))
  labels = fsa.get_labels()
  probs_by_label_by_input = {input_label: {} for input_label in input_labels}
  prob_vars = []
  for label in labels:
    s = 0
    for input_label in input_labels[:-1]:
      probs_by_label_by_input[input_label][label] = sympy.Symbol("prob_%s_in_%s" % (label, input_label))
      s += probs_by_label_by_input[input_label][label]
      prob_vars.append(probs_by_label_by_input[input_label][label])
    probs_by_label_by_input[input_labels[-1]][label] = 1 - s
  print(prob_vars)
  theta_a, theta_b = prob_vars

  initial_vec = sympy.Matrix([[1, 0, 0]])
  final_vec = sympy.Matrix([[1] if state in fsa.final_states else [0] for state in states])
  v = initial_vec
  trans_mat_product = None
  for i in range(len(label_seq_template)):
    probs_by_label = probs_by_label_by_input[label_seq_template[i]]
    trans_mat = sympy.Matrix([
      [fsa.get_deterministic_source_to_target_prob(
        source_state=src_state, target_state=tgt_state, probs_by_label=probs_by_label)
       for tgt_state in states]
      for src_state in states])
    trans_mat = sympy.Pow(trans_mat, n)
    v *= trans_mat
    if trans_mat_product is None:
      trans_mat_product = trans_mat
    else:
      trans_mat_product *= trans_mat
  res = v * final_vec
  assert res.shape == (1, 1)
  sum_ = res[0, 0]

  print("sum:", sum_)
  # sum_ = sum_.simplify()  -- makes it actually harder?
  for _  in range(0):
    sum_ = sum_.simplify()
    print("sum simplified:", sum_)

  print("max:", sympy.maximum(sum_, theta_a, sympy.Interval(0, 1)))

  syms = (theta_a, theta_b)
  syms = (theta_b,)
  sum_diff = sum_.diff(*syms)
  print("diff:", sum_diff)
  # sum_diff = sum_diff.simplify()  # -- also makes it harder?
  for _ in range(0):
    sum_diff = sum_diff.simplify()
    print("diff simplified:", sum_diff)
  opts = sympy.solve(sum_diff, *syms)
  print("num opts:", len(opts))
  print("opts:", opts)


def test_tf_grad_log_sm():
  import tensorflow as tf
  print("TF version:", tf.__version__)
  with tf.Session() as session:
    x = tf.constant([[0., 0., 0.], [0., 0., 0.]])
    y = tf.nn.log_softmax(x)
    scores = [0., float("-inf"), float("-inf")]

    def combine(s_, y_):
      return tf.where(tf.is_finite(s_), s_ + y_, s_)  # not using this results in inf/nan grads
      # return s_ + y_

    for t in range(2):
      ys = y[t]
      scores = [
        combine(scores[0], ys[0]),
        tf.reduce_logsumexp([combine(scores[0], ys[1]), combine(scores[1], ys[1])]),
        tf.reduce_logsumexp([combine(scores[1], ys[2]), combine(scores[2], ys[2])])
      ]
    z = scores[-1]
    # z = tf.reduce_logsumexp([
    #   tf.reduce_logsumexp([
    #     y[0][1],
    #     float("-inf") + y[0][1]])
    #   + y[1][2],
    #   tf.reduce_logsumexp([
    #     float("-inf") + y[0][2],
    #     float("-inf") + y[0][2]])
    #   + y[1][2]])
    dx, = tf.gradients(z, x)
    print(session.run(dx))


def test_ctc():
  import tensorflow as tf
  print("TF version:", tf.__version__)
  fsa = get_std_fsa_3label_blank()
  num_batch = 1
  num_frames = 100
  num_labels = 4  # including blank
  with tf.Session() as session:
    labels = tf.SparseTensor(indices=[[0, 0], [0, 1], [0, 2]], values=[0, 1, 2], dense_shape=[num_batch, 3])  # 0-1-2
    logits = tf.random_normal((num_frames, num_batch, num_labels), seed=42)
    logits_normalized = tf.nn.log_softmax(logits)
    score1 = tf.nn.ctc_loss(labels=labels, inputs=logits, sequence_length=[num_frames] * num_batch)
    score2 = -fsa.tf_get_full_sum(logits=logits_normalized)  # -log space
    dscore1, = tf.gradients(score1, logits)
    dscore2, = tf.gradients(score2, logits)
    res = session.run({"score1": score1, "score2": score2, "dscore1": dscore1, "dscore2": dscore2})
    pprint(res)
    numpy.testing.assert_allclose(res["score1"], res["score2"], rtol=1e-5)
    numpy.testing.assert_allclose(res["dscore1"], res["dscore2"], atol=1e-4)


def main():
  if len(sys.argv) >= 2:
    globals()[sys.argv[1]]()  # eg test_ctc()
    return

  label_seq_template = Label1StrTemplate
  fsa = get_std_fsa_1label()
  print("fsa:", fsa)
  assert fsa.is_deterministic_by_label()
  assert match(fsa=fsa, input_seq=label_seq_template)
  num_frames = 16
  print(
    "T=%i, labels=%r count of all paths:" % (num_frames, len(fsa.get_labels())),
    len(fsa.get_labels()) ** num_frames)
  # for path in iterate_all_paths(fsa, num_frames=num_frames):
  #  print("".join([arc.label for arc in path]))
  test_count_all_paths(fsa=fsa, num_frames=num_frames)
  test_count_all_paths_with_label_in_frame(fsa=fsa, num_frames=num_frames, frame_idx=0, label=Label1)
  test_count_all_paths_with_label_in_frame(fsa=fsa, num_frames=num_frames, frame_idx=num_frames // 2, label=Label1)
  for t in range(num_frames):
    test_count_all_paths_with_label_in_frame(fsa=fsa, num_frames=num_frames, frame_idx=t, label=BlankLabel)
  count_paths_with_label(fsa=fsa, num_frames=num_frames * 2, label=BlankLabel)

  print("Relevant for bias model:")
  count_all_paths_with_label_avg(fsa=fsa, label=BlankLabel, num_frames=num_frames)
  count_all_paths_with_label_avg(fsa=fsa, label=Label1)

  bias_model(fsa=fsa, num_frames=5)
  bias_model_1label(num_frames=5)

  print("Relevant for FFNN / generative model:")
  count_all_paths_with_label_seq(fsa=fsa, label_seq_template=label_seq_template)
  # full_sum(fsa=fsa, label_seq_template=label_seq_template)  # -- incomplete


if __name__ == '__main__':
  import better_exchook
  better_exchook.install()
  main()
