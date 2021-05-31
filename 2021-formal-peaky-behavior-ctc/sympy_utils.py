
import typing
import sympy
# noinspection PyProtectedMember
from sympy.solvers.inequalities import _solve_inequality


def range_from_relationals(and_expr: typing.Union[sympy.And, sympy.Rel], gen: sympy.Symbol) -> (
      typing.Optional[sympy.Basic], typing.Optional[sympy.Basic]):
  """
  :return whether there is a solution, optional start range, optional end range
    (including; assume integer; assume simplified)
  """
  if isinstance(and_expr, sympy.Rel):
    args = [and_expr]
  else:
    assert isinstance(and_expr, sympy.And)
    args = and_expr.args
  assert all([isinstance(part, sympy.Rel) and gen in part.free_symbols for part in args])

  rel_ops = [">=", "<=", "=="]
  rhs_by_c = {}

  for part in args:
    assert isinstance(part, sympy.Rel)
    part = _solve_inequality(part, gen)
    assert isinstance(part, sympy.Rel)
    assert part.lhs == gen
    rel_op, rhs = part.rel_op, part.rhs
    assert rel_op in rel_ops
    assert rel_op not in rhs_by_c  # not simplified?
    rhs_by_c[rel_op] = rhs

  if "==" in rhs_by_c:
    assert set(rhs_by_c.keys()) == {"=="}  # only op. not simplified?
    return rhs_by_c["=="], rhs_by_c["=="]
  return rhs_by_c.get(">=", None), rhs_by_c.get("<=", None)


def simplify_and(
      x: sympy.Basic, gen: typing.Optional[sympy.Symbol] = None,
      extra_conditions: typing.Optional[sympy.Basic] = True) -> sympy.Basic:
  """
  Some rules, because SymPy currently does not automatically simplify them...
  """
  assert isinstance(x, sympy.Basic), "type x: %r" % type(x)
  from sympy.solvers.inequalities import reduce_rational_inequalities
  from sympy.core.relational import Relational

  syms = []
  if gen is not None:
    syms.append(gen)

  w1 = sympy.Wild("w1")
  w2 = sympy.Wild("w2")
  for sub_expr in x.find(sympy.Eq(w1, w2)):
    m = sub_expr.match(sympy.Eq(w1, w2))
    ws_ = m[w1], m[w2]
    for w_ in ws_:
      if isinstance(w_, sympy.Symbol) and w_ not in syms:
        syms.append(w_)
  for w_ in x.free_symbols:
    if w_ not in syms:
      syms.append(w_)

  if len(syms) >= 1:
    _c = syms[0]
    if len(syms) >= 2:
      n = syms[1]
    else:
      n = sympy.Wild("n")
  else:
    return x

  x = x.replace(((_c - 2 * n >= -1) & (_c - 2 * n <= -1)), sympy.Eq(_c, 2 * n - 1))  # probably not needed anymore...
  apply_rules = True
  while apply_rules:
    apply_rules = False
    for and_expr in x.find(sympy.And):
      assert isinstance(and_expr, sympy.And)

      and_expr_ = reduce_rational_inequalities([and_expr.args], _c)
      # print(and_expr, "->", and_expr_)
      if and_expr_ != and_expr:
        x = x.replace(and_expr, and_expr_)
        and_expr = and_expr_
        if and_expr == sympy.sympify(False):
          continue
        if isinstance(and_expr, sympy.Rel):
          continue
        assert isinstance(and_expr, sympy.And)

      and_expr_args = list(and_expr.args)
      # for i, part in enumerate(and_expr_args):
      #  and_expr_args[i] = part.simplify()
      if all([isinstance(part, Relational) and _c in part.free_symbols for part in and_expr_args]):
        # No equality, as that should have been resolved above.
        rel_ops = ["==", ">=", "<="]
        if not (_c.is_Integer or _c.assumptions0["integer"]):
          rel_ops.extend(["<", ">"])
        rhs_by_c = {op: [] for op in rel_ops}
        for part in and_expr_args:
          assert isinstance(part, Relational)
          part = _solve_inequality(part, _c)
          assert isinstance(part, Relational)
          assert part.lhs == _c
          rel_op, rhs = part.rel_op, part.rhs
          if _c.is_Integer or _c.assumptions0["integer"]:
            if rel_op == "<":
              rhs = rhs - 1
              rel_op = "<="
            elif rel_op == ">":
              rhs = rhs + 1
              rel_op = ">="
          assert rel_op in rhs_by_c, "x: %r, _c: %r, and expr: %r, part %r" % (x, _c, and_expr, part)
          other_rhs = rhs_by_c[rel_op]
          assert isinstance(other_rhs, list)
          need_to_add = True
          for rhs_ in other_rhs:
            cmp = Relational.ValidRelationOperator[rel_op](rhs, rhs_)
            if simplify_and(sympy.And(sympy.Not(cmp), extra_conditions)) == sympy.sympify(False):  # checks True...
              other_rhs.remove(rhs_)
              break
            elif simplify_and(sympy.And(cmp, extra_conditions)) == sympy.sympify(False):
              need_to_add = False
              break
            # else:
            #  raise NotImplementedError("cannot compare %r in %r; extra cond %r" % (cmp, and_expr, extra_conditions))
          if need_to_add:
            other_rhs.append(rhs)
        if rhs_by_c[">="] and rhs_by_c["<="]:
          all_false = False
          for lhs in rhs_by_c[">="]:
            for rhs in rhs_by_c["<="]:
              if sympy.Lt(lhs, rhs) == sympy.sympify(False):
                all_false = True
              if sympy.Eq(lhs, rhs) == sympy.sympify(True):
                rhs_by_c["=="].append(lhs)
          if all_false:
            x = x.replace(and_expr, False)
            continue
        if rhs_by_c["=="]:
          all_false = False
          while len(rhs_by_c["=="]) >= 2:
            lhs, rhs = rhs_by_c["=="][:2]
            if sympy.Eq(lhs, rhs) == sympy.sympify(False):
              all_false = True
              break
            elif sympy.Eq(lhs, rhs) == sympy.sympify(True):
              rhs_by_c["=="].pop(1)
            else:
              raise NotImplementedError("cannot cmp %r == %r. rhs_by_c %r" % (lhs, rhs, rhs_by_c))
          if all_false:
            x = x.replace(and_expr, False)
            continue
          new_parts = [sympy.Eq(_c, rhs_by_c["=="][0])]
          for op in rel_ops:
            for part in rhs_by_c[op]:
              new_parts.append(Relational.ValidRelationOperator[op](rhs_by_c["=="][0], part).simplify())
        else:  # no "=="
          new_parts = []
          for op in rel_ops:
            for part in rhs_by_c[op]:
              new_parts.append(Relational.ValidRelationOperator[op](_c, part))
          assert new_parts
        and_expr_ = sympy.And(*new_parts)
        # print(and_expr, "--->", and_expr_)
        x = x.replace(and_expr, and_expr_)
        and_expr = and_expr_

      # Probably all the remaining hard-coded rules are not needed anymore with the more generic code above...
      if sympy.Eq(_c, 2 * n) in and_expr.args:
        if (_c - 2 * n <= -1) in and_expr.args:
          x = x.replace(and_expr, False)
          continue
        if sympy.Eq(_c - 2 * n, -1) in and_expr.args:
          x = x.replace(and_expr, False)
          continue
        if (_c - n <= -1) in and_expr.args:
          x = x.replace(and_expr, False)
          continue
      if (_c >= n) in and_expr.args and (_c - n <= -1) in and_expr.args:
        x = x.replace(and_expr, False)
        continue
      if sympy.Eq(_c - 2 * n, -1) in and_expr.args:  # assume n>=1
        if (_c >= n) in and_expr.args:
          x = x.replace(and_expr, sympy.And(*[arg for arg in and_expr.args if arg != (_c >= n)]))
          apply_rules = True
          break
        if (_c - n >= -1) in and_expr.args:
          x = x.replace(and_expr, sympy.And(*[arg for arg in and_expr.args if arg != (_c - n >= -1)]))
          apply_rules = True
          break
      if (_c >= n) in and_expr.args:
        if (_c - n >= -1) in and_expr.args:
          x = x.replace(and_expr, sympy.And(*[arg for arg in and_expr.args if arg != (_c - n >= -1)]))
          apply_rules = True
          break
      if (_c - n >= -1) in and_expr.args and (_c - n <= -1) in and_expr.args:
        args = list(and_expr.args)
        args.remove((_c - n >= -1))
        args.remove((_c - n <= -1))
        args.append(sympy.Eq(_c - n, -1))
        if (_c - 2 * n <= -1) in args:
          args.remove((_c - 2 * n <= -1))
        x = x.replace(and_expr, sympy.And(*args))
        apply_rules = True
        break
  return x


def sum_over_piecewise(
      expr: sympy.Piecewise,
      sum_var: sympy.Symbol, sum_start: typing.Union[sympy.Basic, int], sum_end: sympy.Basic,
      extra_condition: sympy.Basic = True) -> sympy.Expr:
  """
  :return: equivalent to Sum(expr, (sum_var, sum_start, sum_end)), but we try to remove the piecewise.
    We assume that the piecewise conditions also depend on sum_var.
  """
  assert sum_var.is_Integer or sum_var.assumptions0["integer"]
  assert isinstance(expr, sympy.Piecewise)
  res = sympy.sympify(0)
  cond_start = sympy.Ge(sum_var, sum_start)
  cond_end = sympy.Le(sum_var, sum_end)
  prev_ranges = [(None, sum_start - 1), (sum_end + 1, None)]

  def check(cond__):
    false_cond = simplify_and(sympy.And(sympy.Not(cond__), extra_condition))
    if false_cond == sympy.sympify(False):
      return True
    true_cond = simplify_and(sympy.And(cond__, extra_condition))
    if true_cond == sympy.sympify(False):
      return False
    return None

  for value, cond in expr.args:
    j = 0
    while j < len(prev_ranges) - 1:
      cond_r_start = sympy.Ge(sum_var, prev_ranges[j][1] + 1)
      cond_r_end = sympy.Le(sum_var, prev_ranges[j + 1][0] - 1)
      cond = sympy.And(cond, cond_start, cond_end, cond_r_start, cond_r_end)
      cond_ = simplify_and(cond, sum_var, extra_conditions=extra_condition)
      # print(cond, "->", cond_)
      if cond_ == sympy.sympify(False):
        j += 1
        continue

      if isinstance(cond_, sympy.And):
        if any([sum_var not in part.free_symbols for part in cond_.args]):
          new_extra_conditions = [part for part in cond_.args if sum_var not in part.free_symbols]
          new_extra_condition = sympy.And(*new_extra_conditions)
          if check(new_extra_condition) is False:
            j += 1
            continue
          assert check(new_extra_condition)
          cond_ = sympy.And(*[part for part in cond_.args if sum_var in part.free_symbols])

      r = range_from_relationals(cond_, sum_var)
      if r[0] is None:  # e.g. if cond_start == True because sum_var is >= 0 always
        r = (sum_start, r[1])
      if sympy.Eq(r[0], r[1]) == sympy.sympify(True):
        res += value.subs(sum_var, r[0])
      else:
        res += sympy.Sum(value, (sum_var, r[0], r[1])).doit()

      for i in range(1, len(prev_ranges) + 1):
        assert i < len(prev_ranges), "where to insert %r?" % (r,)  # should not get past here
        assert check(sympy.Gt(r[0], prev_ranges[i - 1][1]))
        if check(sympy.Eq(r[0] - 1, prev_ranges[i - 1][1])):
          prev_ranges[i - 1] = (prev_ranges[i - 1][0], r[1])
          break
        if check(sympy.Lt(r[0], prev_ranges[i][0])) or check(sympy.Lt(r[1], prev_ranges[i][0])):
          if check(sympy.Eq(r[1] + 1, prev_ranges[i][0])):
            prev_ranges[i] = (r[0], prev_ranges[i][1])
          else:
            prev_ranges.insert(i, r)
          break

      # print("prev ranges:", prev_ranges)
      j = 0  # start over...

    if len(prev_ranges) == 2 and sympy.Eq(prev_ranges[0][1], prev_ranges[1][0]) == sympy.sympify(True):
      # We have the full range covered.
      break

  return res.simplify()
  # fallback
  # return sympy.Sum(expr, (sum_var, sum_start, sum_end))


def binomial_expansion(a, b, exp):
  """
  Applies the binomial expansion (https://en.wikipedia.org/wiki/Binomial_theorem).

  :param sympy.Expr|int a:
  :param sympy.Expr|int b:
  :param sympy.Expr|int exp: assumes to be a nonnegative integer
  :rtype sympy.Expr
  """
  i = sympy.Symbol("i", integer=True, nonnegative=True)
  x = sympy.binomial(exp, i) * sympy.Pow(a, exp - i) * sympy.Pow(b, i)
  return sympy.Sum(x, (i, 0, exp))


def polynomial_exp(a, b, exp, expand=True, flip=True):
  """
  :param sympy.Expr|int a:
  :param sympy.Expr|int b:
  :param sympy.Expr|int exp: assumes to be a nonnegative integer
  :param bool expand:
  :param bool flip:
  :rtype sympy.Expr
  """
  if expand:
    a = sympy.sympify(a)
    b = sympy.sympify(b)
    has_flipped = False
    if flip:
      try:
        should_flip = bool((b <= 0 and a == 1) or (a <= 0 and b == 1))
      except TypeError:
        should_flip = False
      if should_flip:
        a, b = -a, -b
        has_flipped = True
    res = binomial_expansion(a, b, exp)
    if has_flipped:
      res *= sympy.Pow(-1, exp)
    return res
  return sympy.Pow(a + b, exp)
