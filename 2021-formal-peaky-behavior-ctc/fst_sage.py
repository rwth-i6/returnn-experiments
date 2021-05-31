
from sage.all import *
# noinspection PyUnresolvedReferences
from sage.calculus.all import var
#import numpy


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
  na = var("na", domain=ZZ)
  nb = var("nb", domain=ZZ)
  theta_a = var("theta_a", domain=RR)
  theta_b = var("theta_b", domain=RR)
  t = var("t", domain=ZZ)
  # Make 2 parts of the sum, one t=0...na, another t=na..T.
  # Should get rid of the min/max cases, simplify it.
  p1 = theta_a ** min_symbolic(t, na)
  p2 = (1 - theta_a) ** max_symbolic(t - na, 0)
  p3 = theta_b ** min_symbolic(na + nb - t, nb)  # exp = min(na - t, 0) + nb
  p4 = (1 - theta_b) ** max_symbolic(na - t, 0)
  sum_ = sum(p1 * p2 * p3 * p4, t, 0, na + nb)

  for _ in range(6):
    sum_ = sum_.simplify()
  print(sum_)

  #sum__ = sum_.substitute(na=10, nb=10)
  #xs = ys = numpy.linspace(0, 1., num=11)
  #values = numpy.zeros((len(xs), len(ys)))
  #for ix, x in enumerate(xs):
  #  for iy, y in enumerate(ys):
  #    value = sum__.substitute(theta_a=x, theta_b=y)
  #    print("theta = (%f, %f) -> sum = %s" % (x, y, value))
      #values[ix, iy] = float(sum__.subs(theta_a, x).subs(theta_b, y).doit())
  #print(values)

  #syms = (theta_a, theta_b)
  #syms = (theta_b,)
  syms = (theta_a,)
  sum_diff = sum_.diff(*syms)
  sum_diff = sum_diff.simplify()
  print("diff:", sum_diff)
  #for _ in range(5):
  #  sum_diff = sum_diff.simplify()
  #  print(sum_diff)
  # sum_diff = sum_diff.simplify()  # -- also makes it harder?
  opts = solve(sum_diff == 0, *syms, domain=RR)
  print("num opts:", len(opts))
  for opt in opts:
    print("opt:", opt)


def main():
  if len(sys.argv) >= 2:
    globals()[sys.argv[1]]()  # eg test_ctc()
    return

  print("Usage: %s <func>" % __file__)
  sys.exit(1)


if __name__ == '__main__':
  #import better_exchook
  #better_exchook.install()
  main()
