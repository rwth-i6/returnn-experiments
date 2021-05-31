#!/usr/bin/env python3

import better_exchook
import matplotlib.pyplot as plt
from pprint import pprint


better_exchook.install()

# via: ./simple_model.py run_ratio_t_n
data = \
{1: ([0.20000004768371582,
      0.20000001788139343,
      0.20000001788139343,
      0.4000000059604645],
     [0.33082541823387146,
      0.26116153597831726,
      0.40800315141677856,
      9.883728125714697e-06],
     169),
 2: ([0.1714288592338562,
      0.1714285910129547,
      0.1714286059141159,
      0.4857146143913269],
     [0.30248337984085083,
      0.2955132722854614,
      0.3454301953315735,
      0.056573133915662766],
     289),
 3: ([0.16190464794635773,
      0.16190479695796967,
      0.16190491616725922,
      0.5142859220504761],
     [0.1276986300945282,
      0.20203253626823425,
      0.0701727643609047,
      0.6000960469245911],
     308),
 4: ([0.15714283287525177,
      0.15714266896247864,
      0.15714314579963684,
      0.5285714864730835],
     [0.08661448210477829,
      0.08519323170185089,
      0.05187408998608589,
      0.7763181924819946],
     343),
 5: ([0.1542857140302658,
      0.1542852818965912,
      0.15428604185581207,
      0.5371428728103638],
     [0.09774811565876007,
      0.06279565393924713,
      0.1766882836818695,
      0.6627679467201233],
     395),
 6: ([0.15238071978092194,
      0.1523803323507309,
      0.15238158404827118,
      0.542856752872467],
     [0.09057383239269257,
      0.05933886393904686,
      0.16666169464588165,
      0.6834256052970886],
     440),
 7: ([0.15102015435695648,
      0.15101996064186096,
      0.15102119743824005,
      0.5469379425048828],
     [0.07278444617986679,
      0.056146856397390366,
      0.11365215480327606,
      0.7574165463447571],
     591),
 8: ([0.15000019967556,
      0.15000024437904358,
      0.15000078082084656,
      0.5500000715255737],
     [0.07418833673000336,
      0.06262041628360748,
      0.047744251787662506,
      0.8154470324516296],
     600),
 9: ([0.14920666813850403,
      0.1492064893245697,
      0.1492069512605667,
      0.5523802638053894],
     [0.0689312294125557,
      0.06439986079931259,
      0.04494192451238632,
      0.8217269778251648],
     698)}


#pprint(plt.rcParams)
plt.rcParams["figure.figsize"] = (6, 3)
#plt.rcParams["figure.figsize"] = (6, 1.5)
plt.rcParams["figure.frameon"] = True
plt.rcParams["figure.subplot.top"] = 0.93
plt.rcParams["figure.subplot.bottom"] = 0.15
plt.rcParams["figure.subplot.left"] = 0.1
plt.rcParams["figure.subplot.right"] = 0.9

xs = list(sorted(data))
ys = [data[x] for x in xs]
xs = [x * 10 for x in xs]
ys1 = [a[-1] for (a, b, l) in ys]
ys2 = [b[-1] for (a, b, l) in ys]
ys3 = [l for (a, b, l) in ys]

fig, ax = plt.subplots()

ax.plot(xs, ys1, label="Uniform q(B)", color="red")
ax.plot(xs, ys2, label="Trained q(B)", color="blue")
ax.set_xlabel("T")
ax.set_ylabel("avg q(B)")
#ax.legend()

secaxy = ax.twinx()
secaxy.set_ylabel("i")
secaxy.plot(xs, ys3, label="Training Convergence Time", linestyle="--", color="blue")
#secaxy.legend()

lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = secaxy.get_legend_handles_labels()
secaxy.legend(lines + lines2, labels + labels2, loc="lower right")

#plt.title("Average q(B|t) for different T")
#plt.legend()

#plt.show()
plt.savefig("../figures/ctc2020/ratio-t-n.pdf")
