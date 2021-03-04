#!/usr/bin/env python3

import gzip
import timeit
import ast
import os
import time
import better_exchook

better_exchook.install()
txt_fn_gz = os.path.expanduser("train-other-500.txt.gz")

t = time.time()
txt = gzip.open(txt_fn_gz, "rb").read().decode("utf8")
print("Gunzip + read time:", time.time() - t)

print(timeit.timeit(lambda: eval(txt), number=1))

print(timeit.timeit(lambda: ast.literal_eval(txt), number=1))
