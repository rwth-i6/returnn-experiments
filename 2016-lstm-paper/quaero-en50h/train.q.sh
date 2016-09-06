#!qint.py

concurrent part:
    trainset 1

concurrent part

# If you want to change the amount of iterations:
# * Above, rename "it-<number>" such that <number> = number of iterations.
# * In recog.pass1.init, change ITER = number of iterations.

parallel train(qsub="-notify -hard -l h_vmem=15G -l h_rt=80:00:00 -l gpu=1"):
	/u/zeyer/dotfiles/system-tools/helpers/cgroup-mem-log-rss-max.py &
	/u/zeyer/dotfiles/system-tools/helpers/cgroup-mem-limit-watcher.py &

	source tools/tools.sh && source tools/utilities.training.sh
	source theano-cuda-activate.sh
	source settings.sh

	ensure config-train/$model.config

	cacheArchive "dependencies/alignment" "alignment-cache-path"
	cacheArchive "dependencies/energy.bundle" "energy-cache-path"
	cacheArchive "dependencies/features.bundle" "mfcc-cache-path"
	export SPRINT_OPTIONS="$CACHE_OPTION"

	python crnn/rnn.py config-train/$model.config || exit 1

