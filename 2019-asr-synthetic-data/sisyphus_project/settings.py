def engine():
    from sisyphus.localengine import LocalEngine
    return LocalEngine(cpus=4, gpus=1)

JOB_USE_TAGS_IN_PATH = False
JOB_AUTO_CLEANUP = False

START_KERNEL = False
SHOW_JOB_TARGETS=False

PRINT_ERROR=True

