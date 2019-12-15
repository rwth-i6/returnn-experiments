from sisyphus import setup_path

Path = setup_path(__package__)

SUBWORD_NMT_DIR = Path('../src/subword-nmt/', hash_overwrite="SUBWORD_NMT_DIR")
RETURNN_PYTHON_EXE = Path('../bin/returnn_launcher.sh', hash_overwrite='RETURNN_PYTHON_EXE')
RETURNN_SRC_ROOT = Path('../src/returnn', hash_overwrite='RETURNN_SRC_ROOT')
