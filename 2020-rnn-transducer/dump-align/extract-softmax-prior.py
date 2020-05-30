#!/usr/bin/env python3


import os
import sys
import argparse
import better_exchook
from returnn.rnn import init
from returnn.Config import get_global_config
from returnn.TFEngine import get_global_engine
from returnn.Dataset import init_dataset


my_dir = os.path.dirname(os.path.abspath(__file__))
setup_base_dir = os.path.dirname(my_dir)
data_dir = "%s/data" % my_dir


def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("setup")
  arg_parser.add_argument("--softmax-layer", default="ctc_out")
  args = arg_parser.parse_args()

  config_filename = "%s/config-train/%s.config" % (setup_base_dir, args.setup)
  setup_dir = "%s/data-train/%s" % (setup_base_dir, args.setup)
  assert os.path.exists(config_filename) and os.path.isdir(setup_dir)

  os.chdir(setup_dir)
  init(
    config_filename=config_filename,
    extra_greeting="extract softmax prior",
    config_updates={
      "need_data": False,  # do not load it automatically
    })
  config = get_global_config()

  datasets_dict = {"train": config.typed_dict["train"]}
  for dataset_name, dataset_dict in datasets_dict.items():
    assert isinstance(dataset_dict, dict)
    assert dataset_dict["class"] == "ExternSprintDataset"
    assert "partition_epoch" in dataset_dict and "estimated_num_seqs" in dataset_dict
    dataset_dict["estimated_num_seqs"] *= dataset_dict["partition_epoch"]
    dataset_dict["partition_epoch"] = 1
    sprint_args = dataset_dict["sprintConfigStr"]
    assert isinstance(sprint_args, list)
    shuffle_chunk_size_opt = [
      arg for arg in sprint_args
      if isinstance(arg, str) and "segment-order-sort-by-time-length-chunk-size=" in arg]
    assert len(shuffle_chunk_size_opt) == 1
    sprint_args.remove(shuffle_chunk_size_opt[0])
    dataset_dict["name"] = dataset_name

  engine = get_global_engine()
  engine.init_network_from_config()
  print("Initialized network, epoch:", engine.epoch)

  for dataset_name, dataset_dict in datasets_dict.items():
    print("Load data", dataset_name, "...")
    dataset = init_dataset(dataset_dict)
    print(dataset)
    out_filename_parts = [args.setup, "epoch-%i" % engine.epoch, "smprior", "txt"]
    output_filename = "%s/%s" % (data_dir, ".".join(out_filename_parts))
    print("Store prior as:", output_filename)
    assert not os.path.exists(output_filename)

    config.set("forward_output_layer", args.softmax_layer)
    config.set("output_file", output_filename)
    config.set("max_seq_length", sys.maxsize)
    engine.compute_priors(dataset=dataset, config=config)
    assert os.path.exists(output_filename)

  print("Finished.")


if __name__ == '__main__':
  better_exchook.install()
  main()
