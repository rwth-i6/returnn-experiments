from __future__ import annotations
from typing import Dict, Any
from .pipeline import Context, Topology


def update_net_for_alignment_dumping(net: Dict[str, Any],
                                     extend: bool,
                                     ctx: Context,
                                     stage_num: int,
                                     align_dir: str,
                                     alignment_topology: Topology,
                                     decoder: str = "output",
                                     output_log_prob: str = "output_log_prob_wb",
                                     encoder: str = "encoder") -> Dict[str, Any]:
  """
  This function extends the `decoder` "unit" with logic to create and dump max-alginments in .hdf files.
  During this step, 1 subepoch passes and no training takes place.

  Args:
      net (Dict[str, Any]): network that provides the `encoder` and `decoder` keys.
      extend (bool): True if the already existing .hdf alignments should be extended.
      ctx (Context): holds information such as the label topology, target and output path.
      stage_num (int): stage nr, used in the name of the .hdf file we dump in.
      align_dir (str): Path to the folder the .hdf files should be saved.
      decoder (str): Assumed the decoder is a recurent network and provides `output_log_prob` in its 'unit'.
      output_log_prob (str): log distribution over the whole vocab inkl blank.
      encoder (str): Output of the encoder

  Returns:
      Dict[str, Any]: Updated network
  """

  target = ctx.target

  subnet = net[decoder]["unit"]
  subnet[output_log_prob]["is_output_layer"] = True
  subnet["max_alignment"] = {
    "class": "eval",
    "from": [output_log_prob, f"base:data:{target.key}", f"base:{encoder}"],
    "eval": alignment_topology.alignment,
    "out_type": alignment_topology.alignment_out_type,
    "is_output_layer": True
  }

  subnet["_align_dump"] = {
    "class": "hdf_dump",
    "from": "max_alignment",
    "is_output_layer": True,
    "dump_per_run": True,
    "extend_existing_file": extend,  # TODO: extend only after the first time
    # dataset_name comes from **opts of the lambda in filename
    "filename":
      (lambda **opts:
        "{align_dir}/align.stage_{stage_num}_{dataset_name}.hdf".format(align_dir=align_dir,
                                                                        stage_num=stage_num,
                                                                        **opts))
  }
  net["#trainable"] = False  # disable training
  net["#finish_all_data"] = True
  return net
