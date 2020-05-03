from sisyphus import tk

def train_f2l_config(config_file, name, parameter_dict=None):
  from recipe.returnn import RETURNNTrainingFromFile
  f2l_train = RETURNNTrainingFromFile(config_file, parameter_dict=parameter_dict, mem_rqmt=16)
  f2l_train.add_alias("f2l_training/" + name)

  # f2l_train.rqmt['qsub_args'] = '-l qname=%s' % "*080*"

  f2l_train.rqmt['time'] = 96
  f2l_train.rqmt['cpu'] = 8
  tk.register_output("f2l_training/" + name + "_model", f2l_train.model_dir)
  tk.register_output("f2l_training/" + name + "_training-scores", f2l_train.learning_rates)
  return f2l_train


def convert_with_f2l(config_file, name, model_dir, epoch, features):
  from recipe.returnn.forward import RETURNNForwardFromFile

  parameter_dict = {'ext_forward': True,
                    'ext_model': model_dir,
                    'ext_load_epoch': epoch,
                    'ext_eval_features': features,
                    }

  f2l_apply = RETURNNForwardFromFile(config_file,
                                     parameter_dict=parameter_dict,
                                     hdf_outputs=['linear_features'],
                                     mem_rqmt=8)

  f2l_apply.add_alias("f2l_forward/" + name)
  f2l_apply.rqmt['qsub_args'] = '-l qname=%s' % "*080*" + ' -l h_fsize=400G'
  f2l_apply.rqmt['time'] = 24
  f2l_apply.rqmt['mem'] = 16

  return f2l_apply.outputs['linear_features'], f2l_apply

def griffin_lim_ogg(linear_hdf, name, iterations=1):

  from recipe.tts.toolchain import GriffinLim
  gl_job = GriffinLim(linear_hdf,
                      iterations=iterations,
                      sample_rate=16000,
                      window_shift=0.0125,
                      window_size=0.05,
                      preemphasis=0.97,
                      )
  gl_job.add_alias("gl_conversion/" + name)
  tk.register_output("generated_audio/" + name +"_audio", gl_job.out_folder)
  return gl_job.out_corpus, gl_job
