import sys
import socket

def engine():
    from sisyphus.engine import EngineSelector
    from sisyphus.localengine import LocalEngine
    from sisyphus.son_of_grid_engine import SonOfGridEngine
    return EngineSelector(engines={'short': LocalEngine(cpus=4),
                                   'long': SonOfGridEngine(default_rqmt={'cpu' : 1, 'mem' : 2, 'gpu' : 0, 'time' : 1},
#                                                           gateway="cluster-cn-01")},
                                                           )},
                          default_engine='long')


SIS_COMMAND = ['/u/rossenbach/opt/sisyphus/bin/python3.7', sys.argv[0]]


JOB_USE_TAGS_IN_PATH = False

JOB_AUTO_CLEANUP = False
START_KERNEL = False
SHOW_JOB_TARGETS=False
PRINT_ERROR=False

def check_engine_limits(current_rqmt, task):
  """ Check if requested requirements break and hardware limits and reduce them.
  By default ignored, a possible check for limits could look like this::

      current_rqmt['time'] = min(168, current_rqmt.get('time', 2))
      if current_rqmt['time'] > 24:
          current_rqmt['mem'] = min(63, current_rqmt['mem'])
      else:
          current_rqmt['mem'] = min(127, current_rqmt['mem'])
      return current_rqmt

  :param dict[str] current_rqmt: requirements currently requested
  :param sisyphus.task.Task task: task that is handled
  :return: requirements updated to engine limits
  :rtype: dict[str]
  """
  current_rqmt['time'] = min(168, current_rqmt.get('time', 1))
  return current_rqmt