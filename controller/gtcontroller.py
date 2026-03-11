from envs.base_env import BaseEnv
from manipulation.base_manipulation import BaseManipulation
from controller.base_controller import BaseController
from logging import Logger

class GtController(BaseController) :

    def __init__(self, env : BaseEnv, manipulation : BaseManipulation, cfg : dict, logger : Logger):
        super().__init__(env, manipulation, cfg, logger)
        self.collect_grasp = cfg['task']['grasp']
        self.policy = cfg['task']['policy']

    def run(self, eval=False):
        if self.collect_grasp:
            self.manipulation.collect_grasp_data()
        else:
            self.manipulation.collect_manip_data()
