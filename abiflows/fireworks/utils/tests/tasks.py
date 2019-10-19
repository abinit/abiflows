# coding: utf-8
from fireworks import explicit_serialize, FiretaskBase
from abiflows.fireworks.utils.fw_utils import get_lp_and_fw_id_from_task


@explicit_serialize
class LpTask(FiretaskBase):
    def run_task(self, fw_spec):
        lp, fw_id = get_lp_and_fw_id_from_task(self, fw_spec=fw_spec)
