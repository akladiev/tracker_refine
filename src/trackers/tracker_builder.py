from src.config import cfg
from src.trackers.siamrpn_tracker import SiamRPNTracker
from src.trackers.siammask_tracker import SiamMaskTracker

TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamMaskTracker': SiamMaskTracker,
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
