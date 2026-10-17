"""Retinal vessel segmentation utilities.

Import submodules directly; the package init stays import-light so the
serving app can use utils.preprocessing without pulling in torch:

    from utils.preprocessing import preprocess_fundus_image   # runtime
    from utils.architecture import VesselUNet                 # training/export
"""
