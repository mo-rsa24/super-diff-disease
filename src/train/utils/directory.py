import os

def prepare_output_dirs(save_path):
    vis_dir = os.path.join(save_path, "samples")
    feature_dir = os.path.join(save_path, "features")
    proj_dir = os.path.join(save_path, "projections")
    metric_dir = os.path.join(save_path, "metrics")
    for sub in (vis_dir, feature_dir, proj_dir, metric_dir):
        os.makedirs(sub, exist_ok=True)
    return {
        "vis_dir": vis_dir,
        "feature_dir": feature_dir,
        "proj_dir": proj_dir,
        "metric_dir": metric_dir
    }
