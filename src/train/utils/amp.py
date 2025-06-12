from torch.cuda.amp import autocast, GradScaler

def get_amp_context(config, device):
    use_amp = bool(config["training"].get("use_amp", True)) and (device.type == "cuda")
    return autocast(enabled=use_amp)

def get_scaler(config, device):
    use_amp = bool(config["training"].get("use_amp", True)) and (device.type == "cuda")
    return GradScaler(enabled=use_amp)
