from ema_pytorch import EMA

def init_ema(model, config, device):
    ema_beta = float(config["training"].get("ema_beta", 0.995))
    return EMA(model, beta=ema_beta).to(device)

def update_ema(ema):
    ema.update()
