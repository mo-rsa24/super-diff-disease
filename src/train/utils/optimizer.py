from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

def build_optimizer(model, config):
    opt_cfg = config["training"]
    opt_name = opt_cfg.get("optimizer", "adam").lower()
    lr = float(opt_cfg.get("learning_rate", 1e-4))
    beta1 = float(opt_cfg.get("beta1", 0.9))
    beta2 = float(opt_cfg.get("beta2", 0.999))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    if opt_name == "adamw":
        return AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    return Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

def build_scheduler(optimizer, config, num_epochs):
    sched_cfg = config["training"].get("scheduler", {})
    if not sched_cfg:
        return None
    stype = sched_cfg.get("type", "step").lower()
    if stype == "step":
        step_size = int(sched_cfg.get("step_size", 10))
        gamma = float(sched_cfg.get("gamma", 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif stype == "cosine":
        t_max = int(sched_cfg.get("t_max", num_epochs))
        eta_min = float(sched_cfg.get("eta_min", 0.0))
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    else:
        raise ValueError(f"Unsupported scheduler type: {stype}")
