def resume_training_state(ckpt_manager, model, optimizer, scheduler, config, device, logger):
    start_epoch = 0
    start_global_step = 0
    resume_path = config.get("resume_checkpoint", "")
    auto_resume = config.get("monitoring", {}).get("resume", {}).get("auto_resume", False)

    if auto_resume:
        if resume_path:
            try:
                _, _, _, loaded_epoch, loaded_step = ckpt_manager.load_checkpoint(
                    filepath=resume_path,
                    model=model,
                    optimizer=None,
                    scheduler=None,
                    map_location=device,
                )
                start_epoch = loaded_epoch
                start_global_step = loaded_step or 0
                logger.info(f"🔄 Resumed model weights from: {resume_path} (epoch {loaded_epoch})")
            except Exception as e:
                logger.error(f"❌ Failed to load specified checkpoint {resume_path}: {e}")
                logger.info("ℹ️ Falling back to latest available checkpoint…")
                try:
                    _, _, _, loaded_epoch, loaded_step = ckpt_manager.load_latest(
                        model=model,
                        optimizer=None,
                        scheduler=None,
                        map_location=device,
                    )
                    start_epoch = loaded_epoch
                    start_global_step = loaded_step or 0
                    logger.info(f"🔄 Fallback resume from epoch {loaded_epoch}")
                except Exception as e2:
                    logger.info(f"ℹ️ No valid checkpoint found. Starting from scratch. ({e2})")
        else:
            try:
                _, _, _, loaded_epoch, loaded_step = ckpt_manager.load_latest(
                    model=model,
                    optimizer=None,
                    scheduler=None,
                    map_location=device,
                )
                start_epoch = loaded_epoch
                start_global_step = loaded_step or 0
                logger.info(f"🔄 Auto‐resumed from epoch {loaded_epoch}")
            except Exception as e:
                logger.info(f"ℹ️ No checkpoint to resume. Starting from scratch. ({e})")
    else:
        logger.info("ℹ️ Auto‐resume disabled; training from epoch 0.")
        start_epoch = 0
        start_global_step = 0

    # Now re‐create optimizer/scheduler state AFTER setting model weights
    # (We must re‐initialize optimizer & scheduler *after* we know model weights.)
    if start_epoch > 0:
        try:
            _, optimizer, scheduler, loaded_epoch, loaded_step = ckpt_manager.load_checkpoint(
                filepath=resume_path if resume_path else None,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=device,
            )
            start_epoch = loaded_epoch
            start_global_step = loaded_step or start_global_step
            logger.info(f"✅ Restored optimizer & scheduler state at epoch {loaded_epoch}")
        except Exception:
            logger.warning("⚠️ Could not restore optimizer/scheduler from checkpoint; using fresh optimizer.")
    return start_epoch, start_global_step
