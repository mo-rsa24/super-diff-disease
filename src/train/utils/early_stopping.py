def check_early_stopping(current_val_mse, best_val_mse, patience_counter, early_stop_patience):
    """
    Returns (new_best_mse, new_patience_counter, stop_flag)
    """
    if current_val_mse < best_val_mse:
        return current_val_mse, 0, False
    patience_counter += 1
    if patience_counter >= early_stop_patience:
        return best_val_mse, patience_counter, True
    return best_val_mse, patience_counter, False
