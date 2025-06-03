# tools/monitoring/stream_log.py

import argparse, os, sys, time, yaml
import socket
is_cluster = lambda hostname: "login" in hostname or "node" in hostname or os.environ.get("IS_CLUSTER") == "1"

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_log_path(config, experiment_id, run_id):
    # Determine the log base path from config or fallback
    base_dir = config.get("logging", {}).get("base_dir")
    if base_dir is None:
        base_dir = "/gluster/.../logs" if is_cluster(socket.gethostname()) else "logs"

    return os.path.join(base_dir, f"experiment_{experiment_id}", f"run_{run_id}", "training.log")

def stream_log(log_file, levels=None):
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        sys.exit(1)

    print(f"📡 Streaming {log_file} (filtered={levels}) — Ctrl+C to stop.")
    with open(log_file, "r") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if line:
                if levels:
                    if any(level in line for level in levels):
                        print(line.strip())
                else:
                    print(line.strip())
            else:
                time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/config.yaml", help="Path to config file")
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--levels", nargs="*", default=[], help="Filter by log levels (e.g., INFO ERROR WARNING)")

    args = parser.parse_args()
    config = load_config(args.config)
    levels = [lvl.upper() for lvl in args.levels] if args.levels else None

    log_file = get_log_path(config, args.experiment_id, args.run_id)
    stream_log(log_file, levels)
