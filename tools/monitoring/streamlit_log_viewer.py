# tools/monitoring/streamlit_log_viewer.py
# tools/monitoring/streamlit_log_viewer.py

import streamlit as st
import yaml
import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt

import socket



# --- CLI ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="src/config/config.yaml", help="Path to config file")
parser.add_argument("--experiment_id", type=str, default="1")
parser.add_argument("--run_id", type=str, default="2")
parser.add_argument("--log_type", type=str, choices=["training", "error_only", "output", "structured"], default="training")
parser.add_argument("--follow", action="store_true")
args, _ = parser.parse_known_args()

# --- LOAD CONFIG ---
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

is_cluster = lambda hostname: "login" in hostname or "node" in hostname or os.environ.get("IS_CLUSTER") == "1"
log_base = config["paths"]["cluster_base"] if is_cluster(socket.gethostname()) else config["paths"]["local_base"]
experiment_id = f"experiment_{args.experiment_id}"
run_id = f"run_{args.run_id}"
log_dir = os.path.join(log_base, "logs", experiment_id, run_id)



log_file_map = {
    "training": "training.log",
    "error_only": "error_only.log",
    "output": "output.log",
    "structured": "training.jsonl"
}
log_filename = log_file_map[args.log_type]
log_path = os.path.normpath(os.path.join(log_dir,log_filename))

# --- FOLLOW MODE ---
if args.follow:
    st.experimental_set_query_params(refresh=str(int(st.experimental_get_query_params().get("refresh", [0])[0]) + 1))
    st.experimental_rerun()

# --- STRUCTURED JSON LOG VIEW ---
if args.log_type == "structured":
    st.set_page_config(layout="wide")
    st.title("📈 Structured Log Viewer + Metrics")
    st.caption(f"🧪 Experiment: `{experiment_id}` | Run: `{run_id}` | File: `{log_filename}`")

    if not os.path.exists(log_path):
        st.error(f"❌ Log file not found: {log_path}")
    else:
        rows = []
        with open(log_path, "r") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not rows:
            st.warning("No valid log entries found.")
        else:
            df = pd.DataFrame(rows)

            # --- Sidebar Level Summary ---
            st.sidebar.title("🔍 Log Summary")
            if "level" in df.columns:
                level_counts = df["level"].value_counts().to_dict()
                for lvl, count in level_counts.items():
                    st.sidebar.write(f"**{lvl}**: {count}")
                selected_level = st.sidebar.selectbox("Filter by level", ["ALL"] + list(level_counts.keys()))
                if selected_level != "ALL":
                    df = df[df["level"] == selected_level]

            # --- Metrics Section ---
            numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
            metric_cols = [col for col in numeric_cols if col.lower() not in ["epoch"]]

            if metric_cols:
                selected_metric = st.sidebar.selectbox("📊 Plot metric", metric_cols)
                x_axis = df["epoch"] if "epoch" in df.columns else df.index

                st.subheader(f"📉 {selected_metric} over Epochs" if "epoch" in df.columns else f"{selected_metric} over Steps")
                fig, ax = plt.subplots()
                ax.plot(x_axis, df[selected_metric])
                ax.set_xlabel("Epoch" if "epoch" in df.columns else "Step")
                ax.set_ylabel(selected_metric)
                ax.grid(True)
                st.pyplot(fig)


            # --- Data Table and Export ---
            st.subheader("📋 Log Entries")
            st.dataframe(df.tail(500), use_container_width=True)
            st.download_button("📥 Download Full Log (CSV)", df.to_csv(index=False), file_name="training_log.csv")

# --- UNSTRUCTURED VIEW ---
else:
    st.set_page_config(layout="wide")
    st.title("📄 Plaintext Log Viewer")
    st.caption(f"🧪 Log Type: `{log_filename}`")
    if not os.path.exists(log_path):
        st.error("❌ Log file not found.")
    else:
        filter_level = st.selectbox("🔎 Filter log level", ["ALL", "INFO", "WARNING", "ERROR"])
        with open(log_path, "r") as f:
            lines = f.readlines()
            if filter_level != "ALL":
                lines = [line for line in lines if filter_level in line]
        st.text_area("🪵 Log Output", value="".join(lines[-1000:]), height=600)


"""

streamlit run tools/monitoring/streamlit_log_viewer.py \
  -- \
  --experiment_id tb-denoise-aug23 \
  --run_id run03 \
  --log_type training \
  --follow

streamlit run tools/monitoring/streamlit_log_viewer.py \
  -- \
  --experiment_id tb-denoise-aug23 \
  --run_id run03 \
  --log_type output


streamlit run tools/monitoring/streamlit_log_viewer.py \
  -- \
  --experiment_id tb-denoise-aug23 \
  --run_id run03 \
  --log_type structured \
  --follow

"""