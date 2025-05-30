Here is your **refined prompt**, structured for clarity, Git best practices, and multi-environment development. It cleanly explains the branching model, output organization, and collaborative flow between local and cluster work:

---

### ✅ **Refined Prompt: Implementing Branching Strategy and Output Isolation Across Local and Cluster Projects**

We have set up **local and cluster projects** pointing to the **same GitHub repository**. To support collaborative and environment-aware development, we need to formalize:

---

### 🧭 **1. Core Branching Strategy**

#### 🔵 `main` – Production-Ready Code

* **Purpose**: Stable, reproducible releases with finalized models and clean code.
* **Rules**:

  * **Protected branch**: No direct commits.
  * All updates via **Pull Requests (PRs)** from `dev`.
  * Requires:

    * PR review
    * Passing CI/CD (e.g., tests, model reproducibility)
  * Tag releases:

    * `v1.0-superdiff-tb`, `v1.1-pneumonia-combo`

---

#### 🟢 `dev` – Integration Branch

* **Purpose**: Stable integration of experimental features before promotion to `main`.
* **Rules**:

  * Accepts PRs only from `experiment/*` branches.
  * Must:

    * Pass all CI checks
    * Be reproducible using `config.yaml` + `run.sh`

---

#### 🧪 `experiment/*` – Isolated Research Branches

* **Examples**:

  * `experiment/tb-denoising-aug23`
  * `experiment/pneumonia-hypersearch-sep25`

* **Purpose**: Sandbox for training, experiments, config tuning, and model development.

* **Rules**:

  * Freeform development allowed (commits, pushes)
  * Use commit templates referencing experiment ID & purpose
  * Track run outputs under:
    `runs/<experiment_id>/<run_id>/`

---

#### 🧭 **Local & Cluster Development Workflow**

For each experiment, we can create environment-specific child branches:

```bash
# Branch from the experiment base
experiment/tb-denoising-aug23-local
experiment/tb-denoising-aug23-cluster
```

* Work independently on local or cluster
* Regularly sync with base experiment branch
* When ready:

  * Create PRs **back to** `experiment/tb-denoising-aug23`
  * Resolve conflicts centrally
  * Delete `-local` and `-cluster` branches after merge

This prevents persistent conflicts and aligns work across environments.

---

### 🧰 **2. Folder Structure for Output Isolation**

Organize experiment artifacts under the `runs/` directory:

```bash
runs/
  experimenttbaug23/
    run01/
      config.yaml
      logs/
      model.pt
      metrics.json
    samples/
    visualizations/
```

#### 📁 Output Rules

* **No model files or artifacts in repo root**
* Use `.gitignore` to prevent tracking large/binary files:

```gitignore
runs/**
outputs/**
*.pt
*.ckpt
```

#### 🏷️ Tagging Convention

Use structured tags for experiments:

```bash
exp-<disease>-<short-description>-<date>
# e.g.:
exp-tb-denoise-aug23
exp-pneumonia-tune-sep25
```

---

### 🧠 Summary

* Use `experiment/*-local` and `*-cluster` branches for parallel work
* Merge both into base `experiment/*`, then PR into `dev`
* Maintain clean output structure in `runs/`
* Automate reproducibility via `config.yaml` and logging outputs
* Use consistent Git hygiene to avoid large file bloat and merge chaos

---

Would you like a starter shell script to scaffold `runs/`, auto-tag branches, or template PR titles?
