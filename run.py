#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None


def sh(cmd, check=True):
    print("$", " ".join(cmd) if isinstance(cmd, (list, tuple)) else cmd)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=check)


def image_exists(tag: str) -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except Exception:
        return False


def build_image(image: str, context: Path):
    sh(["docker", "build", "-t", image, str(context)])


def load_yaml(path: Path) -> dict:
    if not path.exists():
        print(f"[ERROR] YAML not found: {path}")
        sys.exit(2)
    if yaml is None:
        print("[ERROR] PyYAML not available. Please ensure it is installed in Dockerfile.")
        sys.exit(2)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dataset_keys(ds_all: dict, key: str) -> tuple[dict, str]:
    if key not in ds_all:
        print(f"[ERROR] Dataset key '{key}' not found in config/dataset.yaml")
        sys.exit(2)
    ds = ds_all[key]
    # Global data_path lives at top-level (deduplicated)
    data_path = ds_all.get("data_path", "/workspace/data")
    # Required keys inside dataset entry
    for k in ("data_pre", "train_name", "test_name", "dataset_type", "prefix", "data_dim", "k"):
        if k not in ds:
            print(f"[ERROR] Missing '{k}' in dataset '{key}'")
            sys.exit(2)
    return ds, data_path


def compose_paths(ds: dict, data_path: str) -> dict:
    data_path = data_path.rstrip("/")
    data_pre = ds["data_pre"].strip("/")
    train_name = ds["train_name"]
    test_name = ds["test_name"]
    train_path = f"{data_path}/{data_pre}/{train_name}"
    test_path = f"{data_path}/{data_pre}/{test_name}"
    train_stem = Path(train_name).stem
    test_stem = Path(test_name).stem
    return {
        "train_path": train_path,
        "test_path": test_path,
        "train_stem": train_stem,
        "test_stem": test_stem,
        "data_pre_path": f"{data_path}/{data_pre}",
    }


def cmake_build_target(workdir: str, target: str) -> str:
    return f"""
set -e
cd {workdir}
rm -rf build
cmake -S . -B build
cmake --build build -j --target {target}
"""


def run_container(image: str, project_dir: Path, data_dir: Path, algo: str, dataset_key: str, mode: str):
    cfg_dir = project_dir / "config"
    ds_all = load_yaml(cfg_dir / "dataset.yaml")
    algo_all = load_yaml(cfg_dir / "algorithm.yaml")

    ds, data_path = ensure_dataset_keys(ds_all, dataset_key)
    ds_paths = compose_paths(ds, data_path)

    algo_cfg = (algo_all.get(algo) or {}) if isinstance(algo_all, dict) else {}

    workdir = "/workspace"
    pre_path = f"{workdir}/benchmark_index"
    algo_dir = f"{pre_path}/{algo}"

    inner = None

    if algo == "hnsw":
        efc = int(algo_cfg.get("efc", 256))
        M = int(algo_cfg.get("M", 32))
        efs = list(map(int, algo_cfg.get("efs", [100, 200, 300, 400, 500, 600, 800, 1000, 1500])))
        atype = algo_cfg.get("type", "nn")
        index_prefix = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}.index"
        result_prefix = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}.result"
        log_file = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}.log"
        result_name = f"{ds['prefix']}_M{M}_L{efc}"

        inner = (
            cmake_build_target(workdir, "benchmark_hnsw")
            + f"""
mkdir -p {algo_dir}
if [ "{mode}" = build ]; then
  cd build
  ./test/benchmark_hnsw "{ds_paths['train_path']}" "{ds_paths['test_path']}" build "{ds['data_dim']}" "{ds['k']}" "{efc}" "{M}" "{index_prefix}" "{result_prefix}" | tee -a "{log_file}"
else
  cd build
  for ef in {' '.join(map(str, efs))}; do
    echo ======================================== | tee -a "{log_file}"
    echo Running with efs: $ef | tee -a "{log_file}"
    ./test/benchmark_hnsw "{ds_paths['train_path']}" "{ds_paths['test_path']}" search "{ds['data_dim']}" "{ds['k']}" "{efc}" "{M}" "{index_prefix}" "{result_prefix}" "$ef" | tee -a "{log_file}"
    python3 "{workdir}/evaluator.py" "{ds_paths['data_pre_path']}" "{ds['prefix']}" "{ds_paths['train_stem']}" "{ds_paths['test_stem']}" "{algo}" "{ds['k']}" "{atype}" "{result_name}" "{pre_path}" --dataset-type "{ds['dataset_type']}" | tee -a "{log_file}"
    echo ======================================== | tee -a "{log_file}"
  done
fi
"""
        )

    elif algo == "scann":
        num_leaves = int(algo_cfg.get("num_leaves", 2000))
        num_leaves_to_search = int(algo_cfg.get("num_leaves_to_search", 1000))
        index_prefix = f"{algo_dir}/{ds['prefix']}.index"
        log_file = f"{algo_dir}/{ds['prefix']}.log"
        inner = f"""
set -e
cd {workdir}
mkdir -p {algo_dir}
python3 "{workdir}/test/benchmark_scann.py" \
  "{ds_paths['data_pre_path']}" \
  "{ds['prefix']}" \
  "{ds_paths['train_stem']}" \
  "{ds_paths['test_stem']}" \
  "{algo}" \
  "{ds['k']}" \
  "{ds.get('data_num', 0)}" \
  "{ds.get('query_num', 0)}" \
  "{ds['data_dim']}" \
  "{num_leaves}" \
  "{num_leaves_to_search}" \
  "{ds['dataset_type']}" \
  "{mode}" \
  "{index_prefix}" \
  --project_root "{workdir}" | tee -a "{log_file}"
"""

    elif algo == "nsg":
        L = int(algo_cfg.get("L", 60))
        R = int(algo_cfg.get("R", 48))
        C = int(algo_cfg.get("C", 256))
        knng_L = int(algo_cfg.get("knng_L", 100))
        knng_iter = int(algo_cfg.get("knng_iter", 10))
        knng_S = int(algo_cfg.get("knng_S", 15))
        knng_R = int(algo_cfg.get("knng_R", 60))
        efs = list(map(int, algo_cfg.get("efs", [100,150,200,250,300,350,400,450,500,550,600,800,1000,1500])))
        atype = algo_cfg.get("type", "nn")
        index_prefix = f"{algo_dir}/{ds['prefix']}_M{R}_L{C}.index"
        result_prefix = f"{algo_dir}/{ds['prefix']}_M{R}_L{C}.result"
        log_file = f"{algo_dir}/{ds['prefix']}_M{R}_L{C}.log"
        result_name = f"{ds['prefix']}_M{R}_L{C}"
        knng_path = f"{ds_paths['data_pre_path']}/{ds_paths['train_stem']}.knng"

        inner = (
            cmake_build_target(workdir, "benchmark_nsg")
            + f"""
mkdir -p {algo_dir}
cd build
if [ "{mode}" = build ]; then
  ./test/benchmark_nsg "{ds_paths['train_path']}" "{ds_paths['test_path']}" build "{ds['data_dim']}" "{ds['k']}" "{knng_L}" "{knng_iter}" "{knng_S}" "{knng_R}" "{L}" "{R}" "{C}" "{index_prefix}" "{knng_path}" "{result_prefix}" | tee -a "{log_file}"
else
  for ef in {' '.join(map(str, efs))}; do
    echo ======================================== | tee -a "{log_file}"
    echo Running with efs: $ef | tee -a "{log_file}"
    ./test/benchmark_nsg "{ds_paths['train_path']}" "{ds_paths['test_path']}" search "{ds['data_dim']}" "{ds['k']}" "{knng_L}" "{knng_iter}" "{knng_S}" "{knng_R}" "{L}" "{R}" "{C}" "{index_prefix}" "{knng_path}" "{result_prefix}" "$ef" | tee -a "{log_file}"
    python3 "{workdir}/evaluator.py" "{ds_paths['data_pre_path']}" "{ds['prefix']}" "{ds_paths['train_stem']}" "{ds_paths['test_stem']}" "{algo}" "{ds['k']}" "{atype}" "{result_name}" "{pre_path}" --dataset-type "{ds['dataset_type']}" | tee -a "{log_file}"
    echo ======================================== | tee -a "{log_file}"
  done
fi
"""
        )

    elif algo == "mag":
        L = int(algo_cfg.get("L", 60))
        R = int(algo_cfg.get("R", 48))
        C = int(algo_cfg.get("C", 300))
        R_IP = int(algo_cfg.get("R_IP", 15))
        M = int(algo_cfg.get("M", 48))
        threshold = int(algo_cfg.get("threshold", 5))
        efs = list(map(int, algo_cfg.get("efs", [100,150,200,250,300,500,800,1000,1500])))
        atype = algo_cfg.get("type", "ip")
        index_prefix = f"{algo_dir}/{ds['prefix']}_M{M}_L{C}.index"
        result_prefix = f"{algo_dir}/{ds['prefix']}_M{M}_L{C}.result"
        log_file = f"{algo_dir}/{ds['prefix']}_M{M}_L{C}.log"
        result_name = f"{ds['prefix']}_M{M}_L{C}"
        knng_path = f"{ds_paths['data_pre_path']}/{ds_paths['train_stem']}.knng"

        inner = (
            cmake_build_target(workdir, "benchmark_mag")
            + f"""
mkdir -p {algo_dir}
cd build
if [ "{mode}" = build ]; then
  ./test/benchmark_mag "{ds_paths['train_path']}" "{ds_paths['test_path']}" build "{ds['data_dim']}" "{ds['k']}" "{L}" "{R}" "{C}" "{R_IP}" "{M}" "{threshold}" "{index_prefix}" "{knng_path}" "{result_prefix}" | tee -a "{log_file}"
else
  for ef in {' '.join(map(str, efs))}; do
    echo ======================================== | tee -a "{log_file}"
    echo Running with efs: $ef | tee -a "{log_file}"
    ./test/benchmark_mag "{ds_paths['train_path']}" "{ds_paths['test_path']}" search "{ds['data_dim']}" "{ds['k']}" "{L}" "{R}" "{C}" "{R_IP}" "{M}" "{threshold}" "{index_prefix}" "{knng_path}" "{result_prefix}" "$ef" | tee -a "{log_file}"
    python3 "{workdir}/evaluator.py" "{ds_paths['data_pre_path']}" "{ds['prefix']}" "{ds_paths['train_stem']}" "{ds_paths['test_stem']}" "{algo}" "{ds['k']}" "{atype}" "{result_name}" "{pre_path}" --dataset-type "{ds['dataset_type']}" | tee -a "{log_file}"
    echo ======================================== | tee -a "{log_file}"
  done
fi
"""
        )

    elif algo == "ipnsw":
        efc = int(algo_cfg.get("efc", 256))
        M = int(algo_cfg.get("M", 32))
        efs = list(map(int, algo_cfg.get("efs", [100,200,300,400,450,500,600,800,1000,1500,2000,3000])))
        atype = algo_cfg.get("type", "ip")
        index_prefix = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}.index"
        result_prefix = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}.result"
        log_file = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}.log"
        result_name = f"{ds['prefix']}_M{M}_L{efc}"

        inner = (
            cmake_build_target(workdir, "benchmark_ipnsw")
            + f"""
mkdir -p {algo_dir}
cd build
if [ "{mode}" = build ]; then
  ./test/benchmark_ipnsw "{ds_paths['train_path']}" "{ds_paths['test_path']}" build "{ds['data_dim']}" "{ds['k']}" "{efc}" "{M}" "{index_prefix}" "{result_prefix}" | tee -a "{log_file}"
else
  for ef in {' '.join(map(str, efs))}; do
    echo ======================================== | tee -a "{log_file}"
    echo Running with efs: $ef | tee -a "{log_file}"
    ./test/benchmark_ipnsw "{ds_paths['train_path']}" "{ds_paths['test_path']}" search "{ds['data_dim']}" "{ds['k']}" "{efc}" "{M}" "{index_prefix}" "{result_prefix}" "$ef" | tee -a "{log_file}"
    python3 "{workdir}/evaluator.py" "{ds_paths['data_pre_path']}" "{ds['prefix']}" "{ds_paths['train_stem']}" "{ds_paths['test_stem']}" "{algo}" "{ds['k']}" "{atype}" "{result_name}" "{pre_path}" --dataset-type "{ds['dataset_type']}" | tee -a "{log_file}"
    echo ======================================== | tee -a "{log_file}"
  done
fi
"""
        )

    elif algo == "ipnsw+":
        efc = int(algo_cfg.get("efc", 256))
        M = int(algo_cfg.get("M", 32))
        cos_M = int(algo_cfg.get("cos_M", 10))
        cos_efConstruction = int(algo_cfg.get("cos_efConstruction", 100))
        cos_efsearch = int(algo_cfg.get("cos_efsearch", 1))
        efs = list(map(int, algo_cfg.get("efs", [100,120,150,200,250,300,350,400,450,500,550,600,800,1000,1500])))
        atype = algo_cfg.get("type", "ip")
        index_prefix = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}_COSM{cos_M}_COSL{cos_efConstruction}.index"
        result_prefix = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}_COSM{cos_M}_COSL{cos_efConstruction}.result"
        log_file = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}_COSM{cos_M}_COSL{cos_efConstruction}.log"
        result_name = f"{ds['prefix']}_M{M}_L{efc}_COSM{cos_M}_COSL{cos_efConstruction}"

        inner = (
            cmake_build_target(workdir, "benchmark_ipnsw+")
            + f"""
mkdir -p {algo_dir}
cd build
if [ "{mode}" = build ]; then
  ./test/benchmark_ipnsw+ "{ds_paths['train_path']}" "{ds_paths['test_path']}" build "{ds['data_dim']}" "{ds['k']}" "{efc}" "{cos_efConstruction}" "{M}" "{cos_M}" "{cos_efsearch}" "{index_prefix}" "{result_prefix}" | tee -a "{log_file}"
else
  for ef in {' '.join(map(str, efs))}; do
    echo ======================================== | tee -a "{log_file}"
    echo Running with efs: $ef | tee -a "{log_file}"
    ./test/benchmark_ipnsw+ "{ds_paths['train_path']}" "{ds_paths['test_path']}" search "{ds['data_dim']}" "{ds['k']}" "{efc}" "{cos_efConstruction}" "{M}" "{cos_M}" "{cos_efsearch}" "{index_prefix}" "{result_prefix}" "$ef" | tee -a "{log_file}"
    python3 "{workdir}/evaluator.py" "{ds_paths['data_pre_path']}" "{ds['prefix']}" "{ds_paths['train_stem']}" "{ds_paths['test_stem']}" "{algo}" "{ds['k']}" "{atype}" "{result_name}" "{pre_path}" --dataset-type "{ds['dataset_type']}" | tee -a "{log_file}"
    echo ======================================== | tee -a "{log_file}"
  done
fi
"""
        )

    elif algo == "napg":
        efc = int(algo_cfg.get("efc", 256))
        M = int(algo_cfg.get("M", 32))
        norm_range = int(algo_cfg.get("norm_range", 100))
        sample_size = int(algo_cfg.get("sample_size", 10))
        efs = list(map(int, algo_cfg.get("efs", [100,200,300,400,500,600,800,1000,1500])))
        atype = algo_cfg.get("type", "ip")
        index_prefix = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}.index"
        result_prefix = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}.result"
        log_file = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}.log"
        result_name = f"{ds['prefix']}_M{M}_L{efc}"

        inner = (
            cmake_build_target(workdir, "benchmark_napg")
            + f"""
mkdir -p {algo_dir}
cd build
if [ "{mode}" = build ]; then
  ./test/benchmark_napg "{ds_paths['train_path']}" "{ds_paths['test_path']}" build "{ds['data_dim']}" "{ds['k']}" "{efc}" "{M}" "{norm_range}" "{sample_size}" "{index_prefix}" "{result_prefix}" | tee -a "{log_file}"
else
  for ef in {' '.join(map(str, efs))}; do
    echo ======================================== | tee -a "{log_file}"
    echo Running with efs: $ef | tee -a "{log_file}"
    ./test/benchmark_napg "{ds_paths['train_path']}" "{ds_paths['test_path']}" search "{ds['data_dim']}" "{ds['k']}" "{efc}" "{M}" "{norm_range}" "{sample_size}" "{index_prefix}" "{result_prefix}" "$ef" | tee -a "{log_file}"
    python3 "{workdir}/evaluator.py" "{ds_paths['data_pre_path']}" "{ds['prefix']}" "{ds_paths['train_stem']}" "{ds_paths['test_stem']}" "{algo}" "{ds['k']}" "{atype}" "{result_name}" "{pre_path}" --dataset-type "{ds['dataset_type']}" | tee -a "{log_file}"
    echo ======================================== | tee -a "{log_file}"
  done
fi
"""
        )

    elif algo == "mobius":
        efc = int(algo_cfg.get("efc", 256))
        M = int(algo_cfg.get("M", 32))
        efs = list(map(int, algo_cfg.get("efs", [100,300,500,800,1000,1500])))
        atype = algo_cfg.get("type", "ip")
        index_prefix = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}.index"
        result_prefix = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}.result"
        log_file = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}.log"
        result_name = f"{ds['prefix']}_M{M}_L{efc}"

        inner = (
            cmake_build_target(workdir, "benchmark_mobius")
            + f"""
mkdir -p {algo_dir}
cd build
if [ "{mode}" = build ]; then
  ./test/benchmark_mobius "{ds_paths['train_path']}" "{ds_paths['test_path']}" build "{ds['data_dim']}" "{ds['k']}" "{efc}" "{M}" "{index_prefix}" "{result_prefix}" | tee -a "{log_file}"
else
  for ef in {' '.join(map(str, efs))}; do
    echo ======================================== | tee -a "{log_file}"
    echo Running with efs: $ef | tee -a "{log_file}"
    ./test/benchmark_mobius "{ds_paths['train_path']}" "{ds_paths['test_path']}" search "{ds['data_dim']}" "{ds['k']}" "{efc}" "{M}" "{index_prefix}" "{result_prefix}" "$ef" | tee -a "{log_file}"
    python3 "{workdir}/evaluator.py" "{ds_paths['data_pre_path']}" "{ds['prefix']}" "{ds_paths['train_stem']}" "{ds_paths['test_stem']}" "{algo}" "{ds['k']}" "{atype}" "{result_name}" "{pre_path}" --dataset-type "{ds['dataset_type']}" | tee -a "{log_file}"
    echo ======================================== | tee -a "{log_file}"
  done
fi
"""
        )

    elif algo == "vamana":
        L = int(algo_cfg.get("L", 60))
        R = int(algo_cfg.get("R", 48))
        C = int(algo_cfg.get("C", 256))
        alpha = float(algo_cfg.get("alpha", 1.2))
        efs = list(map(int, algo_cfg.get("efs", [100,150,200,250,300,350,400,450,500,550,600,800,1000,1500])))
        atype = algo_cfg.get("type", "nn")
        index_prefix = f"{algo_dir}/{ds['prefix']}_M{R}_L{C}_alpha{alpha}.index"
        result_prefix = f"{algo_dir}/{ds['prefix']}_M{R}_L{C}.result"
        log_file = f"{algo_dir}/{ds['prefix']}_M{R}_L{C}_alpha{alpha}.log"
        result_name = f"{ds['prefix']}_M{R}_L{C}"

        inner = (
            cmake_build_target(workdir, "benchmark_vamana")
            + f"""
mkdir -p {algo_dir}
cd build
if [ "{mode}" = build ]; then
  ./test/benchmark_vamana "{ds_paths['train_path']}" "{ds_paths['test_path']}" build "{ds['data_dim']}" "{ds['k']}" "{L}" "{R}" "{C}" "{alpha}" "{index_prefix}" "{result_prefix}" | tee -a "{log_file}"
else
  for ef in {' '.join(map(str, efs))}; do
    echo ======================================== | tee -a "{log_file}"
    echo Running with efs: $ef | tee -a "{log_file}"
    ./test/benchmark_vamana "{ds_paths['train_path']}" "{ds_paths['test_path']}" search "{ds['data_dim']}" "{ds['k']}"  "{L}" "{R}" "{C}" "{alpha}" "{index_prefix}" "{result_prefix}" "$ef" | tee -a "{log_file}"
    python3 "{workdir}/evaluator.py" "{ds_paths['data_pre_path']}" "{ds['prefix']}" "{ds_paths['train_stem']}" "{ds_paths['test_stem']}" "{algo}" "{ds['k']}" "{atype}" "{result_name}" "{pre_path}" --dataset-type "{ds['dataset_type']}" | tee -a "{log_file}"
    echo ======================================== | tee -a "{log_file}"
  done
fi
"""
        )

    elif algo == "dblsh":
        L = int(algo_cfg.get("L", 5))
        K_dblsh = int(algo_cfg.get("K_dblsh", 10))
        beta_list = list(map(float, algo_cfg.get("beta_list", [0.8, 0.9, 0.92, 0.94, 0.96, 0.98])))
        atype = algo_cfg.get("type", "nn")
        result_prefix = f"{algo_dir}/{ds['prefix']}_L{L}_K{K_dblsh}.result"
        log_file = f"{algo_dir}/{ds['prefix']}_L{L}_K{K_dblsh}.log"
        result_name = f"{ds['prefix']}_L{L}_K{K_dblsh}"

        inner = (
            cmake_build_target(workdir, "benchmark_dblsh")
            + f"""
mkdir -p {algo_dir}
cd build
for beta in {' '.join(map(str, beta_list))}; do
  echo ======================================== | tee -a "{log_file}"
  echo Running with beta: $beta | tee -a "{log_file}"
  ./test/benchmark_dblsh "{ds_paths['train_path']}" "{ds_paths['test_path']}" "{ds['data_dim']}" "{ds['k']}" "{L}" "{K_dblsh}" 1.5 "$beta" 0.1 "{result_prefix}" | tee -a "{log_file}"
  python3 "{workdir}/evaluator.py" "{ds_paths['data_pre_path']}" "{ds['prefix']}" "{ds_paths['train_stem']}" "{ds_paths['test_stem']}" "{algo}" "{ds['k']}" "{atype}" "{result_name}" "{pre_path}" --dataset-type "{ds['dataset_type']}" | tee -a "{log_file}"
  echo ======================================== | tee -a "{log_file}"
done
"""
        )

    elif algo == "fargo":
        L = int(algo_cfg.get("L", 6))
        K_fargo = int(algo_cfg.get("K_fargo", 12))
        ub_list = list(map(int, algo_cfg.get("ub", [0,30,60,80,100,130,160,180,200,300,400,500,600,700])))
        atype = algo_cfg.get("type", "ip")
        index_prefix = f"{algo_dir}/{ds['prefix']}_L{L}_K{K_fargo}.index"
        result_prefix = f"{algo_dir}/{ds['prefix']}_L{L}_K{K_fargo}.result"
        log_file = f"{algo_dir}/{ds['prefix']}_L{L}_K{K_fargo}.log"
        result_name = f"{ds['prefix']}_L{L}_K{K_fargo}"

        inner = (
            cmake_build_target(workdir, "benchmark_fargo")
            + f"""
mkdir -p {algo_dir}
cd build
if [ "{mode}" = build ]; then
  ./test/benchmark_fargo "{ds_paths['train_path']}" "{ds_paths['test_path']}" build "{ds['data_dim']}" "{ds['k']}" "{L}" "{K_fargo}" 0 0.8 1 "{index_prefix}" "{result_prefix}" | tee -a "{log_file}"
else
  for ub in {' '.join(map(str, ub_list))}; do
    echo ======================================== | tee -a "{log_file}"
    echo Running with ub: $ub, break_ratio: 1 | tee -a "{log_file}"
    ./test/benchmark_fargo "{ds_paths['train_path']}" "{ds_paths['test_path']}" search "{ds['data_dim']}" "{ds['k']}" "{L}" "{K_fargo}" "$ub" 0.8 1 "{index_prefix}" "{result_prefix}" | tee -a "{log_file}"
    python3 "{workdir}/evaluator.py" "{ds_paths['data_pre_path']}" "{ds['prefix']}" "{ds_paths['train_stem']}" "{ds_paths['test_stem']}" "{algo}" "{ds['k']}" "{atype}" "{result_name}" "{pre_path}" --dataset-type "{ds['dataset_type']}" | tee -a "{log_file}"
    echo ======================================== | tee -a "{log_file}"
  done
fi
"""
        )

    elif algo == "rabitq":
        efc = int(algo_cfg.get("efc", 4096))
        M = int(ds["data_dim"]) if algo_cfg.get("M_from_dim", True) else int(algo_cfg.get("M", ds["data_dim"]))
        efs = list(map(int, algo_cfg.get("efs", [1,2,4,5,8,10,20,30,50,60,100,200,300,500])))
        atype = algo_cfg.get("type", "nn")
        index_prefix = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}.index"
        result_prefix = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}.result"
        log_file = f"{algo_dir}/{ds['prefix']}_M{M}_L{efc}.log"
        result_name = f"{ds['prefix']}_M{M}_L{efc}"
        dataset_name = ds['prefix']

        inner = (
            cmake_build_target(workdir, "benchmark_rabitq")
            + f"""
mkdir -p {algo_dir}
cd build
if [ "{mode}" = build ]; then
  ./test/benchmark_rabitq "{ds_paths['train_path']}" "{ds_paths['test_path']}" build "{ds['data_dim']}" "{ds['k']}" "{efc}" "{M}" "{index_prefix}" "{result_prefix}" "{dataset_name}" "{workdir}" | tee -a "{log_file}"
else
  for ef in {' '.join(map(str, efs))}; do
    echo ======================================== | tee -a "{log_file}"
    echo Running with efs: $ef | tee -a "{log_file}"
    ./test/benchmark_rabitq "{ds_paths['train_path']}" "{ds_paths['test_path']}" search "{ds['data_dim']}" "{ds['k']}" "{efc}" "{M}" "{index_prefix}" "{result_prefix}" "{dataset_name}" "$ef" | tee -a "{log_file}"
    python3 "{workdir}/evaluator.py" "{ds_paths['data_pre_path']}" "{ds['prefix']}" "{ds_paths['train_stem']}" "{ds_paths['test_stem']}" "{algo}" "{ds['k']}" "{atype}" "{result_name}" "{pre_path}" --dataset-type "{ds['dataset_type']}" | tee -a "{log_file}"
    echo ======================================== | tee -a "{log_file}"
  done
fi
"""
        )

    elif algo == "ivfpq":
        nlist = int(algo_cfg.get("nlist", 12000))
        pqm = int(algo_cfg.get("pqm", 48))
        rerank_k = int(algo_cfg.get("rerank_k", 300))
        index_prefix = f"{algo_dir}/{ds['prefix']}_ivf{nlist}_pq{pqm}.index"
        log_file = f"{algo_dir}/{ds['prefix']}_ivf{nlist}_pq{pqm}.log"

        inner = f"""
set -e
cd {workdir}
mkdir -p {algo_dir}
python3 "{workdir}/test/benchmark_ivfpq.py" \
    "{ds_paths['data_pre_path']}" \
    "{ds['prefix']}" \
    "{ds_paths['train_stem']}" \
    "{ds_paths['test_stem']}" \
    "{workdir}/tools/recall_{ds['dataset_type']}.py" \
    --dim "{ds['data_dim']}" \
    --data_num "{ds.get('data_num', 0)}" \
    --query_num "{ds.get('query_num', 0)}" \
    --mode "{mode}" \
    --algorithm "{algo}" \
    --top_k "{ds['k']}" \
    --nlist "{nlist}" \
    --pqm "{pqm}" \
    --rerank_k "{rerank_k}" \
    --index_path "{index_prefix}" \
    --output_path "{pre_path}" \
    --data_type "{ds['dataset_type']}" | tee -a "{log_file}"
"""

    else:
        print(f"[ERROR] Unsupported algorithm in demo: {algo}")
        sys.exit(2)

    volumes = [
        "-v", f"{project_dir.resolve()}:{workdir}:rw",
        "-v", f"{data_dir.resolve()}:{workdir}/data:ro",
    ]
    envs = [
        "-e", "PROJECT_ROOT=/workspace",
        "-e", "OpenBLAS_DIR=/usr/lib/x86_64-linux-gnu/openblas-pthread/cmake/openblas",
    ]
    cmd = [
        "docker", "run", "--rm",
        *envs,
        *volumes,
        image,
        "bash", "-lc", inner,
    ]
    sh(cmd)


def main():
    parser = argparse.ArgumentParser(description="Iceberg runner for all algorithms")
    parser.add_argument("algorithm", choices=[
        "hnsw", "scann", "nsg", "mag", "ipnsw", "ipnsw+", "napg", "mobius", "vamana", "dblsh", "fargo", "rabitq", "ivfpq"
    ], help="Algorithm to run")
    parser.add_argument("dataset", type=str, help="Dataset key in config/dataset.yaml (e.g., imagenet1k_dinov2)")
    parser.add_argument("--mode", default="search", choices=["build", "search"], help="Run mode")
    parser.add_argument("--image", default="iceberg:latest", help="Docker image tag")
    parser.add_argument("--data", default=None, help="Host data root")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    data_dir = Path(args.data) if args.data else (repo_root / "data")
    if not data_dir.exists():
        print(f"[WARN] Data directory does not exist on host: {data_dir}")

    if not image_exists(args.image):
        build_image(args.image, repo_root)

    run_container(
        image=args.image,
        project_dir=repo_root,
        data_dir=data_dir,
        algo=args.algorithm,
        dataset_key=args.dataset,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
