# =======================================
# Universal ML Training Script Analyzer
# =======================================

import ast, json, os, re, tempfile, shutil, subprocess, nbformat, requests
from nbconvert import PythonExporter
import google.generativeai as genai
# Make torch optional
try:
    import torch  # noqa: F401
except Exception:
    torch = None
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

# =======================================
# Local Generic Parameter Estimation
# =======================================
def safe_universal_param_estimate(script_text):
    """
    Safely estimate model parameters from any ML script without executing the code.
    Returns (total_params_estimate, trainable_params_estimate)
    """
    import ast, re
    total = 0
    framework = None
    script_lower = script_text.lower()

    if "torch" in script_lower:
        framework = "pytorch"
    elif "tensorflow" in script_lower or "keras" in script_lower:
        framework = "tensorflow"
    elif "sklearn" in script_lower:
        framework = "sklearn"
    elif "xgboost" in script_lower:
        framework = "xgboost"
    elif "lightgbm" in script_lower:
        framework = "lightgbm"
    elif "jax" in script_lower:
        framework = "jax"
    else:
        framework = "unknown"

    tree = ast.parse(script_text)

    # Deep models
    if framework in ["pytorch", "tensorflow"]:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id
                func_name_lower = func_name.lower()
                args = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, int)]

                if "conv" in func_name_lower and len(args) >= 3:
                    total += args[0]*args[1]*args[2]*args[2] + args[1]
                elif "linear" in func_name_lower or "dense" in func_name_lower:
                    if len(args) >= 2:
                        total += args[0]*args[1] + args[1]

    # Classical
    elif framework == "sklearn":
        if re.search(r"logisticregression", script_lower):
            total += 10000
        elif re.search(r"randomforest", script_lower):
            total += 100000
        elif re.search(r"svm|svc", script_lower):
            total += 50000

    # XGBoost / LightGBM
    elif framework in ["xgboost", "lightgbm"]:
        m = re.search(r"n_estimators\s*=\s*(\d+)", script_lower)
        n_estimators = int(m.group(1)) if m else 100
        total += n_estimators * 1000
        total = int(total)

    # NumPy
    elif "numpy" in script_lower:
        matches = re.findall(r"np\.zeros\((\[?\(?\d+.*?\)?\]?)\)", script_lower)
        for m in matches:
            nums = [int(x) for x in re.findall(r"\d+", m)]
            n = 1
            for d in nums: n *= d
            total += n

    if total == 0:
        total = None
    return total, total

# =======================================
# Enhanced Distributed Detection
# =======================================
def detect_distributed(script_text):
    distributed_libs = {
        "torch.distributed": "PyTorch DDP",
        "deepspeed": "DeepSpeed",
        "accelerate": "HuggingFace Accelerate",
        "horovod": "Horovod",
        "ray.train": "Ray Train",
        "jax.pmap": "JAX Parallel",
    }
    distributed_funcs = [
        "init_process_group",
        "DistributedDataParallel",
        "launch",
        "spawn",
        "deepspeed.initialize",
    ]

    for lib, name in distributed_libs.items():
        if lib in script_text:
            return True, name

    for func in distributed_funcs:
        if func in script_text:
            return True, f"Custom Distributed ({func})"

    return False, None

# =======================================
# LLM Fallback (Generalized)
# =======================================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

json_schema = {
    "type": "object",
    "properties": {
        "framework": {"type": "string", "nullable": True},
        "model_class": {"type": "string", "nullable": True},
        "pretrained": {"type": "boolean"},
        "params_total": {"type": "integer", "nullable": True},
        "trainable_params": {"type": "integer", "nullable": True},
        "optimizer": {"type": "string", "nullable": True},
        "batch_size": {"type": "integer", "nullable": True},
        "epochs": {"type": "integer", "nullable": True},
        "loss_fn": {"type": "string", "nullable": True},
        "distributed_used": {"type": "boolean"},
        "distributed_token": {"type": "string", "nullable": True},
        "data_pipeline": {"type": "string", "nullable": True},
        "parallel_framework": {"type": "string", "nullable": True},
        "preprocessing_steps": {"type": "array", "items": {"type": "string"}, "nullable": True},
        "num_workers": {"type": "integer", "nullable": True},
        "comment": {"type": "string", "nullable": True}
    },
    "required": [
        "framework", "model_class", "pretrained", "params_total",
        "trainable_params", "optimizer", "batch_size", "epochs",
        "loss_fn", "distributed_used", "distributed_token","data_pipeline","parallel_framework",
        "preprocessing_steps","num_workers", "comment"
    ]
}

model = genai.GenerativeModel("gemini-2.5-pro")

def call_llm_for_analysis(script_text):
    prompt = f"""
You are a static machine learning code auditor. Analyze the Python training script provided.
You reason and reply, only in English (and no Chinese at all) and must produce one valid JSON object (no markdown, no extra text).

Your tasks:
1. Extract the framework, model class, optimizer, batch_size, epochs, loss_fn, distributed info.
2. **Calculate `params_total` and `trainable_params`** ...
3. If a parameter cannot be calculated exactly, make a reasonable estimate and note it in the `comment`.
4. If a value is missing, assume a common default.
5. Keep your reasoning concise in the `comment` field (1–3 sentences max).

Return JSON only:

Script:
{script_text}
"""
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=json_schema
        )
    )
    return response.text

# =======================================
# AST-Based Universal Analyzer
# =======================================
class ScriptAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.result = {
            "framework": None,
            "model_class": None,
            "pretrained": False,
            "params_total": None,
            "trainable_params": None,
            "optimizer": None,
            "batch_size": None,
            "epochs": None,
            "loss_fn": None,
            "distributed_used": False,
            "distributed_token": None,
            "data_pipeline": None,
            "parallel_framework": None,
            "preprocessing_steps": [],
            "num_workers": None,
            "comment": None,
            # classical extras
            "iterations": None,
            "n_estimators": None,
            "max_iter": None,
            "max_depth": None,
        }

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.name.lower()
            comment_list = []
            if any(f in name for f in [
                "torch", "tensorflow", "jax", "sklearn", "xgboost",
                "lightgbm", "catboost", "keras", "fastai", "transformers"
            ]):
                self.result["framework"] = alias.name.split(".")[0]
                comment_list.append(f"Detected ML framework: {alias.name}")
            if any(f in name for f in ["pandas", "datasets", "polars", "numpy", "torch.utils.data", "tensorflow.data"]):
                self.result["data_pipeline"] = alias.name.split(".")[0]
                comment_list.append(f"Detected data pipeline library: {alias.name}")
            if any(f in name for f in ["ray", "dask", "joblib", "multiprocessing", "concurrent.futures", "threading"]):
                self.result["parallel_framework"] = alias.name.split(".")[0]
                comment_list.append(f"Detected parallel framework: {alias.name}")
            if comment_list:
                self.result["comment"] = "; ".join([c for c in [self.result["comment"], *comment_list] if c])
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        if any(k in node.name.lower() for k in ["model", "net", "cnn", "transformer", "trainer", "agent"]):
            self.result["model_class"] = node.name
            self.result["comment"] = (self.result["comment"] or "") + f"; Found class {node.name} likely defining model."
        self.generic_visit(node)

    def safe_eval(self, node):
        try:
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Num):
                return node.n
        except Exception:
            return None

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id.lower()
            if name in ["epochs", "num_epochs"]:
                self.result["epochs"] = self.safe_eval(node.value)
            elif name in ["batch_size", "bs"]:
                self.result["batch_size"] = self.safe_eval(node.value)
            elif name in ["n_estimators", "num_boost_round"]:
                self.result["n_estimators"] = self.safe_eval(node.value)
            elif name in ["max_iter"]:
                self.result["max_iter"] = self.safe_eval(node.value)
            elif name in ["max_depth"]:
                self.result["max_depth"] = self.safe_eval(node.value)
        self.generic_visit(node)

    def get_func_name(self, node):
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        if isinstance(node.func, ast.Name):
            return node.func.id
        return ""

    def visit_Call(self, node):
        func = self.get_func_name(node)
        lname = func.lower()
        if any(k in lname for k in ["optim", "adam", "sgd", "adagrad", "rmsprop", "optimizer"]):
            self.result["optimizer"] = func
        if any(k in lname for k in ["loss", "criterion", "mse", "crossentropy"]):
            self.result["loss_fn"] = func

        preprocessing_funcs = ["normalize", "resize", "crop", "randomcrop", "totensor",
                               "standardscaler", "minmaxscaler", "labelencoder", "onehotencoder"]
        if any(pf in lname for pf in preprocessing_funcs):
            if func not in self.result["preprocessing_steps"]:
                self.result["preprocessing_steps"].append(func)

        for kw in getattr(node, "keywords", []):
            if kw.arg == "num_workers":
                try:
                    self.result["num_workers"] = self.safe_eval(kw.value)
                except:
                    pass
            elif kw.arg == "n_estimators":
                try:
                    self.result["n_estimators"] = self.safe_eval(kw.value)
                except:
                    pass
            elif kw.arg == "max_iter":
                try:
                    self.result["max_iter"] = self.safe_eval(kw.value)
                except:
                    pass
            elif kw.arg == "max_depth":
                try:
                    self.result["max_depth"] = self.safe_eval(kw.value)
                except:
                    pass

        if lname == "from_pretrained":
            self.result["pretrained"] = True
        self.generic_visit(node)

# =======================================
# Input Normalization
# =======================================
def get_script_text(source):
    # Local file
    if os.path.isfile(source):
        if source.endswith(".ipynb"):
            nb = nbformat.read(source, as_version=4)
            script_text, _ = PythonExporter().from_notebook_node(nb)
            reasoning = "Extracted code from local Jupyter notebook."
            return f"# Reasoning: {reasoning}\n{script_text}"
        elif source.endswith(".py"):
            reasoning = "Read directly from local Python file."
            return f"# Reasoning: {reasoning}\n" + open(source, "r", encoding="utf-8").read()
        else:
            raise ValueError("Unsupported local file format. Use .py or .ipynb")

    # JSON config
    if source.strip().startswith("{"):
        try:
            json.loads(source)
            reasoning = "Parsed as JSON configuration for model hyperparameters."
            return f"# Reasoning: {reasoning}\n# Auto JSON config\nconfig = {source}"
        except json.JSONDecodeError:
            pass

    # Git repos
    if any(domain in source for domain in ["github.com", "gitlab.com"]):
        tmp = tempfile.mkdtemp()
        try:
            subprocess.run(["git", "clone", source, tmp], check=True, stdout=subprocess.DEVNULL)
            candidates = ["train.py", "main.py", "finetune.py"]
            for root, _, files in os.walk(tmp):
                for f in files:
                    if f.lower() in candidates:
                        reasoning = f"Cloned repo and detected training script '{f}'"
                        script_path = os.path.join(root, f)
                        script_text = open(script_path, "r", encoding="utf-8").read()
                        return f"# Reasoning: {reasoning}\n{script_text}"
            raise FileNotFoundError("No main training script found in repo.")
        finally:
            # On Windows, some .git files may be locked; ignore errors during cleanup.
            try:
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception:
                pass

    # HF repo metadata
    if "huggingface.co" in source:
        model_id = source.split("huggingface.co/")[-1].strip("/")
        api_url = f"https://huggingface.co/api/models/{model_id}"
        resp = requests.get(api_url)
        if resp.status_code == 200:
            data = resp.json()
            reasoning = f"Fetched metadata from Hugging Face for model '{model_id}'."
            script_text = json.dumps(data, indent=2)
            return f"# Reasoning: {reasoning}\n{script_text}"
        else:
            raise ValueError(f"Failed to fetch Hugging Face model info: {model_id}")

    # Cloud placeholders
    cloud_prefixes = ["s3://", "gs://", "https://storage.googleapis.com", "https://blob.core.windows.net"]
    if any(prefix in source for prefix in cloud_prefixes):
        reasoning = "Detected cloud storage path (future implementation placeholder)."
        return f"# Reasoning: {reasoning}\n# Cloud link placeholder for {source}"

    # Raw .py URL
    if source.startswith("http") and source.endswith(".py"):
        resp = requests.get(source)
        if resp.status_code == 200:
            reasoning = f"Downloaded raw Python script from URL: {source}"
            return f"# Reasoning: {reasoning}\n{resp.text}"
        else:
            raise ValueError(f"Could not download Python file from {source}")

    raise ValueError("Unknown or unsupported source type.")

# =======================================
# Master Analyze Function
# =======================================
def analyze_script_source(source, use_llm_fallback=True):
    script_text = get_script_text(source)
    dist, token = detect_distributed(script_text)

    analyzer = ScriptAnalyzer()
    tree = ast.parse(script_text)
    analyzer.visit(tree)
    result = analyzer.result
    result["distributed_used"] = dist
    result["distributed_token"] = token

    total, trainable = safe_universal_param_estimate(script_text)
    if total: result["params_total"] = total
    if trainable: result["trainable_params"] = trainable

    result_llm = None
    if use_llm_fallback:
        result_llm = call_llm_for_analysis(script_text)
        for k, v in {}.items() if not result_llm else json.loads(result_llm).items():
            if result.get(k) in [None, False]:
                result[k] = v
    return {"result": result, "result_llm": result_llm}

HF_API_URL = "https://huggingface.co/api/models/{}"

def call_llm_agent(prompt, model_name="gemini-2.5-flash"):
    try:
        model_obj = genai.GenerativeModel(model_name)
        json_schema_local = {
            "type": "object",
            "properties": {
                "framework": {"type": "string", "nullable": True},
                "model_class": {"type": "string", "nullable": True},
                "pretrained": {"type": "boolean"},
                "params_total": {"type": "integer", "nullable": True},
                "trainable_params": {"type": "integer", "nullable": True},
                "optimizer": {"type": "string", "nullable": True},
                "batch_size": {"type": "integer", "nullable": True},
                "epochs": {"type": "integer", "nullable": True},
                "loss_fn": {"type": "string", "nullable": True},
                "distributed_used": {"type": "boolean"},
                "distributed_token": {"type": "string", "nullable": True},
                "comment": {"type": "string", "nullable": True}
            },
            "required": [
                "framework", "model_class", "pretrained", "params_total",
                "trainable_params", "optimizer", "batch_size", "epochs",
                "loss_fn", "distributed_used", "distributed_token", "comment"
            ]
        }
        response = model_obj.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=json_schema_local
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}")
        return None

def get_hf_model_metadata(model_name):
    """
    Always calls Hugging Face API (no cache).
    """
    try:
        print(f"[INFO] Fetching metadata from Hugging Face for '{model_name}' ...")
        resp = requests.get(HF_API_URL.format(model_name), timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            print("✅ Raw HF data received:")
            print(json.dumps(data, indent=2)[:1500])

            params = None
            if "safetensors" in data and isinstance(data["safetensors"], dict):
                params = data["safetensors"].get("total")
            if not params:
                params = data.get("config", {}).get("num_parameters")

            config = data.get("config", {})
            framework = (config.get("framework") or data.get("library_name") or "pytorch")
            architecture = (
                data.get("pipeline_tag")
                or config.get("model_type")
                or (config.get("architectures", [None])[0] if isinstance(config.get("architectures"), list) else None)
            )

            result = {
                "params_total": params,
                "trainable_params": params,
                "framework": framework,
                "architecture": architecture,
                "source": "huggingface_api"
            }
            print(f"[INFO] ✅ HF API success for '{model_name}' → {result}")
            return result
        else:
            print(f"[WARN] Hugging Face API returned status {resp.status_code} for {model_name}")
    except Exception as e:
        print(f"[WARN] HF API fetch failed for {model_name}: {e}")

    prompt = f"""
    Find the approximate number of parameters and architecture type for the Hugging Face model "{model_name}".
    Return JSON only: {{ "params_total": <int>, "trainable_params": <int>, "framework": "pytorch/tensorflow", "architecture": "<str>" }}
    """
    llm_result = call_llm_agent(prompt)
    if llm_result and "params_total" in llm_result:
        llm_result["source"] = "llm_fallback"
        print(f"[INFO] ✅ Using LLM fallback for '{model_name}'")
        return llm_result

    print(f"[WARN] ❌ Unable to retrieve metadata for '{model_name}'")
    return {
        "params_total": None,
        "trainable_params": None,
        "framework": None,
        "architecture": None,
        "source": "unknown"
    }

def extract_hf_model_from_script(script_text):
    matches = re.findall(r'from_pretrained\(\s*["\']([\w\-/]+)["\']', script_text)
    return matches[0] if matches else None

def analyze_script_source_with_hf(source, use_llm_fallback=True):
    script_text = get_script_text(source)
    dist, token = detect_distributed(script_text)

    analyzer = ScriptAnalyzer()
    tree = ast.parse(script_text)
    analyzer.visit(tree)
    result = analyzer.result
    result["distributed_used"] = dist
    result["distributed_token"] = token

    hf_model_name = extract_hf_model_from_script(script_text)
    if hf_model_name:
        hf_meta = get_hf_model_metadata(hf_model_name)
        result["framework"] = hf_meta.get("framework") or result["framework"]
        result["model_class"] = hf_meta.get("architecture") or result["model_class"]
        result["pretrained"] = True
        result["params_total"] = hf_meta.get("params_total")
        result["trainable_params"] = hf_meta.get("trainable_params")
        result["comment"] = f"Detected Hugging Face model '{hf_model_name}', metadata filled from cache/API."
    else:
        total, trainable = safe_universal_param_estimate(script_text)
        if total: result["params_total"] = total
        if trainable: result["trainable_params"] = trainable

    result_llm = None
    if use_llm_fallback:
        result_llm = call_llm_for_analysis(script_text)
        result_llm = json.loads(result_llm)
        for k, v in result_llm.items():
            if result.get(k) in [None, False, []]:
                result[k] = v

    return {"result": result, "result_llm": result_llm}

# =======================================
# Agent 2: Data profiler
# =======================================
def agent2_data_profiler(base_path):
    data_profiler = {}

    # Tabular
    tabular_summary = {
        "data_type": "tabular",
        "num_samples": 0,
        "num_features": 0,
        "avg_sample_size_bytes": 0,
        "total_size_bytes_estimate": 0
    }
    tabular_files = [f for f in os.listdir(base_path) if f.endswith(('.csv', '.parquet', '.tsv', '.xlsx'))]
    for file in tabular_files:
        file_path = os.path.join(base_path, file)
        if file.endswith('.csv') or file.endswith('.tsv'):
            df = pd.read_csv(file_path)
        elif file.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            continue
        tabular_summary["num_samples"] += int(len(df))
        tabular_summary["num_features"] = int(df.shape[1])
        tabular_summary["avg_sample_size_bytes"] = float(df.memory_usage(deep=True).sum() / len(df) if len(df) > 0 else 0)
        tabular_summary["total_size_bytes_estimate"] += int(df.memory_usage(deep=True).sum())
    if tabular_summary["num_samples"] > 0:
        data_profiler["tabular"] = tabular_summary

    # Image
    image_summary = {
        "data_type": "image",
        "num_samples": 0,
        "input_shape": None,
        "avg_sample_size_bytes": 0,
        "total_size_bytes_estimate": 0
    }
    image_files = [f for f in os.listdir(base_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    img_sizes, img_bytes = [], []
    for file in image_files:
        file_path = os.path.join(base_path, file)
        with Image.open(file_path) as img:
            img_sizes.append(img.size)
            img_bytes.append(os.path.getsize(file_path))
    if img_sizes:
        w, h = img_sizes[0]
        with Image.open(os.path.join(base_path, image_files[0])) as _im:
            c = len(_im.getbands())
        image_summary["input_shape"] = [int(c), int(h), int(w)]
        image_summary["num_samples"] = int(len(img_sizes))
        image_summary["avg_sample_size_bytes"] = float(sum(img_bytes) / len(img_bytes))
        image_summary["total_size_bytes_estimate"] = int(sum(img_bytes))
        data_profiler["image"] = image_summary

    # Text
    text_summary = {
        "data_type": "text",
        "num_samples": 0,
        "avg_seq_len_tokens": 0,
        "max_seq_len_tokens": 0,
        "avg_sample_size_bytes": 0,
        "total_size_bytes_estimate": 0
    }
    text_files = [f for f in os.listdir(base_path) if f.lower().endswith(('.txt', '.md'))]
    seq_lengths, text_sizes = [], []
    for file in text_files:
        file_path = os.path.join(base_path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.strip().split()
                seq_lengths.append(len(tokens))
                text_sizes.append(len(line.encode('utf-8')))
    if seq_lengths:
        text_summary["num_samples"] = int(len(seq_lengths))
        text_summary["avg_seq_len_tokens"] = float(sum(seq_lengths) / len(seq_lengths))
        text_summary["max_seq_len_tokens"] = int(max(seq_lengths))
        text_summary["avg_sample_size_bytes"] = float(sum(text_sizes) / len(text_sizes))
        text_summary["total_size_bytes_estimate"] = int(sum(text_sizes))
        data_profiler["text"] = text_summary

    return data_profiler

# =======================================
# Training cost estimators
# =======================================
def estimate_training_cost(agent1_output, agent2_output, gpu_options):
    results = []
    total_data_bytes = sum(x.get("total_size_bytes_estimate", 0) for x in agent2_output.values())
    batch_size = agent1_output.get("batch_size")
    epochs = agent1_output.get("epochs")
    params_total = agent1_output.get("params_total")
    num_samples = sum(x.get("num_samples", 0) for x in agent2_output.values())

    steps = (num_samples * epochs) / max(1, batch_size)
    for gpu in gpu_options:
        mem_per_batch_gb = (params_total * 4 / 1e9) + (total_data_bytes * batch_size / 1e9)
        time_per_batch_hours = mem_per_batch_gb / gpu["vRAM"]
        total_time_hours = steps * time_per_batch_hours
        price_hour = float(gpu["Price_per_Hour"])
        total_cost = total_time_hours * price_hour
        results.append({
            "gpu": gpu["Item"],
            "total_time_hours": total_time_hours,
            "total_cost": total_cost
        })
    cheapest = min(results, key=lambda x: x["total_cost"])
    fastest = min(results, key=lambda x: x["total_time_hours"])
    results_sorted = sorted(results, key=lambda x: (x["total_cost"], x["total_time_hours"]))
    balanced = results_sorted[len(results_sorted)//2]
    return {"cheapest": cheapest, "fastest": fastest, "balanced": balanced}

def estimate_training_cost_v2(agent1_output, agent2_output):
    """
    FLOPs / TFLOPs based cost estimator.
    NO ASSUMPTIONS:
      - batch_size, params_total, epochs, num_samples MUST exist and be > 0.
      - If anything is invalid → return "Not able to recommend".

    Returns ONLY:
      cheapest, fastest, balanced
      each containing:
        est_time_hours, est_time_minutes, est_time_seconds,
        est_cost, cost_unit
        (all rounded to 5 decimal places)
    """
    import math

    ERROR_RESP = {
        "cheapest": {"message": "Not able to recommend"},
        "fastest": {"message": "Not able to recommend"},
        "balanced": {"message": "Not able to recommend"}
    }

    try:
        # -------- Extract values --------
        batch_size   = agent1_output.get("batch_size")
        params_total = agent1_output.get("params_total")
        epochs       = agent1_output.get("epochs")

        # Compute num_samples
        num_samples = sum(
            d.get("num_samples", 0)
            for d in (agent2_output or {}).values()
            if isinstance(d, dict)
        )

        # -------- Validate required inputs --------
        if (
            batch_size is None or batch_size <= 0 or
            params_total is None or params_total <= 0 or
            epochs is None or epochs <= 0 or
            num_samples is None or num_samples <= 0
        ):
            return ERROR_RESP

        # Total dataset bytes (needed for load overhead)
        total_dataset_bytes = sum(
            d.get("total_size_bytes_estimate", 0)
            for d in (agent2_output or {}).values()
            if isinstance(d, dict)
        )

        # -------- GPU specs --------
        gpu_specs = [
            {"name": "NVIDIA H200",       "vram": 141, "vcpus": 30, "ram": 375, "price_hour": 3.49, "tflops": 120e12},
            {"name": "NVIDIA H100",       "vram": 80,  "vcpus": 26, "ram": 250, "price_hour": 2.90, "tflops": 60e12},
            {"name": "NVIDIA A100-80GB",  "vram": 80,  "vcpus": 16, "ram": 115, "price_hour": 3.39, "tflops": 40e12},
            {"name": "NVIDIA A100-40GB",  "vram": 40,  "vcpus": 16, "ram": 115, "price_hour": 2.55, "tflops": 20e12},
            {"name": "NVIDIA A40",        "vram": 48,  "vcpus": 16, "ram": 100, "price_hour": 1.44, "tflops": 14e12},
        ]

        # -------- FLOPs estimation --------
        flops_per_batch = 3.0 * float(params_total) * float(batch_size)
        steps_per_epoch = max(1, int(math.ceil(num_samples / float(batch_size))))
        total_flops     = flops_per_batch * steps_per_epoch * epochs

        # -------- Data loading time --------
        memory_bw = 900e9  # ~900 GB/s
        data_load_time_sec = float(total_dataset_bytes) / memory_bw

        results = []

        for gpu in gpu_specs:
            gpu_time_sec = float(total_flops) / float(gpu["tflops"]) + data_load_time_sec

            # convert time
            sec = round(gpu_time_sec, 5)
            mins = round(sec / 60.0, 5)
            hrs = round(sec / 3600.0, 5)

            # convert cost
            cost_usd_raw = (gpu_time_sec / 3600.0) * float(gpu["price_hour"])
            cost_usd_round = round(cost_usd_raw, 5)

            # Convert to cents if rounded USD hits zero
            if cost_usd_round == 0.0 and cost_usd_raw > 0.0:
                cost_value = round(cost_usd_raw * 100.0, 5)
                cost_unit = "cents"
            else:
                cost_value = cost_usd_round
                cost_unit = "USD"

            results.append({
                "gpu": gpu["name"],
                "est_time_hours": hrs,
                "est_time_minutes": mins,
                "est_time_seconds": sec,
                "est_cost": cost_value,
                "cost_unit": cost_unit
            })

        if not results:
            return ERROR_RESP

        # -------- Pick outputs --------
        cheapest = min(results, key=lambda x: x["est_cost"])
        fastest  = min(results, key=lambda x: x["est_time_seconds"])
        balanced = min(
            results,
            key=lambda x: (x["est_cost"] * max(x["est_time_seconds"], 1e-9))
        )

        return {
            "cheapest": cheapest,
            "fastest": fastest,
            "balanced": balanced
        }

    except Exception:
        return ERROR_RESP


# =======================================
# Classical-ML refinements for estimation
# =======================================
def _infer_linear_params(num_features: int, num_classes: int = 2, fit_intercept: bool = True):
    if num_features is None or num_features <= 0:
        return None
    if num_classes is None or num_classes < 2:
        num_classes = 2
    k = 1 if num_classes == 2 else num_classes
    weights = num_features * k
    bias = (k if fit_intercept else 0)
    return int(weights + bias)

def _infer_tree_params(n_estimators: int = 100, max_depth: int | None = None):
    if not n_estimators or n_estimators <= 0:
        n_estimators = 100
    if max_depth and max_depth > 0:
        nodes_per_tree = (2 ** (max_depth + 1)) - 1
        params_per_node = 6
        per_tree = nodes_per_tree * params_per_node
    else:
        per_tree = 1000
    return int(n_estimators * per_tree)

def _choose_epochs_for_classical(agent1_result):
    for key in ("max_iter", "n_estimators", "iterations"):
        v = agent1_result.get(key)
        if isinstance(v, int) and v > 0:
            return v
    return 1

def _choose_batch_size_for_classical(num_samples: int | None):
    if num_samples and num_samples > 0:
        return int(num_samples)
    return 1

def refine_classical_model_estimates(agent1_result: dict, data_profile: dict | None) -> dict:
    r = dict(agent1_result or {})
    framework = (r.get("framework") or "").lower()
    if not any(k in framework for k in ("sklearn", "xgboost", "lightgbm", "catboost")):
        return r

    num_samples = None
    num_features = None
    if data_profile and "tabular" in data_profile:
        num_samples = data_profile["tabular"].get("num_samples")
        num_features = data_profile["tabular"].get("num_features")

    if r.get("epochs") in (None, 0):
        r["epochs"] = _choose_epochs_for_classical(r)

    if r.get("batch_size") in (None, 0):
        r["batch_size"] = _choose_batch_size_for_classical(num_samples)

    params_total = r.get("params_total")
    if params_total in (None, 0):
        model_class = (r.get("model_class") or "").lower()
        if any(x in model_class for x in ("logistic", "ridge", "lasso", "sgd", "linear", "svm", "svc", "svr")):
            est = _infer_linear_params(num_features=num_features, num_classes=2, fit_intercept=True)
            r["params_total"] = est if est else 50_000
        elif any(x in model_class for x in ("randomforest", "gradientboost", "xgb", "lgbm", "lightgbm", "catboost")) or "xgboost" in framework or "lightgbm" in framework:
            n_est = r.get("n_estimators") or 100
            mdep = r.get("max_depth")
            r["params_total"] = _infer_tree_params(n_estimators=n_est, max_depth=mdep)
        else:
            r["params_total"] = 50_000
        if r.get("trainable_params") in (None, 0):
            r["trainable_params"] = r["params_total"]

    comment = r.get("comment") or ""
    comment += "; refined classical estimates (epochs/batch_size/params_total)"
    r["comment"] = comment.strip("; ")
    return r

# =======================================
# DEMO (guarded)
# =======================================
if __name__ == "__main__":
    ist_time = datetime.now(ZoneInfo("Asia/Kolkata"))
    print(ist_time.strftime("%Y-%m-%d %H:%M:%S"))
    if os.path.exists("train.py"):
        result_agent1 = analyze_script_source_with_hf("train.py")
        print(result_agent1)
