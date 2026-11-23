# LLaMA 3.2 1B ExecuTorch Edge Runtime

This directory contains the runtime binaries, libraries, and tokenizer configuration required to execute on-device LLaMA 3.2 1B inference on ARM64 Linux platforms, specifically optimized for the Raspberry Pi 5 (Quad-core ARM Cortex-A76 @ 2.4 GHz) via Meta ExecuTorch and XNNPACK.

---

## Directory Contents

| File | Size | Description |
| :--- | :--- | :--- |
| `llama_main` | 7.16 MB | Pre-compiled ARM64 (Linux aarch64) C++ executable runner for ExecuTorch with XNNPACK acceleration. |
| `libllama_runner.so` | 4.43 MB | Dynamic shared library dependency providing ExecuTorch operator kernels and runtime interfaces. |
| `tokenizer.json` | 17.2 MB | Hugging Face Byte-Pair Encoding (BPE) tokenizer vocabulary and merges for LLaMA 3.2 1B Instruct. |
| `fine_tune_and_export_int4.ipynb` | 31 KB | Automated Google Colab notebook for QLoRA fine-tuning and INT4 SpinQuant ExecuTorch export. |
| `generate_qc_llama_dataset.py` | 17 KB | Standalone script generating 1,200 domain-specific instruction samples across 4 materials and 20 defect classes. |
| `README.md` | -- | Runtime deployment and execution guide. |
| `FINETUNING.md` | -- | Detailed technical report on QLoRA fine-tuning, INT4 SpinQuant quantization, and ExecuTorch export. |

---

## Model Weights (`model_int4.pte`)

Due to GitHub's 100 MB single-file limit, the fine-tuned INT4 ExecuTorch model (`model_int4.pte`, 789.3 MB) is hosted externally.

### Download Link
The model file is accessible at the following Google Drive repository:

**URL:** https://drive.google.com/drive/folders/1pvlX7Eze1BcTVn0lEq0u6UkBroa4ebnA?usp=drive_link

### Download Instructions

1. Download `model_int4.pte` from the link above.
2. Verify the file size (approximately 827,646,336 bytes / 789.3 MB).
3. Place `model_int4.pte` directly inside this `llama/` directory or within `pi_deployment/`:
   ```bash
   cp /path/to/downloaded/model_int4.pte llama/
   ```

Alternatively, you can use `gdown` on Linux / Raspberry Pi:
```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1pvlX7Eze1BcTVn0lEq0u6UkBroa4ebnA?usp=drive_link
mv model_int4.pte llama/
```

---

## Fine-Tuning and INT4 Export Notebook

The end-to-end training and quantization workflow is self-contained in the provided Jupyter notebook:
- [`fine_tune_and_export_int4.ipynb`](fine_tune_and_export_int4.ipynb)

### Pipeline Stages in the Notebook
1. **Environment Setup & Google Drive Storage:** Mounts Google Drive and configures Unsloth, Hugging Face TRL/PEFT, and Meta ExecuTorch dependencies.
2. **Domain Dataset Synthesis:** Generates 1,200 balanced instruction-tuning pairs across four inspection domains (Hot-Rolled Steel Strip, Ceramic Tile, Glass Bottle, and Wire Mesh Grid) with 20 defect classes.
3. **QLoRA Fine-Tuning:** Loads `Llama-3.2-1B-Instruct-bnb-4bit` with Rank $r=16$, $\alpha=16$ on all linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), trained for 150 steps using cosine learning rate scheduling ($2 \times 10^{-4}$).
4. **16-Bit Weight Merging:** Merges LoRA adapter deltas back into FP16 Hugging Face model format (`/content/llama_3_2_1b_qc_custom`).
5. **INT4 SpinQuant & ExecuTorch Compilation:** Quantizes weights via `optimum-cli export executorch` using `8da4w` dynamic linear quantization and `4w` embedding quantization with the XNNPACK kernel backend, producing the standalone `model_int4.pte` binary (~789 MB).

### Standalone Dataset Generation
To re-generate the 1,200 instruction dataset locally or on a server without running the full notebook:
```bash
python llama/generate_qc_llama_dataset.py
```
This writes `industrial_qc_llama_dataset.json` and `.jsonl` into `llama/data/`.

---

## Hardware Deployment on Raspberry Pi 5

### 1. Prerequisites
Ensure the target system has the necessary shared library dependencies installed:
```bash
sudo apt update
sudo apt install -y libgomp1 libatomic1
```

### 2. Permissions and Library Path
Grant execute permissions to the binary and configure the runtime library search path:
```bash
chmod +x llama/llama_main
export LD_LIBRARY_PATH=$PWD/llama:$LD_LIBRARY_PATH
```

### 3. POSIX RAM-Disk Optimization (`/dev/shm`)
To avoid slow random read speeds and flash wear on MicroSD storage, stage the model in RAM disk:
```bash
cp llama/model_int4.pte /dev/shm/
cp llama/tokenizer.json /dev/shm/
```

### 4. Standalone CLI Verification
Execute the model directly using the C++ runner:
```bash
./llama/llama_main \
  --model_path /dev/shm/model_int4.pte \
  --tokenizer_path /dev/shm/tokenizer.json \
  --prompt "Defect: crack on ceramic tile. Assessment:" \
  --max_new_tokens 32 \
  --temperature 0.0 \
  --cpu_threads 4
```

Expected output speed on Raspberry Pi 5 Cortex-A76 cores: **7.88 to 7.95 tokens/second**.

---

## Integration with Industrial QC Pipeline

The runtime is integrated into the web dashboard and inspection service through `pi_deployment/backend/llama_engine.py`.

The engine automatically:
1. Discovers `llama_main`, `model_int4.pte`, and `tokenizer.json`.
2. Stages the `.pte` model into `/dev/shm` upon initialization.
3. Formats compact prompts using the Neuro-Fuzzy triage verdict.
4. Prefills the assistant header (`- Severity: {SEVERITY}`) to eliminate hallucination and save 6 to 8 decoding steps.
