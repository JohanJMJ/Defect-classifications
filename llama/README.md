# LLaMA 3.2 1B ExecuTorch Edge Runtime

This directory contains the lightweight runtime files and tokenizer required to execute **LLaMA 3.2 1B Instruct** locally on the **Raspberry Pi 5** (ARM Cortex-A76 CPU) via **Meta ExecuTorch**.

---

## 📦 Included Files

| File | Size | Description |
| :--- | :--- | :--- |
| **`llama_main`** | ~6.8 MB | Pre-compiled ARM64 (Linux aarch64) C++ executable runner for ExecuTorch with XNNPACK engine support. |
| **`libllama_runner.so`** | ~4.2 MB | Dynamic shared library dependency for the `llama_main` ExecuTorch runtime on ARM64 Linux. |
| **`tokenizer.json`** | ~16.4 MB | Fast tokenizer vocabulary and BPE merges for LLaMA 3.2 1B Instruct. |

---

## ⬇️ Model Weights (`model_int4.pte`)

Due to GitHub's **100 MB** single-file upload limit, the fine-tuned, quantized model file (`model_int4.pte`, **~789.3 MB**) is hosted externally:

🔗 **Google Drive Download Link**: *[INSERT YOUR GOOGLE DRIVE LINK HERE]*

### How to obtain the model:
1. **Option A (Direct Download)**: Download `model_int4.pte` from the Google Drive link above.
2. **Option B (Re-export via Notebook)**: Run the self-contained Google Colab notebook [`fine_tune_and_export_int4.ipynb`](../fine_tune_and_export_int4.ipynb). It automatically fine-tunes LLaMA 3.2 1B with Unsloth QLoRA on 1,200 industrial QC QA pairs and exports the INT4 `.pte` model directly to your Google Drive.

---

## 🚀 Setup on Raspberry Pi 5

When deploying on the Raspberry Pi 5, copy or link these files into the `pi_deployment/` directory:

```bash
# Ensure execution permissions on the C++ runner
chmod +x llama/llama_main

# Copy runtime binaries and tokenizer to the deployment folder
cp llama/llama_main pi_deployment/
cp llama/libllama_runner.so pi_deployment/
cp llama/tokenizer.json pi_deployment/

# Place the downloaded model_int4.pte into pi_deployment/
# (e.g. from USB drive or Google Drive download)
cp /path/to/downloaded/model_int4.pte pi_deployment/
```

Once in place, the pipeline automatically detects and mirrors `model_int4.pte` and `tokenizer.json` into `/dev/shm` (POSIX RAM disk) for ultra-fast execution (~6–7 tok/s on ARM Cortex-A76).
