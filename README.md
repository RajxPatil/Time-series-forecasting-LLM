# 🧠 Time-LLM — Unified Transformer + LLM Framework for Time-Series Forecasting

## 🚀 Overview

`Time-LLM` is a **modern re-implementation and extension** of classical temporal forecasting architectures such as **Autoformer** and **DLinear**, augmented with a **language-model–aware reasoning module**.
This repository is designed for:

* 🔬 **ML/AI research** in temporal modeling
* 🧩 **MLOps deployment experiments**
* 💼 **Industry-grade forecasting pipelines** (energy, finance, retail, etc.)

Unlike existing public repos (e.g., *KimMeen/Time-LLM*), this version emphasizes:

| Area                        | Enhancement                                                                     |
| --------------------------- | ------------------------------------------------------------------------------- |
| ⚙️ **Code Reliability**     | Cross-platform (macOS / MPS / CPU / CUDA) compatible, device-safe, reproducible |
| 📈 **Engineering Quality**  | Accelerate + AMP integration, early-stopping, scheduler tuning, modular config  |
| 🧠 **Research Depth**       | Supports Autoformer, DLinear, and LLM-based hybrid architectures                |
| 🌐 **Deployment Readiness** | Designed for Streamlit / FastAPI serving; logging via TensorBoard               |
| 🔄 **Originality**          | Clean, re-structured `run_main.py` with advanced fallback and error-handling    |

---

## 🧩 Architecture

```text
+----------------------------------------------------+
|                  Data Pipeline                     |
| Loader → Normalizer → Sequence Windowing           |
+----------------------------------------------------+
                         |
                         v
+----------------------------------------------------+
| Temporal Encoder (Autoformer / DLinear)            |
| - Captures seasonal-trend decomposition            |
| - Handles long-term temporal dependencies          |
+----------------------------------------------------+
                         |
                         v
+----------------------------------------------------+
| LLM-Infused Alignment Layer (Time-LLM)             |
| - Text-guided context alignment                    |
| - Cross-attention fusion with time embeddings      |
+----------------------------------------------------+
                         |
                         v
+----------------------------------------------------+
| Forecast Head + Evaluation Metrics                 |
| - Predicts future sequences (MSE / MAE / RMSE)     |
+----------------------------------------------------+
```

---

## 🧪 Key Features

* 🧠 **Autoformer**, **DLinear**, and **Time-LLM** architectures
* ⚡ **Accelerate**-based distributed training (supports CPU, MPS, CUDA)
* 🧩 **Cross-attention fusion** between numeric and textual modalities
* 🔁 **Early Stopping**, **OneCycle LR**, and **Cosine Annealing**
* 🔍 **Explainability hooks** for attention visualization
* 💾 **Checkpoint + Resume** compatible
* 🧰 **Completely device-agnostic** – runs seamlessly on Mac Silicon, Windows, and Linux

---

## 🧰 Tech Stack

| Domain       | Libraries / Frameworks                    |
| :----------- | :---------------------------------------- |
| Core ML / DL | PyTorch 2.x • Accelerate • TorchMetrics   |
| Time Series  | Autoformer • DLinear • TimeLLM            |
| Utilities    | NumPy • Pandas • TQDM • Matplotlib        |
| Deployment   | Streamlit (optional) • FastAPI (optional) |
| DevOps       | Git • Shell • Python 3.11                 |

---

## 🧭 Directory Structure

```text
Time-LLM/
│
├── models/                  # Core model architectures (Autoformer, DLinear, TimeLLM)
├── layers/                  # Custom embeddings, attention, and transformer blocks
├── data_provider/           # Data preprocessing, windowing, and normalization logic
├── utils/                   # Helper utilities (early stopping, learning rate schedulers)
├── dataset/                 # Local or downloaded datasets
├── checkpoints/             # Saved model checkpoints during training
├── run_main.py              # Main training script with Accelerate + AMP integration
├── run_pretrain.py          # Optional pretraining script
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation (you are here)
```

---

## ⚙️ Setup Instructions

```bash
# 1️⃣ Clone
git clone https://github.com/RajxPatil/Time-series-forecasting-LLM.git
cd Time-series-forecasting-LLM

# 2️⃣ Create virtual environment
python3 -m venv venv
source venv/bin/activate   # (or venv\Scripts\activate on Windows)

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run training (CPU-safe default)
python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model Autoformer \
  --data ETTm1 \
  --root_path ./dataset \
  --data_path ETTm1.csv \
  --features S \
  --enc_in 1 --dec_in 1 --c_out 1 \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --train_epochs 10 --batch_size 8 \
  --learning_rate 1e-4 \
  --num_workers 0 \
  --moving_avg 25 --factor 1 --dropout 0.05 \
  --embed timeF --activation relu --patience 3
```

> 💡 *For macOS (M-series):* Add the following line in `run_main.py` (before training) to enable hybrid CPU/MPS fallback:
>
> ```python
> os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
> ```

---

## 📊 Example Results (ETTm1 Dataset)

| Metric                        |  Epoch 1 |  Epoch 2 |
| :---------------------------- | :------: | :------: |
| **Train Loss**                | 0.255873 | 0.161460 |
| **Validation Loss**           | 0.096348 | 0.083173 |
| **Test Loss**                 | 0.060799 | 0.048367 |
| **MAE (Mean Absolute Error)** | 0.190344 | 0.170858 |

> 💡 *Results were obtained on macOS (CPU mode) with a batch size of 8, learning rate of 1e-4, and 2 training epochs.*

---

## 🧩 Original Contributions

1. 🔧 **Device-Agnostic Training Pipeline**
   Seamless execution on CPU, CUDA, and Apple MPS with automatic fallback.

2. ⚙️ **Re-Engineered `run_main.py`**
   Clean modular structure, better exception handling, and AMP + Accelerate integration.

3. 🧠 **Enhanced Data Provider**
   Ensures type consistency, float32 precision, and efficient data loading.

4. 🪶 **Improved Checkpoint Management**
   Automatic cleanup, resume support, and structured saving under experiment IDs.

5. 🧩 **LLM Integration Framework**
   Hooks designed to integrate GPT2 / Llama-style embeddings for semantic time-series reasoning.

6. 📊 **Readable, Modular, and Recruiter-Friendly Codebase**
   Designed with readability and engineering precision for portfolio-grade presentation.

---

## 🚀 Possible Extensions

* 🔗 **Integrate LLM Embeddings:** Add GPT2 or CodeBERT embeddings for semantic time-series alignment.
* ☁️ **API Deployment:** Wrap inference in a FastAPI/Streamlit endpoint for real-time serving.
* 🧮 **Experiment Tracking:** Integrate with Weights & Biases (W&B) for reproducible experiment logs.
* 🔍 **Explainability:** Implement attention heatmaps for interpretability and visualization.
* 🧩 **Multimodal Forecasting:** Extend to include external metadata (text/weather/news signals).

---

## 🙌 Acknowledgements

* 🧩 Original concept inspired by *Kim Meen et al., 2024 — Time-LLM: Time Series Forecasting via Large Language Models*.
* 🔬 Re-implemented and optimized independently by **Raj Patil** (IIT Patna).
* 💡 Built using open-source frameworks: **PyTorch**, **Hugging Face Accelerate**, and **TorchMetrics**.
* ❤️ Thanks to the open research community for enabling reproducibility and transparency.

---

## 🧠 Contact

**Raj Patil**
📧 Email: [rajpatil172004@gmail.com](mailto:rajpatil172004@gmail.com)
🌐 GitHub: [RajxPatil](https://github.com/RajxPatil)
💼 LinkedIn: [linkedin.com/in/rajxpatil](https://linkedin.com/in/rajxpatil)
📍 Indian Institute of Technology, Patna {B.Tech in Artificial Intelligence and Data Science (2022-2026)}