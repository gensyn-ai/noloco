# NoLoCo: No-all-reduce Low Communication Training Method for Large Models

This repository implements __NoLoCo__, a novel optimization method designed to reduce communication overhead during the training of large language models. By eliminating explicit parameter synchronization, NoLoCo achieves faster convergence rates and reduced idling time compared to existing methods.

## 🔗 Paper

The full paper is available on [arXiv](https://arxiv.org/abs/2506.10911). 

- HuggingFace paper : 
<a href="https://huggingface.co/papers/2506.10911?utm_source=chatgpt.com" target="_blank">
  <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face" width="20"/>
</a>

- Tech Blog: https://www.gensyn.ai/



## ⚙️ Features

- **No collective communication**: Eliminates the need for explicit parameter synchronization.
- **Faster convergence**: Achieves up to 4% faster convergence compared to existing methods.
- **Scalable**: Benchmarked on models ranging from 125M to 6.8B parameters.
- **Efficient**: Reduces communication overhead and accelerator idling time.


## 🚀 Installation

Clone this repository and install the required dependencies. (You will need a machine(s) equipped with Nvidia GPUs.)

```bash
git clone https://github.com/yourusername/NoLoCo.git
cd NoLoCo
pip install .
```

## 🧪 Usage

### Step 1. Prepare Configs
Start by preparing the YAML configuration files. Example configs can be found under `configs/no_reduce/C4`.

Use `c4_base.yaml` as a template, and make the following adjustments:

- Add your **HuggingFace access token** to enable downloading the LLaMA 3 tokenizer.
- Ensure that your token has the necessary permissions for accessing the tokenizer.
- Set the appropriate path to your **local copy** of the HuggingFace C4 English partition in the data loader section.

Example data loader configuration:

```yaml
train_data_loader:
  _target_: dipaco.c4_data_loader.TokenizedHuggingFaceShardedDataset
  partition: train
  tokenizer_path: *tokenizer
  sequence_length: *sequence_length
  access_token: *access_token
  path: *path_local_c4_en_copy
```
📝 Note: Ensure that the path_local_c4_en_copy points to a pre-downloaded and properly structured C4 dataset directory on your machine or storage system.


### Step 2: Data Preparation

**C4 Dataset:**  
The C4 dataset is processed on the fly and requires no additional preprocessing beyond loading the files from Git. Simply ensure that your local copy is available and correctly referenced in the configuration.

**Reddit Dataset:**  
The Reddit data must be **pre-tokenized** by the user prior to training. To prepare it:

1. Download the HuggingFace [Pushshift Reddit dataset](https://huggingface.co/datasets/fddemarco/pushshift-reddit-comments).
2. Tokenize the text documents using your model tokenizer (e.g., LLaMA 3).
3. Save the tokenized outputs as **Parquet** files containing `input_ids` and `attention_mask` fields.

The expected data format matches the input format used in [gensyn-ai/hdee](https://github.com/gensyn-ai/hdee), so you can refer to that repository for an exact schema and preprocessing example.

### Step 3. Run Traning Job
Navigate to the `script` folder and modify `run.sh` to specify the configuration file you want to use (e.g., `c4_diloco_100m_8.yaml`):
```bash
#!/bin/bash

source ~/.profile
ROOT=$PWD

torchrun \
    --nnodes=1 \
    --nproc-per-node=8 \
    --rdzv-id=dipaco.run \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$ROOT/src/dipaco/train.py" \
    --config-path "$ROOT/configs/C4" \
    --config-name c4_diloco_100m_8.yaml
```

🧠 Note:

- The total number of training processes is determined by `nnodes × nproc-per-node`. In the example above, it's `1 × 8 = 8`.

- To launch training across multiple nodes, run the same script on each participating node.

- Set the environment variables `MASTER_ADDR` and `MASTER_PORT` to point to a single designated node (e.g., the first node), using an IP address and port that are reachable from all other nodes. For local runs (e.g. single node) you can use `localhost` as `MASTER_ADDR` and `29501` as the `MASTER_PORT` (`localhost:29501`)

- Make sure SSH or other networking between nodes is properly configured and accessible before launching.


## 📊 Results

NoLoCo has been evaluated across a wide range of model sizes and accelerator configurations.

- Models ranging from **125M** to **6.8B** parameters
- Benchmarked on **various node counts and GPU configurations**
- Compared against DiLoCo and Fully Sharded Data Parallel (FSDP)

Key findings:
- Up to **4% faster convergence** compared to DiLoCo.
- No global blocking communication.
- Speedup compared to all-reduce scales as **log(N)**; for 1024 GPUs, this translates to a 10× improvement over standard all-reduce.

Detailed benchmarking results and graphs are available in the [paper](https://arxiv.org/abs/2506.10911).

## 📄 Citation

If you use this code or method in your research, please cite the following paper:

```bibtex
@article{Kolehmainen2025NoLoCo,
  title={NoLoCo: No-all-reduce Low Communication Training Method for Large Models},
  author={Jari Kolehmainen and Nikolay Blagoev and John Donaghy and Oğuzhan Ersoy and Christopher Nies},
  journal={arXiv preprint arXiv:2506.10911},
  year={2025}
}
```

## 📬 Contact

For questions, suggestions, or collaborations, please contact the authors via jari@gensyn.ai or open an issue on this GitHub repository.

