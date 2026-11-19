# Image Captioning with Vision-Language Models (CS 6350, TPA-19)

Fine-tuning **GIT-base** and **BLIP-base** on MS-COCO 2017 for image captioning, with Optuna hyperparameter search, evaluation on 5,000 held-out images (BLEU-4, METEOR, ROUGE-L, CIDEr), and extensions: model comparison, attention maps, prompt-conditioned captioning and zero-shot captioning on our own photos.

**Authors:** Iniyan V N (ME23B170), Vansh Garg (ME23B210)  
**Course:** CS 6350, IIT Madras

---

## Important: datasets and model checkpoints are NOT included

This repository contains the source code, documentation, results, report, slides and reference papers required to reproduce our experiments.

The following files are intentionally excluded from the repository:

| Not included | Reason | How to obtain |
|---|---|---|
| MS-COCO train2017 images | Dataset size and licence | Download separately |
| MS-COCO val2017 images | Dataset size and licence | Download separately |
| COCO caption annotations | Dataset size and licence | Download separately |
| Generated train/validation split files | Generated automatically | Run the data preparation stage |
| Fine-tuned GIT model checkpoints and tensors | Large file sizes | Run the GIT training pipeline |
| Fine-tuned BLIP model checkpoints and tensors | Large file sizes | Run the BLIP training pipeline |

**You must run the training pipeline to obtain the fine-tuned model weights.** The trained model tensors are saved locally in the `checkpoints/` directory and are not uploaded to GitHub.

The pretrained base models are downloaded automatically from Hugging Face when required. You do not need to manually upload or provide pretrained model weights.

All evaluation and inference stages should be run using the checkpoints generated through training.

## Data setup

Download MS-COCO 2017 from https://cocodataset.org/#download :

- Train images: http://images.cocodataset.org/zips/train2017.zip
- Validation images: http://images.cocodataset.org/zips/val2017.zip
- Annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip

Extract the files into the following structure:

```text
coco_data/
├── train2017/
├── val2017/
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

The split files `captions_train_split.json` and `captions_val_split.json` are generated automatically by the `prepare_data` stage (90% training / 10% tuning-validation, split by image ID, seed 42).

**Data roles:**

- `train2017` is split into training (90%) and tuning-validation (10%).
- Optuna uses only the tuning-validation split for hyperparameter selection.
- `val2017` (5,000 images, approximately five reference captions each) is the final test set and is never used for training or tuning.

## Pretrained models

The following pretrained models are downloaded automatically from the Hugging Face Hub:

- `microsoft/git-base` – not fine-tuned on COCO captions.
- `microsoft/git-base-coco` – pretrained on COCO.
- `Salesforce/blip-image-captioning-base` – pretrained on COCO.

The pretrained weights are downloaded when the corresponding model is loaded. Fine-tuned weights are generated locally by running the training stages.

## Installation

```bash
pip install torch torchvision transformers optuna nltk rouge-score pycocoevalcap pillow matplotlib

python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"
```

A CUDA-enabled GPU is strongly recommended. One GIT-base epoch of 100,000 caption pairs took approximately 58 minutes on our GPU. BLIP-base is considerably slower (approximately 6 hours per epoch).

## Repository contents

| Path | Purpose |
|---|---|
| `GIT.py` | Self-contained GIT pipeline: data preparation, Optuna tuning, training, evaluation, qualitative analysis, attention maps, prompted captioning, zero-shot and model comparison |
| `BLIP.ipynb` | BLIP-base training and evaluation pipeline |
| `outputs/` | Experimental results, predictions, metrics and figures |
| `image_dir/` | Photographs used for the zero-shot experiment |
| `report/` | IEEE-format project report and figures |
| `slides/` | Project presentation |
| `papers/` | Reference papers |
| `README.md` | Project documentation |

The following directories are generated locally and are not tracked by Git:

- `coco_data/` – downloaded dataset and generated split files.
- `checkpoints/` – pretrained/fine-tuned model checkpoints and tensors saved by the training pipeline.

## Usage (GIT)

After downloading and extracting the dataset, execute the following stages in order.

```bash
# 1. Generate the 90/10 training and tuning-validation splits
python GIT.py --stage prepare_data

# 2. Evaluate the pretrained model as a baseline
python GIT.py --stage evaluate --checkpoint microsoft/git-base

# 3. (Optional) Perform Optuna hyperparameter search
python GIT.py --stage tune --n_trials 16

# 4. Fine-tune GIT-base for one epoch
# This generates the model tensors/checkpoints locally
python GIT.py --stage train --run_name git-base-1ep --epochs 1 --max_train_samples 100000

# 5. Evaluate the locally generated checkpoint
python GIT.py --stage evaluate --checkpoint ./checkpoints/git-base-1ep/epoch0

# 6. Perform qualitative analysis
python GIT.py --stage qualitative --checkpoint ./checkpoints/git-base-1ep/epoch0

# 7. Run zero-shot captioning on your own photographs
python GIT.py --stage zero_shot --checkpoint ./checkpoints/git-base-1ep/epoch0 --image_dir ./image_dir
```

### Usage from a notebook

```python
import GIT

# Train the model and generate checkpoints
GIT.train(run_name="git-base-1ep", num_epochs=1)

# Evaluate the trained model
GIT.evaluate("./checkpoints/git-base-1ep/epoch0")
```

Every training run saves its checkpoints to a separate directory:

```text
checkpoints/
└── git-base-1ep/
    └── epoch0/
        └── ... model files ...
```

For BLIP, open `BLIP.ipynb` and execute the training and evaluation stages in order. Its fine-tuned checkpoints are also generated locally.

**Note:** Evaluation using a fine-tuned checkpoint requires completing the corresponding training stage first. The checkpoint paths above are generated by training and are not available immediately after cloning the repository.

## Method in brief

- **Objective:** Next-token cross-entropy over non-padding caption tokens (padding labels are -100). For BLIP, the first decoder input is the `[DEC]` token (ID 30522). For GIT, loss is computed manually from text-position logits because the built-in loss in our Transformers version mis-shifted the labels (7.79 versus the correct 1.76 on the untuned model).
- **Optimisation:** AdamW, linear warm-up and decay, mixed precision, 40-token captions, seed 42, beam search with 4 beams at inference.
- **Tuning:** Optuna (TPE) over learning rate, batch size {8, 16} and warm-up ratio, 200 steps per trial, scored on 500 tuning-validation captions.
- **Metrics:** BLEU-4 (NLTK, smoothing method 4), METEOR (NLTK), ROUGE-L (`rouge-score`), CIDEr (`pycocoevalcap`). We did not use the PTB tokenizer, so absolute values are not comparable to published leaderboards; all models here are evaluated using the same code on the same 5,000 images.

## Results (5,000 val2017 test images)

| Model | Setting | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|
| GIT-base | Untuned | 0.1437 | 0.3061 | 0.4410 | 0.5333 |
| GIT-base | Fine-tuned, 1 epoch | 0.3120 | 0.5482 | 0.6152 | 1.0239 |
| GIT-base | Fine-tuned, 3 epochs | 0.3068 | 0.5447 | 0.6120 | 1.0132 |
| GIT-base-coco | Untuned | 0.3189 | 0.5490 | 0.6208 | 1.0400 |
| GIT-base-coco | Fine-tuned, 1 epoch | 0.3142 | 0.5482 | 0.6159 | 1.0305 |
| GIT-base-coco | Fine-tuned, 3 epochs | 0.3070 | 0.5447 | 0.6119 | 1.0087 |
| BLIP-base | Untuned | 0.2560 | 0.4696 | 0.5603 | 0.8496 |
| BLIP-base | Fine-tuned, 1 epoch | 0.2888 | 0.5294 | 0.5950 | 0.9457 |
| BLIP-base | Fine-tuned, 3 epochs | 0.2778 | 0.5261 | 0.5875 | 0.9143 |

GIT-base fine-tuned for one epoch is our primary model.

**Limitations:** One run per setting (single seed, no confidence intervals), and the pretrained checkpoints may have seen COCO images that overlap val2017 (not checked).

## References and sources

### Papers

- J. Li et al., "BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation," ICML 2022.
- J. Wang et al., "GIT: A generative image-to-text transformer for vision and language," arXiv:2205.14100, 2022.
- T.-Y. Lin et al., "Microsoft COCO: Common objects in context," ECCV 2014.
- X. Chen et al., "Microsoft COCO captions: Data collection and evaluation server," arXiv:1504.00325, 2015.
- R. Vedantam et al., "CIDEr: Consensus-based image description evaluation," CVPR 2015.
- T. Akiba et al., "Optuna: A next-generation hyperparameter optimization framework," KDD 2019.
- T. Wolf et al., "Transformers: State-of-the-art natural language processing," EMNLP 2020 (System Demonstrations).

### Code and data

- Salesforce BLIP: https://github.com/salesforce/BLIP
- Microsoft GenerativeImage2Text (GIT): https://github.com/microsoft/GenerativeImage2Text
- Hugging Face Transformers: https://github.com/huggingface/transformers
- Optuna: https://github.com/optuna/optuna
- pycocoevalcap: https://github.com/salesforce/pycocoevalcap
- rouge-score: https://github.com/google-research/google-research/tree/master/rouge
- NLTK: https://github.com/nltk/nltk
- MS-COCO 2017: https://cocodataset.org/#download

## AI tools used

- **Claude** by Anthropic (https://claude.ai) for coding help, debugging and report drafting.
- Hugging Face Transformers (https://huggingface.co) and Optuna (https://optuna.org) as libraries.

## Licence

Our code is released under [choose a licence, e.g. MIT]. MS-COCO images and annotations, and pretrained model weights, are governed by their respective licences; see the links above.

