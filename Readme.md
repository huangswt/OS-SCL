# OS-SCL

**Noise Supervised Contrastive Learning and Feature-Perturbed for Anomalous Sound Detection**

This project is designed for equipment sound classification and anomaly detection using supervised contrastive learning and feature perturbation. Follow the steps below to prepare data, train the model, and evaluate performance.

------

## 📦 Step 1: Download the Dataset

Run the following command to automatically download and extract the dataset:

```bash
bash datadownload.sh
```

- The dataset will be downloaded and extracted to the designated directory specified in the script.

------

## 🏷️ Step 2: Label the Evaluation Test Dataset

To label the evaluation dataset:

1. Copy the following files from the `Preparation` folder into the `eval_dataset` directory:

   ```
   fan.csv
   pump.csv
   slider.csv
   split.py
   ToyCar.csv
   ToyConveyor.csv
   valve.csv
   ```

2. Run the `split.py` script to generate the evaluation labels:

   ```bash
   python split.py
   ```

- This step ensures that the evaluation set has the necessary labels for performance assessment.

------

## 🤪 Step 3: Run Main Experiment

Start training the model by running:

```bash
python train.py --m 0.4 --gpu_num <GPU_ID> --fussion 1 --ht basic --desc main
```

- Replace `<GPU_ID>` with the index of the GPU to use.
- Parameters:
  - `--m`: margin value for arcface loss.
  - `--fussion`: use fusion method 1.
  - `--ht`: activation function type.
  - `--desc`: description tag for this experiment.

------

## 🧹 Step 4: Prune Trained Model Parameters (Optional)

After training, prune the model to remove non-essential training modules:

```bash
python prune_model.py --input <MODEL_PATH>
```

To overwrite the original model with the pruned one:

```bash
python prune_model.py --input <MODEL_PATH> --overwrite
```

- This keeps only the inference-related weights by removing components such as `pre_block4`, `spec_augmenter`, and others.
- Benefits: reduced model size and faster loading for deployment.

------

## 📊 Step 5: Evaluate the Model

### ✅ Evaluate on the Development Test Set:

```bash
python eval.py --m 0.4 --gpu_num <GPU_ID> --fussion 1 --ht basic --model_path <MODEL_PATH> --d
```

### ✅ Evaluate on the Evaluation Test Set:

```bash
python eval.py --m 0.4 --gpu_num <GPU_ID> --fussion 1 --ht basic --model_path <MODEL_PATH> --e
```

- Replace `<GPU_ID>` with your GPU index.
- Replace `<MODEL_PATH>` with the path to the trained (and optionally pruned) model checkpoint.

------

## 🙏 Acknowledgements

We sincerely thank the following repositories and authors for their valuable contributions:

- [STgram-MFN](https://github.com/liuyoude/STgram-MFN)
- [Noisy-ArcMix](https://github.com/soonhyeon/Noisy-ArcMix)
- [Dr. Jiang Anbai](https://github.com/jianganbai)

------

## 📚 Citation

If you use this work in your research, please cite the following paper:

```bibtex
@INPROCEEDINGS{10888995,
  author={Huang, Shun and Fang, Zhihua and He, Liang},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Noise Supervised Contrastive Learning and Feature-Perturbed for Anomalous Sound Detection}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Training;Time-frequency analysis;Source coding;Perturbation methods;Contrastive learning;Production;Feature extraction;Stability analysis;Noise measurement;Speech processing;Anomalous sound detection;self-supervised learning;supervised contrastive learning},
  doi={10.1109/ICASSP49660.2025.10888995}
}
```
