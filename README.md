# Visual Sign Language Recognition Model

This document details the architecture and usage instructions for the visual sign language recognition model.

## Model Architecture

The model processes video frames through a series of layers to generate sign language predictions.

```text
Input: Video Frames [Batch x Time x Channels x Height x Width]
            │
            ▼
   ┌──────────────────────────────┐
   │     2D CNN Backbone (ResNet) │
   │    - Extracts frame features │
   │    - FC layer removed        │
   └──────────────────────────────┘
            │
            ▼
   Framewise Features [Batch x Channels x Time]
            │
            ▼
   ┌─────────────────────────────────────────┐
   │         TemporalConv Layer              │
   │ - 1D temporal convolutions              │
   │ - Input size: 512                       │
   │ - Hidden size: configurable (e.g. 1024) │
   └─────────────────────────────────────────┘
            │
            ├──────────────► conv_logits (used in ConvCTC loss and conv_sents)
            │
            ▼
   visual_feat [Time x Batch x Hidden]
            │
            ▼
   ┌─────────────────────────────────────────┐
   │         BiLSTM Temporal Layer           │
   │ - 2 layers                              │
   │ - Bidirectional                         │
   │ - Input/output: [Time x Batch x Hidden] │
   └─────────────────────────────────────────┘
            │
            ▼
   ┌──────────────────────────────┐
   │      Classification Layer    │
   │ - Linear or NormLinear       │
   │ - Output: num_classes logits │
   └──────────────────────────────┘
            │
            ├──────────────► sequence_logits (used in SeqCTC loss and recognized_sents)
            ▼
   ┌──────────────────────┐
   │   Beam Decoder       │
   │ - Decodes predictions│
   └──────────────────────┘
```
*Flow Diagram Simplified:*
```text
Video Frames
     │
     ▼
ResNet Backbone (Spatial Feat)
     │
     ▼
TemporalConv (Temporal Conv Feat)
     │
     ▼
BiLSTM (Temporal Context)
     │
     ▼
Classification (Predictions)
     │
     ▼
  Logits
```

## Preprocessing

**Important:** [vsl_preprocess.py](src/vsign/data/vsl_preprocess.py), [extract_frames.py](src/vsign/data/extract_frames.py) before running the following steps.

1.  **Extract Frames and Annotations:**
    ```bash
    python extract_frames.py
    ```

2.  **Extract Features:** (Resizes images, creates gloss dictionary, info file, and groundtruth)
    ```bash
    python /home/martinvalentine/Desktop/v-sign/src/vsign/data/vsl_preprocess.py --process-image --multiprocessing
    ```

## Training

Run the training script using the baseline configuration.

```bash
python run_baseline.py \
  --config ./configs/baseline.yaml \
  --device 0
```

## Evaluation:
Evaluate the model performance on the test set using saved weights.
```bash
python run_baseline.py \
  --config ./configs/baseline.yaml \
  --device 0 \
  --load-weights /home/martinvalentine/Desktop/v-sign/outputs/logs/baseline_res18/_best_model.pt \
  --phase test
```

## Demo:
Run the inference demo script with a trained model.
```bash
python src/vsign/inference/demo.py  \
   --model_path /home/martinvalentine/Desktop/v-sign/outputs/logs/baseline_res18/_best_model.pt     
   --device 0 \
   --dict_path /home/martinvalentine/Desktop/v-sign/data/processed/VSL_V0/gloss_dict.npy
```
