## Comparative Analysis in Speech Recognition with Transformers: Whisper-Tiny for Indonesian

### Overview
This project focuses on a comparative analysis of speech recognition algorithms using Transformer-based models. The study explores the performance of the Whisper-tiny model, both in its base form and after fine-tuning, specifically for recognizing Indonesian speech.

### Project Objectives
The project aims to:

Enhance understanding of Transformer model systematics, including fine-tuning implementations.

Compare base and fine-tuned Whisper-tiny models based on relevant factors.

Identify the most suitable model for diverse datasets through comparative analysis and modifications.

Assess the relevance of fine-tuning for specific tasks.

### Preprocessing
Whisper requires log-mel spectrograms as input. Raw audio (MP3 files at 4000 Hz) is preprocessed to a 16000 Hz sample rate using Whisper’s feature extractor and tokenizer, ensuring compatibility without additional standardization.

### Model Design
Whisper-tiny, with 39,000 parameters, 4 layers, 384 width, 6 heads, and a max learning rate of 1.5e-3, uses an encoder/decoder Transformer with sequence-to-sequence learning. It processes 30-second audio clips into log-mel spectrograms, applies Gaussian Error Linear Unit (GELU) activation, and uses sinusoidal positional encoding for sequence awareness. Multi-head self-attention layers and feed-forward networks enhance audio representation.

### Training
Training employs a Seq2SeqTrainingArguments setup with Whisper-tiny, a DataColatorSpeechSeq2SeqWithPadding class for batch processing, and WER as the evaluation metric. The process includes forward/backward passes, gradient computation, and periodic model saving based on performance.

### Evaluation and Analysis
After 1000 training steps, results show:

| Model Name      | WER     |
| :-------------- | :------ |
| `Whisper-tiny`  | `63.0%` |
| `Whisper-tiny-finetuned`  | `27.0%` |
| `Whisper-base`  | `41.0%` |
| `Whisper-small`  | `20.0%` |

Fine-tuning significantly reduces WER compared to the pre-trained model, though it lags behind the small model. This suggests fine-tuning improves accuracy for specific languages like Indonesian by adapting to unique linguistic characteristics.

### Conclusion and Recommendations
Whisper-tiny struggles with language detection due to its multitasking nature, but fine-tuning with Indonesian-specific data improves performance. For specialized language tasks, fine-tuning is recommended. For multilingual applications, a base model may suffice despite lower accuracy.

### Attachments
HuggingFace Model: ChronoStellar/whisper-tiny-id

Datasets:

Common Voice 11.0

Common Voice 17.0

### References
Raffel, A., et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision." International Conference on Machine Learning, pp. 24092-25118, PMLR.
