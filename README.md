# Whisper-Tiny-id

Project README: Comparative Analysis in Speech Recognition with Transformers
Overview
This project, conducted at BINUS University during the 2023/2024 academic semester, focuses on a comparative analysis of speech recognition algorithms using Transformer-based models. The study explores the performance of the Whisper-tiny model, both in its base form and after fine-tuning, specifically for recognizing Indonesian speech.
Background
The world is advancing toward Industry 4.0, characterized by the dominance of sophisticated technologies that simplify complex tasks. Artificial Intelligence (AI), emerging in the 1950s and evolving significantly in the 21st century, offers effective solutions to complex problems. Speech Recognition, a key AI branch, enables machines to interpret human speech based on audio data, forming a critical area of this research.
Problem Statement
The evolution of speech recognition models raises several questions:

Which model (base Transformer or fine-tuned version) provides a better solution for existing challenges?
What factors should be considered to determine the most relevant model?
Are fine-tuning or model modifications necessary to address specific tasks?
What implementation steps can enhance model performance with varied audio data?

Project Objectives
The project aims to:

Enhance understanding of Transformer model systematics, including fine-tuning implementations.
Compare base and fine-tuned Whisper-tiny models based on relevant factors.
Identify the most suitable model for diverse datasets through comparative analysis and modifications.
Assess the relevance of fine-tuning for specific tasks.

Scope
Due to the impracticality of analyzing all speech recognition models, this project narrows its focus to comparing the base Whisper-tiny model with its fine-tuned counterpart, both utilizing Transformer architecture. The comparison evaluates their ability to recognize Indonesian speech.
Dataset Description
The project utilizes two datasets from Mozilla Common Voice:

Training and Validation: Common Voice 11.0.
Testing: Common Voice 17.0.These datasets, publicly available, support 12 languages, but only Indonesian is used here. The choice aligns with Whisper’s training on 600,000+ hours of multilingual audio, including Common Voice and FLEURS, ensuring supervised learning consistency for Indonesian.

Experimental Phases

Import libraries (transformers, datasets, accelerate, evaluate, gradio) for model training and UI development.
Load and split Mozilla Common Voice 11.0 and 17.0 into training, testing, and validation sets.
Load Whisper-tiny model with tokenizer, feature extractor, and processor.
Process datasets to match model input requirements.
Define trainer functions and optimize parameters.
Train the model with optimized parameters.
Upload the model to Hugging Face for pipeline access.
Evaluate using Word Error Rate (WER) on the test dataset.

Preprocessing
Whisper requires log-mel spectrograms as input. Raw audio (MP3 files at 4000 Hz) is preprocessed to a 16000 Hz sample rate using Whisper’s feature extractor and tokenizer, ensuring compatibility without additional standardization.
Model Design
Whisper-tiny, with 39,000 parameters, 4 layers, 384 width, 6 heads, and a max learning rate of 1.5e-3, uses an encoder/decoder Transformer with sequence-to-sequence learning. It processes 30-second audio clips into log-mel spectrograms, applies Gaussian Error Linear Unit (GELU) activation, and uses sinusoidal positional encoding for sequence awareness. Multi-head self-attention layers and feed-forward networks enhance audio representation.
Training
Training employs a Seq2SeqTrainingArguments setup with Whisper-tiny, a DataColatorSpeechSeq2SeqWithPadding class for batch processing, and WER as the evaluation metric. The process includes forward/backward passes, gradient computation, and periodic model saving based on performance.
Evaluation and Analysis
After 100 training steps, results show:

Training loss: 0.53%
Validation loss: 0.58%
Word Error Rate (WER): 38.8%Fine-tuning significantly reduces WER compared to the pre-trained model, though it lags behind the small model. This suggests fine-tuning improves accuracy for specific languages like Indonesian by adapting to unique linguistic characteristics.

Conclusion and Recommendations
Whisper-tiny struggles with language detection due to its multitasking nature, but fine-tuning with Indonesian-specific data improves performance. For specialized language tasks, fine-tuning is recommended. For multilingual applications, a base model may suffice despite lower accuracy.
Attachments

HuggingFace: https://huggingface.co/ChronoStellar/whisper-tiny-id 
Datasets:
Common Voice 11.0
Common Voice 17.0


References: Raffel, A., et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision." International Conference on Machine Learning, pp. 24092-25118, PMLR.
