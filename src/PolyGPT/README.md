### **README: GPT Fine-Tuning with Curve-Based Embeddings**

---

## **Project Overview**

This project explores a novel method for fine-tuning **GPT-2** by integrating **curve-based embeddings** derived from contextual word representations. The approach leverages **BERT** for generating word embeddings, which are transformed into **curve embeddings** that enhance the local context. These embeddings are then fed into GPT-2 for fine-tuning on a larger dataset, **WikiText-103**, to improve language modeling, generalization, and text generation.

---

## **Goals**

1. **Enhance Contextual Representation**:
   - Use **curve embeddings** to represent sentences, capturing richer local contexts by blending neighboring word embeddings.

2. **Integrate Custom Normalization**:
   - Replace standard softmax with **SoftEMax**, a custom normalization function, for improved stability and numerical robustness in the attention mechanism.

3. **Achieve Better Generalization**:
   - Fine-tune GPT-2 on **WikiText-103**, a large and diverse dataset, to enhance the model's performance on downstream tasks like text generation.

4. **Evaluate Effectiveness with Metrics**:
   - Measure the model's performance using **BLEU**, **Perplexity**, and **Token Accuracy**.

5. **Track Training Progress**:
   - Use **Weights & Biases (WandB)** for real-time logging of training and validation metrics.

---

## **Approach**

### **1. Curve-Based Embeddings**
- A custom embedding mechanism that computes **curve embeddings** for each word in a sentence:

- These embeddings:
  - Capture the **local context**.
  - Are normalized using **SoftEMax** for stability.
  - Are projected into the hidden space of GPT-2 and augmented with **positional encodings** for sequence modeling.

### **2. Custom Attention Mechanism**
- Replaces the standard attention normalization (softmax) with **SoftEMax** to improve numerical stability and computational efficiency.

### **3. Fine-Tuning**
- Fine-tune GPT-2 on **WikiText-103** using curve embeddings to improve text generation quality.

---

## **Key Features**

1. **Custom Normalization**:
   - **SoftEMax** replaces softmax in the attention mechanism for stable scaling.

2. **Integration with Pretrained Models**:
   - Uses **BERT** for generating word embeddings.
   - Fine-tunes **GPT-2** with curve embeddings for enhanced language modeling.

3. **Metrics for Evaluation**:
   - **BLEU**: Evaluates the quality of generated text.
   - **Perplexity**: Measures the model's uncertainty in text generation.
   - **Token Accuracy**: Assesses how accurately the model predicts the correct tokens.

4. **Dataset**:
   - **WikiText-103**: A large-scale dataset for training and validation.

5. **Logging with WandB**:
   - Tracks training and validation losses, BLEU, Perplexity, and Token Accuracy.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/bryn-gnolbs/polynet.git
   cd gpt-polycurve-finetune
   ```

2. Install dependencies:
   ```bash
   pip install torch transformers datasets wandb nltk
   ```

3. Set up WandB:
   ```bash
   wandb login
   ```

---

## **Usage**

1. **Train the Model**:
   ```bash
   python main.py
   ```

2. **Monitor Training**:
   - Visit your WandB project page to view real-time training metrics.

3. **Evaluate the Model**:
   - After training, validation metrics like BLEU, Perplexity, and Token Accuracy will be logged to WandB.

4. **Saved Model**:
   - The fine-tuned model is saved to `./Models/fine_tuned_model.pth`.

---

## **Evaluation Metrics**

1. **BLEU**:
   - Measures the similarity between generated and reference text.
   - Logged as `bleu_score` in WandB.

2. **Perplexity**:
   - Indicates how well the model predicts text.
   - Lower perplexity signifies better performance.
   - Logged as `perplexity`.

3. **Token Accuracy**:
   - Fraction of correctly predicted tokens.
   - Logged as `token_accuracy`.

---

## **Project Structure**

```
gpt-polycurve-finetune/
├── main.py                # Main training script
├── Models/                # Directory for saving fine-tuned models
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## **Future Work**

1. **Dataset Expansion**:
   - Incorporate additional datasets for multilingual and cross-domain fine-tuning.

2. **Custom Loss Functions**:
   - Experiment with advanced loss functions to better handle rare tokens.

3. **Real-World Applications**:
   - Test on downstream tasks like question answering, summarization, and dialogue generation.

---

## **Acknowledgments**
- **Transformers Library**: For pretrained models and utilities.
- **Weights & Biases**: For tracking and visualizing training progress.
- **WikiText-103**: For providing the training and validation datasets.

---

Feel free to contribute or suggest improvements! For questions or collaborations, contact `help@genolabs.net`. 

--- 