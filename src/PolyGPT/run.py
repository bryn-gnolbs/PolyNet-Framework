import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import wandb

def load_model(model_path: str, device: torch.device) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)  # type: ignore
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model # type: ignore

def generate_text(model: GPT2LMHeadModel, tokenizer, prompt: str, max_length: int = 50) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(model: GPT2LMHeadModel, tokenizer, dataset, device: torch.device):
    model.eval()
    total_loss = 0

    for step, batch in enumerate(dataset):
        input_ids, labels = batch
        input_ids, labels = input_ids.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataset)
    print(f"Evaluation Loss: {avg_loss}")
    wandb.log({"evaluation_loss": avg_loss})

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./Models/fine_tuned_model.pth"
    model = load_model(model_path, device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    prompt = "Once upon a time, in a land far, far away"
    generated_text = generate_text(model, tokenizer, prompt)
    print("Generated Text:")
    print(generated_text)

    # Optional: Load an evaluation dataset and evaluate the model
    # dataset = ... (implement your dataset loading logic here)
    # evaluate_model(model, tokenizer, dataset, device)

if __name__ == "__main__":
    main()
