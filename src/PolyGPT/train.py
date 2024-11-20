import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, BertTokenizer, BertModel
from datasets import load_dataset
import wandb

wandb.init(
    project="gpt-polycurve-finetune",
    config={"epochs": 3, "batch_size": 32, "learning_rate": 1e-3},
)


def softemax(x, device="cuda"):
    x = x.to(device)
    e_x = torch.e**x
    summation = torch.sum(
        torch.exp(torch.linspace(0, torch.e, steps=int(torch.e), device=device))
    )
    return e_x / summation


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2)
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class CurveAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CurveAttention, self).__init__()
        self.hidden_size = hidden_size
        self.scale = hidden_size**-0.5

    def forward(self, query, key, value):
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weights = softemax(attn_scores)  # Replace softmax with softemax
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


class PolyCurveEmbedding(nn.Module):
    def __init__(self, embedding_dim, gpt_hidden_size):
        super(PolyCurveEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.project = nn.Linear(embedding_dim, gpt_hidden_size)
        self.positional_encoding = PositionalEncoding(gpt_hidden_size)

    def forward(self, embeddings):
        batch_size, seq_len, _ = embeddings.size()
        curve_embeddings = torch.zeros(batch_size, seq_len, self.embedding_dim).to(
            embeddings.device
        )
        for i in range(seq_len):
            prev = (
                embeddings[:, i - 1]
                if i > 0
                else torch.zeros(batch_size, self.embedding_dim).to(embeddings.device)
            )
            word = embeddings[:, i]
            nxt = (
                embeddings[:, i + 1]
                if i < seq_len - 1
                else torch.zeros(batch_size, self.embedding_dim).to(embeddings.device)
            )
            curve = 0.25 * prev + 0.5 * word + 0.25 * nxt
            curve_embeddings[:, i] = curve
        projected = self.project(curve_embeddings)
        return self.positional_encoding(projected)


class WikiText2CurveDataset(Dataset):
    def __init__(
        self, tokenizer, bert_model, poly_curve, texts, max_length=128, device="cpu"
    ):
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.poly_curve = poly_curve
        self.texts = texts
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            bert_embeddings = self.bert_model(**tokens).last_hidden_state
        curve_embeddings = self.poly_curve(bert_embeddings)
        input_curves = curve_embeddings.squeeze(0)[:-1]
        target_indices = tokens["input_ids"].squeeze(0)[1:]
        return input_curves, target_indices


class CurveGPTFineTuner:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def train(self, data_loader, optimizer, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for step, (input_curves, target_indices) in enumerate(data_loader):
                input_curves, target_indices = (
                    input_curves.to(self.device),
                    target_indices.to(self.device),
                )
                input_curves = input_curves[:, : self.model.config.n_positions]
                target_indices = target_indices[:, : self.model.config.n_positions]
                outputs = self.model(inputs_embeds=input_curves)
                logits = outputs.logits
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), target_indices.view(-1)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if step % 10 == 0:
                    wandb.log(
                        {
                            "train_loss_step": loss.item(),
                            "epoch": epoch + 1,
                            "step": step,
                        }
                    )
            avg_loss = total_loss / len(data_loader)
            wandb.log({"train_loss_epoch": avg_loss, "epoch": epoch + 1})
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for step, (input_curves, target_indices) in enumerate(data_loader):
                input_curves, target_indices = (
                    input_curves.to(self.device),
                    target_indices.to(self.device),
                )
                input_curves = input_curves[:, : self.model.config.n_positions]
                outputs = self.model(inputs_embeds=input_curves)
                logits = outputs.logits
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), target_indices.view(-1)
                )
                total_loss += loss.item()
                if step % 10 == 0:
                    wandb.log({"val_loss_step": loss.item(), "step": step})
        avg_loss = total_loss / len(data_loader)
        wandb.log({"val_loss_epoch": avg_loss})
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)  # type: ignore
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)  # type: ignore
    poly_curve = PolyCurveEmbedding(
        embedding_dim=768, gpt_hidden_size=gpt_model.config.hidden_size
    ).to(device)
    dataset = load_dataset("wikitext", "wikitext-2-v1")
    train_texts = dataset["train"]["text"]  # type: ignore
    val_texts = dataset["validation"]["text"]  # type: ignore
    train_texts = [text for text in train_texts if len(text.strip()) > 0]
    val_texts = [text for text in val_texts if len(text.strip()) > 0]
    train_dataset = WikiText2CurveDataset(
        bert_tokenizer, bert_model, poly_curve, train_texts, device=device # type: ignore
    )
    val_dataset = WikiText2CurveDataset(
        bert_tokenizer, bert_model, poly_curve, val_texts, device=device # type: ignore
    )
    train_loader = DataLoader(
        train_dataset, batch_size=wandb.config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=wandb.config.batch_size, shuffle=False
    )
    optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=wandb.config.learning_rate)
    fine_tuner = CurveGPTFineTuner(gpt_model, device)
    fine_tuner.train(train_loader, optimizer, epochs=wandb.config.epochs)
    fine_tuner.evaluate(val_loader)
    fine_tuner.save_model("./Models/fine_tuned_model.pth")


if __name__ == "__main__":
    main()
