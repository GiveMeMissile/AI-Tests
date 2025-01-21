import tkinter as tk
import math
import torch
from torch import nn
from tkinter import scrolledtext
from transformers import AutoTokenizer

MODELS_FOLDER = "Depression_detecting_models"
MODELS = ["Model_1.pth", "test_model.pth"]
TOKENIZER_NAME = "bert-base-uncased"
MAX_LENGTH = 64


class PositionalEncoding(nn.Module):
    def __init__(self, dimensions, dropout, max_length):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pos_encoding = torch.zeros(max_length, dimensions)
        positions_list = torch.arange(0, max_length, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dimensions, 2).float() * (-math.log(10000.0))/dimensions)

        pos_encoding[:, 0::2] = torch.sin(positions_list*division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x):
        return self.dropout(x + self.pos_encoding[:x.size(0), :])


class TransformerModel(nn.Module):
    def __init__(self, num_tokens, hidden_features, num_heads, num_layers, dropout, output):
        super(TransformerModel, self).__init__()

        self.hidden_features = hidden_features

        self.positional_encoder = PositionalEncoding(hidden_features, dropout, max_length=MAX_LENGTH)

        self.embedding = nn.Embedding(num_tokens, hidden_features)
        encoder = nn.TransformerEncoderLayer(
            d_model=hidden_features,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_features, output)

    def forward(self, x, attention_mask=None):
        x = self.embedding(x.long()) * math.sqrt(self.hidden_features)
        x = self.positional_encoder(x)

        x = x.permute(1, 0, 2)

        if attention_mask is not None:
            attention_mask = attention_mask.permute(1, 0)
            attention_mask = attention_mask.to(dtype=torch.bool)

        x = self.transformer(x, src_key_padding_mask=attention_mask)

        x = self.output_layer(x.mean(0))
        return x


class DepressionDetector:
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self, model_path):
        try:
            self.model = torch.load(MODELS_FOLDER+"/"+model_path)
        except Exception:
            self.model = torch.load(MODELS_FOLDER+"/"+MODELS[0])

        self.model.eval()
        self.model.to(self.device)

    def prepare_tokens(self, tokens):
        pass

    def predict(self, tokens):
        with torch.inference_mode():
            tokens = tokens.to(self.device)
            output = self.model(tokens)
            output = ((torch.sigmoid(output)).item())/tokens.shape[0]
            
            # convert to percentage
            output = output * 100
            return output


class Application(tk.Frame):
    def __init__(self, tokenizer, detector, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Depression Detector")
        self.master.geometry("800x600")
        self.create_widgets()
        self.pack()
        self.selected_var = tk.StringVar()
        self.selected_model = None
        self.detector = detector
        self.tokenizer = tokenizer

    def create_widgets(self):
        self.text = scrolledtext.ScrolledText(self.master, wrap=tk.WORD, width=60, height=20)
        self.text.pack()

        self.button = tk.Button(self.master, text="Check", command=self.check)
        self.button.pack()

        self.result = tk.Label(self.master, text="")
        self.result.pack()

    def check(self):
        text = self.text.get("1.0", tk.END)
        tokens = self.tokenizer.tokenize(text)
        tokens = self.tokenizer.normalize_tokens(tokens)
        output = self.detector.predict(tokens)
        self.result.config(text=f"Depression percentage: {output:.2f}%")

    def select_model(self, model_path):
        root = tk.Tk()
        root.title("Select model")

        label = tk.Label(root, text="Select a model for prediction (default is model001.pth)")
        label.pack()

        for models in model_path:
            radio_button = tk.Radiobutton(
                root, 
                text=models, 
                value=self.selected_var, 
                variable=models)
            radio_button.pack()
            self.model_path = tk.StringVar()
        
        submit_button = tk.Button(
            root, 
            text="Submit desired model", 
            command=self.get_selected_model
            )
        submit_button.pack()
        return self.selected_model
    
    def get_selected_model(self):
        selected_model = self.selected_var.get()
        self.selected_model = selected_model


class Tokenizer:
    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(self, text):
        tokens = self.tokenizer(text)
        tokens = tokens["input_ids"]
        tokens = torch.tensor(tokens)
        return tokens

    def __len__(self):
        return len(self.tokenizer)

    def normalize_tokens(self, tokens):
        # This function is used to normalize the tokens to the max length
        # So they can be processed by the model.

        batch = []
        max_alter = 0
        i = 0
        while True:
            i += 1
            if tokens.shape[0] == MAX_LENGTH:
                batch.append(tokens)
                break
            if tokens.shape[0] < MAX_LENGTH:
                zeros = torch.zeros(MAX_LENGTH - tokens.shape[0])
                tokens_batch = torch.cat([tokens, zeros])
                batch.append(tokens_batch)
                break
            else:
                tokens_batch = tokens[:MAX_LENGTH]
                batch.append(tokens_batch)
                tokens = tokens[max_alter+MAX_LENGTH:]
                max_alter += MAX_LENGTH

        tokens = torch.stack(batch, dim=0)
        return tokens


def main():
    tokenizer = Tokenizer(TOKENIZER_NAME)
    model = TransformerModel(
        num_tokens=len(tokenizer),
        hidden_features=512,
        num_heads=8,
        num_layers=6,
        dropout=0.4,
        output=1
    )
    root = tk.Tk()
    detector = DepressionDetector()
    app = Application(tokenizer, detector, master=root)
    model = app.select_model(MODELS)
    detector.load_model(model)
    app.mainloop()


if __name__ == "__main__":
    main()