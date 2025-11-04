import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input: [batch_size]
        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size], tgt: [tgt_len, batch_size]
        tgt_len = tgt.shape[0]
        batch_size = src.shape[1]
        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = tgt[0, :]  # <sos>

        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = tgt[t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs

    # 🧩 Thêm hàm greedy_decode để inference mà không cần tgt
    def greedy_decode(self, src, max_len=50, sos_idx=2, eos_idx=3):
        with torch.no_grad():
            hidden, cell = self.encoder(src)
            batch_size = src.shape[1]
            outputs = torch.zeros(max_len, batch_size, self.decoder.output_dim).to(self.device)

            input = torch.tensor([sos_idx] * batch_size).to(self.device)
            for t in range(max_len):
                output, hidden, cell = self.decoder(input, hidden, cell)
                outputs[t] = output
                top1 = output.argmax(1)
                input = top1
                if (top1 == eos_idx).all():
                    break
            return outputs
