import torch
import torch.nn as nn
import math


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=24, output_lstm=128, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, output_lstm, num_layers=1, batch_first=True)
        self.ln1 = torch.nn.LayerNorm(input_dim)

        self.fc1 = torch.nn.Linear(output_lstm, output_lstm)
        self.drop1 = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(output_lstm, output_lstm // 2)
        self.drop2 = torch.nn.Dropout(dropout)
        self.fc3 = torch.nn.Linear(output_lstm // 2, 2)

        self.fc_nn = torch.nn.Sequential(
            self.fc1,
            self.drop1,
            self.relu,
            self.fc2,
            self.drop2,
            self.relu,
            self.fc3
        )

    def forward(self, x):
        x = self.ln1(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc_nn(lstm_out[:, -1, :])
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 256, dropout: float = 0.1, max_len: int = 30):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerNet(nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder.
    """

    def __init__(
            self,
            seq_len=30,
            input_dim=24,
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            num_layers=4,
            dropout=0.1,
            activation="relu",
            classifier_dropout=0.1,
            channel_attention=False
    ):

        super().__init__()
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        # self.emb = nn.Embedding(input_dim, d_model)
        self.channel_attention = channel_attention

        self.lin_time = nn.Linear(input_dim, d_model)
        self.lin_channel = nn.Linear(seq_len, d_model)

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout
        )

        encoder_layer_time = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        #data processing over time
        self.transformer_encoder_time = nn.TransformerEncoder(
            encoder_layer_time,
            num_layers=num_layers,
        )

        encoder_layer_channel = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        #data processing over feature dimensions
        self.transformer_encoder_channel = nn.TransformerEncoder(
            encoder_layer_channel,
            num_layers=num_layers,
        )

        self.out_time = nn.Linear(d_model, d_model)
        self.out_channel = nn.Linear(d_model, d_model)

        self.lin = nn.Linear(d_model * 2, 2)

        if self.channel_attention:
            self.classifier = nn.Linear(d_model * 2, 2)
        else:
            self.classifier = nn.Linear(d_model, 2)

        self.d_model = d_model

    def resh(self, x, y):
        return x.unsqueeze(1).expand(y.size(0), -1)

    def forward(self, x_):

        x = torch.tanh(self.lin_time(x_)) #[seq_len, batch_size, d_model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder_time(x)
        x = x[0, :, :]

        if self.channel_attention:
            y = torch.transpose(x_, 0, 2)
            y = torch.tanh(self.lin_channel(y))
            y = self.transformer_encoder_channel(y)

            x = torch.tanh(self.out_time(x))
            y = torch.tanh(self.out_channel(y[0, :, :]))

            h = self.lin(torch.cat([x, y], dim=1))

            m = nn.Softmax(dim=1)
            g = m(h)

            g1 = g[:, 0]
            g2 = g[:, 1]

            x = torch.cat([self.resh(g1, x) * x, self.resh(g2, x) * y], dim=1)

        x = self.classifier(x)

        return x

class MLP(nn.Module):
    def __init__(self, input_dim=24, output_dim=2, hidden_dims=[128, 64], dropout=0.1, activation='relu'):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dims in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dims))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dims

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleGRU(nn.Module):
    def __init__(self, input_dim=24, output_gru=128, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(input_dim, output_gru, num_layers=1, batch_first=True)
        self.ln1 = torch.nn.LayerNorm(input_dim)

        self.fc1 = torch.nn.Linear(output_gru, output_gru)
        self.drop1 = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(output_gru, output_gru // 2)
        self.drop2 = torch.nn.Dropout(dropout)
        self.fc3 = torch.nn.Linear(output_gru // 2, 2)

        self.fc_nn = torch.nn.Sequential(
            self.fc1,
            self.drop1,
            self.relu,
            self.fc2,
            self.drop2,
            self.relu,
            self.fc3
        )

    def forward(self, x):
        x = self.ln1(x)
        lstm_gru, _ = self.gru(x)
        x = self.fc_nn(lstm_gru[:, -1, :])
        return x

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, seq_len=30, num_features=24, dim=128, dropout=0.5):
        super(SimpleCNN, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()

        conv_output_h = seq_len // 4
        conv_output_w = num_features // 4
        linear_input_size = 32 * conv_output_h * conv_output_w

        self.fc = nn.Linear(linear_input_size, 2)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)


        x = self.dropout(x)
        return self.fc(x)

class GRN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.gate = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )

        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        residual = self.skip(x)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        gated = self.gate(x)
        x = gated * x
        return self.layer_norm(x + residual)


class TFTNet(nn.Module):
    def __init__(self, input_dim=12, static_dim=12, seq_len=30, d_model=128, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len

        self.dynamic_proj = nn.Linear(input_dim, d_model)
        self.static_proj = nn.Linear(static_dim, d_model)

        # add Gated Residual Network to TFT
        self.static_grn = GRN(static_dim, hidden_dim=d_model, output_dim=d_model, dropout=dropout)
        self.dynamic_grn = GRN(input_dim, hidden_dim=d_model, output_dim=d_model, dropout=dropout)

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2*d_model,
            dropout=dropout,
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.final_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, dynamic, static):
        dyn = self.dynamic_proj(dynamic)                       # [B, T, D]
        stat = self.static_proj(static).unsqueeze(1)           # [B, 1, D]
        stat = stat.expand(-1, self.seq_len, -1)               # [B, T, D]

        x = dyn + stat

        x = torch.transpose(x, 0, 1)                            # → [T, B, D]
        x = self.pos_encoder(x)
        x = self.transformer(x)                                 # [T, B, D]
        x = x[-1]
        out = self.final_fc(x)                                  # [B, 2]
        return out

class PositionalEncoding_TFT(nn.Module):
    def __init__(self, d_model=128, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [B, T, D]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, num_inputs, hidden_dim):
        super().__init__()
        self.flattened_grns = nn.ModuleList([
            GRN(input_dim, hidden_dim) for _ in range(num_inputs)
        ])
        self.weight_network = nn.Sequential(
            nn.Linear(input_dim * num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_inputs),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x: [B, T, N, D] or [B, N, D] for static
        if x.dim() == 4:
            B, T, N, D = x.size()
            x_reshaped = x.view(B * T, N, D)
        else:
            B, N, D = x.size()
            x_reshaped = x

        var_outputs = [grn(x_reshaped[:, i, :]) for i, grn in enumerate(self.flattened_grns)]
        var_outputs = torch.stack(var_outputs, dim=1)  # [B*T or B, N, D]
        flattened_input = x_reshaped.reshape(B * T if x.dim() == 4 else B, -1)
        weights = self.weight_network(flattened_input).unsqueeze(-1)  # [B*T or B, N, 1]
        weighted = (weights * var_outputs).sum(dim=1)  # [B*T or B, D]
        if x.dim() == 4:
            return weighted.view(B, T, -1), weights.view(B, T, N)
        else:
            return weighted, weights

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_dim, static_dim, seq_len, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len

        self.static_encoder = GRN(static_dim, hidden_dim=d_model, output_dim=d_model)
        self.static_context_variable_selection = GRN(static_dim, hidden_dim=d_model, output_dim=d_model)

        self.static_context_h = GRN(d_model, hidden_dim=d_model, output_dim=d_model)  # für h0
        self.static_context_c = GRN(d_model, hidden_dim=d_model, output_dim=d_model)  # für c0

        self.dynamic_proj = nn.Linear(input_dim, d_model)
        self.static_proj = nn.Linear(static_dim, d_model)

        self.positional_encoding = PositionalEncoding_TFT(d_model, max_len=seq_len)
        self.vsn_dynamic = VariableSelectionNetwork(d_model, num_inputs=input_dim, hidden_dim=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.local_lstm_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True
        )

        self.local_lstm_decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True
        )

        self.post_grn = GRN(d_model, hidden_dim=d_model, output_dim=d_model)
        self.output_layer = nn.Linear(d_model, 2)

    def forward(self, dynamic_inputs, static_inputs):
        # static_inputs: [B, static_dim]
        # dynamic_inputs: [B, T, input_dim]
        B, T, _ = dynamic_inputs.size()

        static_context = self.static_encoder(static_inputs)  # [B, d_model]
        expanded_static = static_context.unsqueeze(1).expand(-1, T, -1)  # [B, T, d_model]

        dynamic_proj = self.dynamic_proj(dynamic_inputs)  # [B, T, d_model]
        dynamic_combined, _ = self.vsn_dynamic(dynamic_proj.unsqueeze(2).expand(-1, -1, dynamic_inputs.shape[2], -1))
        dynamic_contextual = dynamic_combined + expanded_static  # [B, T, d_model]

        # Initial states from static context
        h0 = self.static_context_h(static_context).unsqueeze(0)  # [1, B, d_model]
        c0 = self.static_context_c(static_context).unsqueeze(0)  # [1, B, d_model]

        # LSTM Encoder and Decoder
        encoder_out, _ = self.local_lstm_encoder(dynamic_contextual, (h0, c0))
        decoder_out, _ = self.local_lstm_decoder(encoder_out, (h0, c0))

        # Positional Encoding and Transformer
        x = self.positional_encoding(decoder_out)
        x = x.transpose(0, 1)  # [T, B, d_model]
        x = self.transformer_encoder(x)

        x = x[-1]  # last time step [B, d_model]
        x = self.post_grn(x)
        out = self.output_layer(x)  # [B, 2]
        return out
