import torch
import torchvision.models as models


class SimpleDNN(torch.nn.Module):
    def __init__(
            self,
            num_input_features: int,
            num_output_features: int,
            layers: int = 3,
            hidden_size: int = 64,
        ):
        super().__init__()
        
        # create layers
        layer_list = [
            torch.nn.Linear(num_input_features, hidden_size),
            torch.nn.ReLU(),
        ]
        for _ in range(1, layers - 1):
            layer_list.append(torch.nn.Linear(hidden_size, hidden_size))
            layer_list.append(torch.nn.ReLU())
        layer_list.append(torch.nn.Linear(hidden_size, num_output_features))
        
        # Stack all layers together in a sequential container
        self.model = torch.nn.Sequential(*layer_list)
    
    def forward(self, x):
        return self.model(x)


class SimpleLSTM(torch.nn.Module):
    def __init__(
            self,
            window_size: int,
            num_input_features: int,
            num_output_features: int,
            hidden_size: int = 64
        ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size = num_input_features,
            hidden_size = hidden_size,
            num_layers = 3,
            batch_first = True,
        )
        self.flatten = torch.nn.Flatten(start_dim = 1)
        self.dense = torch.nn.Linear(hidden_size * window_size, num_output_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape=(batch_size, window_size, num_input_features)
        Returns:
            x (torch.Tensor): shape=(batch_size, num_output_features)
        """
        x, (_, _) = self.lstm(x)  # shape = (batch_size, window_size, hidden_size)
        x = self.flatten(x)  # shaoe = (batch_size, window_size * hidden_size)
        x = self.dense(x)
        return x


class PositionalEncoding(torch.nn.Module):
    """
    PE(pos, i)
        where pos = 0, 1, 2, ..., window_size - 1
        and i = 0, 1, 2, ..., num_input_features - 1
    
    for i = even -> sin(pos / 10000 ** (2 * i / num_input_features))
    for i = odd  -> cos(pos / 10000 ** (2 * i / num_input_features))

    Notes:
        + window_size = content length = sequence length
        + num_input_features = d_model = encoding size = embedding size
    """
    def __init__(self, num_input_features: int, max_len: int = 5000):
        """
        max_len has same axis as window_size. that means it supports all window_size within 1~5000.
        """
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)   # shape = (max_len, 1). i.e. [[0],[1],[2],...,[5000]]
        div_term = torch.exp(torch.arange(0, num_input_features, 2) * (-math.log(10000.0) / num_input_features))
        # div_term (divisor_term): shape = (num_features // 2,)
        # e.g. torch.arange(0, num_input_features=4, 2) = [0, 2]
        # e.g. div_term = exp([0, 2] * (-log(10000) / 4))
        # = [exp(-0 / 4 * log(10000)), exp(-2 / 4 * log(10000))]
        # = [10000 ** -0 / 4, 10000 ** -2 / 4]
        # = [1 / 10000 ** 0 / 4, 1 / 10000 ** 2 / 4]
        
        pe = torch.zeros(max_len, num_input_features)
        pe[:, 0::2] = torch.sin(position * div_term)   # shape = (max_len, num_input_features) = (5000, 4)
        pe[:, 1::2] = torch.cos(position * div_term)   # shape = (max_len, num_input_features) = (5000, 4)
        
        self.register_buffer('pe', pe)  # remain at cpu and non-trainable
    

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape = (batch_size, window_size, features)
        Returns:
            x (torch.Tensor): same shape as above
        """
        window_size = x.shape[1]
        return x + self.pe[:window_size, :].unsqueeze(0)


class ToyTransformerEncoder(torch.nn.Module):
    def __init__(self, window_size: int, num_input_features: int, num_output_features: int):
        super().__init__()
        self.positional_encoding = PositionalEncoding(num_input_features)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model = num_input_features,
            nhead = 2,
            dim_feedforward = 64,
            batch_first = True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.flatten = torch.nn.Flatten(start_dim = 1)
        self.dense = torch.nn.Linear(num_input_features * window_size, num_output_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape = (batch_size, window_size, num_input_features)
        Returns:
            x (torch.Tensor): shape = (batch_size, num_output_features)
        """
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class SimpleCNN(torch.nn.Module):
    def __init__(
            self,
            input_c: int,
            input_h: int,
            input_w: int,
            num_output_features: int,
            hidden: int = 64,
        ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=input_c, out_channels=hidden, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=3, stride=1, padding=1)
        self.dense1 = torch.nn.Linear(input_h * input_w * hidden // 4 ** 2, num_output_features)

        self.relu   = torch.nn.ReLU()
        self.maxpool= torch.nn.MaxPool2d(2)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):  # e.g. (B, 1, 28, 28)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (B, 16, 14, 14)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (B, 16, 7, 7)
        x = self.flatten(x)  # (B, 784)
        x = self.dense1(x)  # (B, 2)
        return x


class ResNet18(torch.nn.Module):
    def __init__(self, num_output_features: int):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        in_features = self.resnet18.fc.in_features
        self.backbone.fc = torch.nn.Linear(in_features, num_output_features)

    def forward(self, x):  # (B, 3, h, w)
        return self.resnet18(x)  # (B, 2)


class ViT_b_16(torch.nn.Module):
    def __init__(self, num_output_features: int):
        super().__init__()
        self.vit = models.vit_b_16(pretrained=True)
        self.dense = torch.nn.Linear(1000, num_output_features)
        
    def forward(self, x):  # (B, 3, h, w)
        x = self.vit(x)  # (B, 10)
        x = self.dense(x)
        return x
