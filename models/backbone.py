import torch
import torch.nn as nn

from .transformer import CustomizedTransformerEncoderLayer, CustomizedTransformerEncoder


class MTLModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, lstm_hidden_size=64, output_feature=32, num_layer=2, nhead=4):
        super(MTLModel, self).__init__()
        self.output_feature = output_feature
        self.hidden_dim = hidden_size
        self.lstm_hidden_size = lstm_hidden_size

        self.bn = nn.BatchNorm1d(input_size)
        self.input_fc = nn.Linear(input_size, hidden_size)

        self.bi_lstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size, num_layers=num_layer,
                               bidirectional=True, batch_first=False)
        self.transformer_encoder_layer = CustomizedTransformerEncoderLayer(d_model=lstm_hidden_size * 2, nhead=nhead,
                                                                           dropout=0.3)
        self.transformer_encoder = CustomizedTransformerEncoder(self.transformer_encoder_layer,
                                                                num_layers=num_layer)

        self.fc = nn.Linear(lstm_hidden_size * 2, output_feature)

        self.classify_head = nn.Linear(output_feature, 1)
        self.regression_head = nn.Linear(output_feature, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_normalized = self.bn(x)
        x_normalized = x_normalized.permute(2, 0, 1)
        x_embedded = self.input_fc(x_normalized)

        lstm_out, _ = self.bi_lstm(x_embedded)
        transformer_out = self.transformer_encoder(lstm_out)
        feature = self.fc(transformer_out[-1, :, :]).view(-1, self.output_feature)

        class_output = self.sigmoid(self.classify_head(feature))
        reg_output = self.regression_head(feature)
        return class_output, reg_output


class LSTMModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, output_feature=2, num_layer=2):
        super(LSTMModel, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_feature)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_normalized = self.bn(x)
        x_normalized = x_normalized.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x_normalized)
        output = self.fc(lstm_out[:, -1, :])
        return output


class RegModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, lstm_hidden_size=64, output_feature=32, num_layer=2, nhead=2):
        super(RegModel, self).__init__()
        self.output_feature = output_feature
        self.hidden_dim = hidden_size
        self.lstm_hidden_size = lstm_hidden_size

        self.bn = nn.BatchNorm1d(input_size)
        self.input_fc = nn.Linear(input_size, hidden_size)

        self.bi_lstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size, num_layers=num_layer,
                               bidirectional=True, batch_first=False)
        self.transformer_encoder_layer = CustomizedTransformerEncoderLayer(d_model=lstm_hidden_size * 2, nhead=nhead,
                                                                           dropout=0.3)
        self.transformer_encoder = CustomizedTransformerEncoder(self.transformer_encoder_layer,
                                                                num_layers=num_layer)

        self.fc = nn.Linear(lstm_hidden_size * 2, output_feature)

        self.regression_head = nn.Linear(output_feature, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_normalized = self.bn(x)
        x_normalized = x_normalized.permute(2, 0, 1)
        x_embedded = self.input_fc(x_normalized)

        lstm_out, _ = self.bi_lstm(x_embedded)
        transformer_out = self.transformer_encoder(lstm_out)
        feature = self.fc(transformer_out[-1, :, :]).view(-1, self.output_feature)

        reg_output = self.regression_head(feature)
        return reg_output


class LSTMplusModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, output_feature=2, num_layer=2):
        super(LSTMplusModel, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_feature)
        self.classify_head = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_normalized = self.bn(x)
        x_normalized = x_normalized.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x_normalized)
        reg_output = self.fc(lstm_out[:, -1, :])
        class_output = self.sigmoid(self.classify_head(torch.mean(lstm_out, dim=1)))
        return class_output, reg_output


class MTLTransModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, lstm_hidden_size=64, output_feature=32, num_layer=2, nhead=2):
        super(MTLTransModel, self).__init__()
        self.output_feature = output_feature
        self.hidden_dim = hidden_size
        self.lstm_hidden_size = lstm_hidden_size

        self.bn = nn.BatchNorm1d(input_size)
        self.input_fc = nn.Linear(input_size, hidden_size)

        self.transformer_encoder_layer = CustomizedTransformerEncoderLayer(d_model=lstm_hidden_size, nhead=nhead,
                                                                           dropout=0.3)
        self.transformer_encoder = CustomizedTransformerEncoder(self.transformer_encoder_layer,
                                                                num_layers=num_layer * 2)

        self.fc = nn.Linear(lstm_hidden_size, output_feature)

        self.classify_head = nn.Linear(output_feature, 1)
        self.regression_head = nn.Linear(output_feature, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_normalized = self.bn(x)
        x_normalized = x_normalized.permute(2, 0, 1)
        x_embedded = self.input_fc(x_normalized)

        transformer_out = self.transformer_encoder(x_embedded)
        feature = self.fc(transformer_out[-1, :, :]).view(-1, self.output_feature)

        class_output = self.sigmoid(self.classify_head(feature))
        reg_output = self.regression_head(feature)
        return class_output, reg_output


class MTLLstmModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, lstm_hidden_size=64, output_feature=32, num_layer=2, nhead=2):
        super(MTLLstmModel, self).__init__()
        self.output_feature = output_feature
        self.hidden_dim = hidden_size
        self.lstm_hidden_size = lstm_hidden_size

        self.bn = nn.BatchNorm1d(input_size)
        self.input_fc = nn.Linear(input_size, hidden_size)

        self.bi_lstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size, num_layers=num_layer * 2,
                               bidirectional=True, batch_first=False)

        self.fc = nn.Linear(lstm_hidden_size * 2, output_feature)

        self.classify_head = nn.Linear(output_feature, 1)
        self.regression_head = nn.Linear(output_feature, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_normalized = self.bn(x)
        x_normalized = x_normalized.permute(2, 0, 1)
        x_embedded = self.input_fc(x_normalized)

        lstm_out, _ = self.bi_lstm(x_embedded)
        feature = self.fc(lstm_out[-1, :, :])

        class_output = self.sigmoid(self.classify_head(feature))
        reg_output = self.regression_head(feature)
        return class_output, reg_output


class MTLRevModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, lstm_hidden_size=64, output_feature=32, num_layer=2, nhead=2):
        super(MTLRevModel, self).__init__()
        self.output_feature = output_feature
        self.hidden_dim = hidden_size
        self.lstm_hidden_size = lstm_hidden_size

        self.bn = nn.BatchNorm1d(input_size)
        self.input_fc = nn.Linear(input_size, hidden_size)

        self.transformer_encoder_layer = CustomizedTransformerEncoderLayer(d_model=lstm_hidden_size, nhead=nhead,
                                                                           dropout=0.3)
        self.transformer_encoder = CustomizedTransformerEncoder(self.transformer_encoder_layer,
                                                                num_layers=num_layer)
        self.bi_lstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size, num_layers=num_layer,
                               bidirectional=True, batch_first=False)

        self.fc = nn.Linear(lstm_hidden_size * 2, output_feature)

        self.classify_head = nn.Linear(output_feature, 1)
        self.regression_head = nn.Linear(output_feature, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_normalized = self.bn(x)
        x_normalized = x_normalized.permute(2, 0, 1)
        x_embedded = self.input_fc(x_normalized)

        transformer_out = self.transformer_encoder(x_embedded)
        print("train_out".format(transformer_out.shape))
        lstm_out, _ = self.bi_lstm(transformer_out)
        print("lstm_out".format(lstm_out.shape))
        feature = self.fc(lstm_out[-1, :, :]).view(-1, self.output_feature)
        print("feature".format(feature.shape))

        class_output = self.sigmoid(self.classify_head(feature))
        reg_output = self.regression_head(feature)
        return class_output, reg_output

