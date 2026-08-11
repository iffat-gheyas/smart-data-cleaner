"""
Module AL: Neural Contextual Imputation (1D CNN + Bidirectional LSTM)
------------------------------------------------------------------------------
The two neural candidates from the spec's Section 8, sharing:
  - Window-tensor construction (a (window_length, n_channels) tensor
    per row: value + missingness-mask channels for the target and
    every other series)
  - A generic PyTorch training loop with early stopping, using an
    INTERNAL train/validation split carved out of the training data
    only -- never touching the actual backtest's held-out validation
    rows, matching the leakage-prevention requirement while still
    satisfying "validation-based epoch selection".

RETROSPECTIVE MODE ONLY (matches Module AJ/36's same scope decision):
the CNN here uses a two-sided window (past AND future context), and
the LSTM is bidirectional -- both appropriate for reconstructing gaps
in a completed historical dataset, not genuine forward-looking
forecasting. Forecast-safe (causal CNN, unidirectional LSTM) is a
real, separate extension, deferred until an actual forecasting
use case needs it.

BOTH MODELS ARE DELIBERATELY SMALL: a handful of channels, dropout,
weight decay, and early stopping -- appropriate given every dataset
this product has actually been tested against is well below what
would justify a larger network, per Module AK's eligibility gate
(which both these functions check before attempting to train at all).

The target's OWN position within its window is ALWAYS masked (value=0,
missing=1) regardless of whether it's genuinely observed -- using the
answer to predict the answer would be a trivial, meaningless leak.
"""

from importlib import import_module

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

fold_transform_mod = import_module("35_time_series_fold_transforms")
eligibility_mod = import_module("37_time_series_neural_eligibility")

torch.manual_seed(42)


def build_window_tensors(transformed_df: pd.DataFrame, target_col: str, other_cols: list, max_lag: int, max_lead: int) -> np.ndarray:
    """
    transformed_df: already detrended+standardized (Module AI), indexed
    by Time_Step, containing target_col and every column in other_cols.

    Returns a numpy array of shape (n_rows, window_length, n_channels),
    window_length = max_lag + max_lead + 1, n_channels = 2 per series
    (value, missingness) for the target and each other column.
    """
    window_length = max_lag + max_lead + 1
    all_cols = [target_col] + list(other_cols)
    n_channels = len(all_cols) * 2

    index_list = transformed_df.index.tolist()
    index_set = set(index_list)
    n_rows = len(index_list)
    windows = np.zeros((n_rows, window_length, n_channels), dtype=np.float32)

    col_arrays = {col: transformed_df[col] for col in all_cols}

    for row_i, t in enumerate(index_list):
        for offset in range(-max_lag, max_lead + 1):
            window_pos = offset + max_lag
            source_t = t + offset
            for col_i, col in enumerate(all_cols):
                if col == target_col and offset == 0:
                    value, missing = 0.0, 1.0  # always masked -- this IS the position being predicted
                elif source_t in index_set:
                    raw_value = col_arrays[col].loc[source_t]
                    if pd.isna(raw_value):
                        value, missing = 0.0, 1.0
                    else:
                        value, missing = float(raw_value), 0.0
                else:
                    value, missing = 0.0, 1.0
                windows[row_i, window_pos, col_i * 2] = value
                windows[row_i, window_pos, col_i * 2 + 1] = missing

    return windows


class SmallCNN1D(nn.Module):
    """A deliberately small 1D CNN: two conv layers, dropout, global average pooling, one output."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8, 1)

    def forward(self, x):  # x: (batch, window_length, n_channels)
        x = x.transpose(1, 2)  # -> (batch, n_channels, window_length), what Conv1d expects
        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


class SmallBiLSTM(nn.Module):
    """A deliberately small bidirectional LSTM: one layer, dropout, mean-pool over the sequence, one output."""

    def __init__(self, n_channels: int, hidden_size: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_channels, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):  # x: (batch, window_length, n_channels)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled).squeeze(-1)


def train_with_early_stopping(
    model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, val_fraction: float = 0.15,
    max_epochs: int = 200, patience: int = 15, batch_size: int = 16, lr: float = 1e-3, weight_decay: float = 1e-4,
) -> nn.Module:
    """
    A generic training loop with early stopping, using an INTERNAL
    split carved out of X_train/y_train only -- this never touches the
    backtest's actual held-out validation rows, satisfying
    "validation-based epoch selection" without any leakage.
    """
    n = len(X_train)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_train_t = torch.tensor(X_train[train_idx], dtype=torch.float32)
    y_train_t = torch.tensor(y_train[train_idx], dtype=torch.float32)
    X_val_t = torch.tensor(X_train[val_idx], dtype=torch.float32)
    y_val_t = torch.tensor(y_train[val_idx], dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        perm_epoch = torch.randperm(len(X_train_t))
        for i in range(0, len(X_train_t), batch_size):
            batch_idx = perm_epoch[i : i + batch_size]
            optimizer.zero_grad()
            preds = model(X_train_t[batch_idx])
            loss = loss_fn(preds, y_train_t[batch_idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = loss_fn(val_preds, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _fit_and_predict_neural(
    target_series: pd.Series, other_series_dict: dict, architecture: str,
    max_lag: int = 3, max_lead: int = 3, standardization_method: str = "zscore",
) -> tuple:
    """Shared implementation for both CNN and LSTM -- identical pipeline, only the model class differs."""
    other_cols = list(other_series_dict.keys())
    n_variables = 1 + len(other_cols)
    window_length = max_lag + max_lead + 1

    n_observed = int(target_series.notna().sum())
    missing_percentage = 100.0 * target_series.isna().sum() / len(target_series) if len(target_series) else 100.0
    eligibility = eligibility_mod.check_neural_eligibility(n_observed, n_variables, window_length, missing_percentage)

    if not eligibility["eligible"]:
        was_imputed = pd.Series(False, index=target_series.index)
        could_not_fill = target_series.isna()
        return target_series.copy(), was_imputed, could_not_fill

    target_col = "__target__"
    combined = pd.DataFrame({target_col: target_series, **other_series_dict})

    fold_transforms = {}
    transformed = pd.DataFrame(index=combined.index)
    for col in combined.columns:
        ft = fold_transform_mod.fit_fold_transform(combined[col], standardization_method)
        fold_transforms[col] = ft
        transformed[col] = fold_transform_mod.transform(combined[col], ft)

    windows = build_window_tensors(transformed, target_col, other_cols, max_lag, max_lead)
    n_channels = windows.shape[2]

    training_mask = target_series.notna().to_numpy()
    y_transformed = transformed[target_col].to_numpy()

    X_train = windows[training_mask]
    y_train = y_transformed[training_mask]

    if architecture == "cnn":
        model = SmallCNN1D(n_channels)
    elif architecture == "lstm":
        model = SmallBiLSTM(n_channels)
    else:
        raise ValueError(f"Unknown architecture '{architecture}'. Choose 'cnn' or 'lstm'.")

    model = train_with_early_stopping(model, X_train, y_train)

    predict_mask = target_series.isna().to_numpy()
    imputed = target_series.copy()
    was_imputed = pd.Series(False, index=target_series.index)
    could_not_fill = pd.Series(False, index=target_series.index)

    if predict_mask.sum() > 0:
        X_predict = torch.tensor(windows[predict_mask], dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            y_pred_transformed = model(X_predict).numpy()

        predict_index = target_series.index[predict_mask]
        y_pred_transformed_series = pd.Series(y_pred_transformed, index=predict_index)
        y_pred_original = fold_transform_mod.inverse_transform(y_pred_transformed_series, fold_transforms[target_col])

        imputed.loc[predict_index] = y_pred_original
        was_imputed.loc[predict_index] = True

    return imputed, was_imputed, could_not_fill


def fit_and_predict_cnn(target_series: pd.Series, other_series_dict: dict, max_lag: int = 3, max_lead: int = 3) -> tuple:
    return _fit_and_predict_neural(target_series, other_series_dict, "cnn", max_lag, max_lead)


def fit_and_predict_bilstm(target_series: pd.Series, other_series_dict: dict, max_lag: int = 3, max_lead: int = 3) -> tuple:
    return _fit_and_predict_neural(target_series, other_series_dict, "lstm", max_lag, max_lead)


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: on a genuinely tiny dataset (well below the 200-row threshold), both must correctly decline")
    print("=" * 60)
    tiny_df = pd.read_csv("Rainfall_multivariate_test.csv").head(36)  # deliberately small, like the shampoo dataset
    tiny_df["Time_Step"] = range(len(tiny_df))
    tiny_df = tiny_df.set_index("Time_Step")
    tiny_other = {"Specific Humidity": tiny_df["Specific Humidity"], "Temperature": tiny_df["Temperature"]}

    imputed_cnn, was_imputed_cnn, could_not_fill_cnn = fit_and_predict_cnn(tiny_df["Precipitation"], tiny_other)
    print("CNN on a 36-row dataset -- declined (was_imputed all False):", not was_imputed_cnn.any())
    print("(Note: the earlier 252-row rainfall file actually PASSES eligibility under these thresholds --")
    print(" already confirmed directly in Module AK/37's own tests -- so a genuinely tiny 36-row file is used here instead.)")

    print("\n" + "=" * 60)
    print("TEST 2: on a LARGE synthetic correlated dataset, both must actually train and produce accurate estimates")
    print("=" * 60)
    rng = np.random.RandomState(7)
    n = 1000
    humidity = 10 + 5 * np.sin(np.arange(n) * 2 * np.pi / 12) + rng.normal(0, 1, n)
    precipitation = 50 + 3.5 * humidity + rng.normal(0, 5, n)
    big_df = pd.DataFrame({"Humidity": humidity, "Precipitation": precipitation}, index=pd.Index(range(n), name="Time_Step"))
    true_precip = big_df["Precipitation"].copy()
    missing_positions = rng.choice(n, size=30, replace=False)
    big_df.loc[missing_positions, "Precipitation"] = np.nan

    other_big = {"Humidity": big_df["Humidity"]}
    for name, fit_fn in [("CNN", fit_and_predict_cnn), ("BiLSTM", fit_and_predict_bilstm)]:
        imputed_big, was_imputed_big, could_not_fill_big = fit_fn(big_df["Precipitation"], other_big)
        errors = (imputed_big.loc[missing_positions] - true_precip.loc[missing_positions]).abs()
        print(f"  {name}: trained and predicted, was_imputed count={was_imputed_big.sum()}, MAE={errors.mean():.2f}")

    print("\n" + "=" * 60)
    print("TEST 3: window tensor construction -- target's own center position always masked")
    print("=" * 60)
    small_transformed = pd.DataFrame({"__target__": [1.0, 2.0, 3.0, 4.0, 5.0], "Y": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=pd.Index(range(5), name="Time_Step"))
    windows_test = build_window_tensors(small_transformed, "__target__", ["Y"], max_lag=1, max_lead=1)
    print("Shape (n_rows, window_length, n_channels):", windows_test.shape, "(should be (5, 3, 4))")
    print("Row 2, center position (index 1), target value+missing channels (should be [0.0, 1.0] -- always masked):", windows_test[2, 1, 0:2])
    print("Row 2, center position, Y value+missing channels (should be [30.0, 0.0] -- Y IS observable at its own current time):", windows_test[2, 1, 2:4])
