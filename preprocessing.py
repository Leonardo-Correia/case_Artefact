from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
) 

def transform_dtypes(dataframe: pd.DataFrame, 
                     feature_list: list,
                     making_predictions = True 
    ) -> pd.DataFrame:
    
    """
    Transform input column types for the training or prediction workflow.

    Parameters
    ----------
        dataframe : pd.DataFrame
            Input transactions whose columns will be selected and converted.

        feature_list : list
            Expected input features, including flag_fraude for training.

        making_predictions : bool
            Whether to exclude the target for prediction. Default is True.

    Returns
    -------
        dataframe : pd.DataFrame or None
            Selected transactions with datetime timestamps, or None when the timestamp feature is absent.

    Notes
    -----
    In prediction mode, the supplied feature list is modified in place.

    """
    # Filtering features according to feature_list
    if making_predictions:
        feature_list_predictions = feature_list
        feature_list_predictions.remove('flag_fraude')

        dataframe = dataframe[feature_list_predictions]
    else:
        dataframe = dataframe[feature_list]

    # Temporal Features
    if 'datahora_transacao' in feature_list:
        dataframe['datahora_transacao'] = pd.to_datetime(dataframe['datahora_transacao'], format='%Y-%m-%d %H:%M:%S')

        return dataframe
    else:
        print(f"Used dataframe has no feature called datahora_transacao")


BEHAVIOR_NUMERICAL_FEATURES = [
    "compras_ultimas_24h", "compras_ultimos_7d", "compras_ultimos_30d",
    "horas_desde_ultima_compra", "valor_medio_ultimos_30d",
    "valor_desvio_padrao_ultimos_30d", "valor_diferenca_media_30d",
    "valor_razao_media_30d",
]


def create_behavior_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer purchase history features using strictly earlier transactions.

    Parameters
    ----------
        dataframe : pd.DataFrame
            Transactions containing customer, timestamp, and amount columns.

    Returns
    -------
        result : pd.DataFrame
            Transactions with historical counts and spending statistics, preserving row order and index.

    Notes
    -----
    Windows include the left boundary and exclude the current timestamp.
    Missing means and intervals remain NaN; sample standard deviation requires
    two earlier purchases. Ratios with a zero historical mean remain NaN.

    """
    # Copy transactions and sort customer histories while preserving row positions.
    result = dataframe.copy()
    history = dataframe[["nome_cliente", "datahora_transacao", "valor_transacao"]].copy()
    history.index = pd.RangeIndex(len(history))
    history = history.sort_values(["nome_cliente", "datahora_transacao"], kind="stable")
    groups = history.groupby("nome_cliente", sort=False, dropna=False)

    # Shift distinct timestamps so simultaneous purchases never see each other.
    distinct = history[["nome_cliente", "datahora_transacao"]].drop_duplicates().copy()
    distinct["previous_time"] = distinct.groupby(
        "nome_cliente", sort=False, dropna=False
    )["datahora_transacao"].shift()
    previous = history.reset_index().merge(
        distinct, on=["nome_cliente", "datahora_transacao"], how="left", validate="many_to_one"
    ).set_index("index").reindex(range(len(result)))
    result["horas_desde_ultima_compra"] = (
        (previous["datahora_transacao"] - previous["previous_time"])
        .dt.total_seconds().to_numpy() / 3600
    )
    # Calculate the interval since the previous distinct transaction timestamp.
    result["sem_historico_anterior"] = previous["previous_time"].isna().to_numpy().astype(int)

    for window, name in [
        ("24h", "compras_ultimas_24h"),
        ("7D", "compras_ultimos_7d"),
        ("30D", "compras_ultimos_30d"),
    ]:
        rolling = groups.rolling(window, on="datahora_transacao", closed="left", min_periods=1)
        counts = rolling["valor_transacao"].count().reset_index(level=0, drop=True)
        result[name] = pd.Series(counts.to_numpy(), index=history.index).reindex(range(len(result))).fillna(0).to_numpy()
        if window == "30D":
            for statistic, column in [
                ("mean", "valor_medio_ultimos_30d"),
                ("std", "valor_desvio_padrao_ultimos_30d"),
            ]:
                values = getattr(rolling["valor_transacao"], statistic)()
                result[column] = pd.Series(values.to_numpy(), index=history.index).reindex(range(len(result))).to_numpy()

    # Calculate purchase counts and spending statistics over historical windows.
    result["valor_diferenca_media_30d"] = result["valor_transacao"] - result["valor_medio_ultimos_30d"]
    denominator = result["valor_medio_ultimos_30d"].replace(0, np.nan)
    result["valor_razao_media_30d"] = result["valor_transacao"] / denominator
    return result


MULTIVARIATE_NUMERICAL_FEATURES = ["novo_device_razao_valor_30d"]
MULTIVARIATE_FEATURES = [
    "novo_device_noite", "novo_device_mudanca_cidade",
    *MULTIVARIATE_NUMERICAL_FEATURES,
]


def create_multivariate_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create interactions between device novelty, transaction time, city changes, and spending.

    Parameters
    ----------
        dataframe : pd.DataFrame
            Transactions containing device, time-period, city-change, and historical amount features.

    Returns
    -------
        result : pd.DataFrame
            Transactions with binary interactions and a numerical spending interaction.

    Notes
    -----
    No labels or fitted statistics are used. Missing historical ratios remain
    missing, including for known devices.

    """
    # Copy transactions and identify new devices.
    result = dataframe.copy()
    new_device = result["is_new_device"].astype(bool)
    # Combine device novelty with nighttime transactions and city changes.
    result["novo_device_noite"] = (
        new_device & result["turno"].eq("noite")
    ).astype(int)
    result["novo_device_mudanca_cidade"] = (
        new_device & result["changed_cidade"].eq(1)
    ).astype(int)
    # Preserve missing historical ratios in the numerical interaction.
    result["novo_device_razao_valor_30d"] = (
        new_device.astype(int) * result["valor_razao_media_30d"]
    )
    return result


def create_features(dataframe: pd.DataFrame
    ) -> pd.DataFrame:

    """
    Create temporal, historical fraud, behavioral change, and interaction features.

    Parameters
    ----------
        dataframe : pd.DataFrame
            Transactions with customer, timestamp, amount, categories, and fraud labels.

    Returns
    -------
        dataframe : pd.DataFrame
            Transactions with engineered features in their original row order.

    """

    dataframe = create_behavior_features(dataframe)

    # Temporal feature "month", "day_of_week", "is_weekend"
    dataframe['month'] = dataframe['datahora_transacao'].dt.month

    dataframe['day_of_week'] = dataframe['datahora_transacao'].dt.dayofweek

    dataframe['is_weekend'] = dataframe['datahora_transacao'].dt.dayofweek >= 5
    ################################################

    # Same past-label window as before, grouped once instead of scanning
    # the full dataframe separately for every client.
    fraud_history = dataframe[["nome_cliente", "datahora_transacao", "flag_fraude"]].copy()
    fraud_history.index = pd.RangeIndex(len(fraud_history))
    fraud_history = fraud_history.sort_values(
        ["nome_cliente", "datahora_transacao"], kind="stable"
    )
    fraud_values = (
        fraud_history.groupby("nome_cliente", sort=False, dropna=False)
        .rolling("30D", on="datahora_transacao", closed="left")["flag_fraude"]
        .sum()
    )
    dataframe["fraudes_ultimos_30d"] = (
        pd.Series(fraud_values.to_numpy(), index=fraud_history.index)
        .reindex(range(len(dataframe))).fillna(0).to_numpy()
    )

    # Creating feature for "changing current state"
    dataframe["_original_order"] = range(len(dataframe))

    dataframe = dataframe.sort_values(
        ["nome_cliente", "datahora_transacao"]
    )

    well_features = [
        "cidade",
        "tipo_estabelecimento",
        "device_type",
        "turno",
        "valor_categoria"
    ]

    grouped_clients = dataframe.groupby(
        "nome_cliente",
        sort=False,
        dropna=False
    )

    for column in well_features:

        feature_change = f"changed_{column}"

        dataframe[feature_change] = (
            dataframe[column].ne(grouped_clients[column].shift())
            .astype(int)
        )

        first_transaction_indices = (
            grouped_clients
            .head(1)
            .index
        )

        dataframe.loc[
            first_transaction_indices,
            feature_change
        ] = 0

    # Restore the original row order
    dataframe = (
        dataframe
        .sort_values("_original_order")
        .drop(columns="_original_order")
        .reset_index(drop=True)
    )

    return create_multivariate_features(dataframe)


def preprocessing_features(
    dataframe: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
    scaler: StandardScaler | None = None,
    encoder: OneHotEncoder | None = None
) -> tuple[pd.DataFrame, StandardScaler, OneHotEncoder]:
    """
    Standardize numerical features and one-hot encode categorical features.

    Parameters
    ----------
        dataframe : pd.DataFrame
            Transactions to transform.

        numerical_features : list[str]
            Numerical columns to standardize.

        categorical_features : list[str]
            Categorical columns to one-hot encode.

        scaler : StandardScaler, optional
            Fitted scaler to reuse; when None, fit a new scaler.

        encoder : OneHotEncoder, optional
            Fitted encoder to reuse; when None, fit a new encoder.

    Returns
    -------
        transformed_dataframe : pd.DataFrame
            Remaining columns joined with scaled and encoded features.

        scaler : StandardScaler
            Fitted numerical scaler.

        encoder : OneHotEncoder
            Fitted categorical encoder.

    """

    # Copy the input before validating and transforming its columns.
    dataframe = dataframe.copy()

    # Verify if the features are in the dataframe
    required_features = numerical_features + categorical_features

    missing_features = [
        feature
        for feature in required_features
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            f"The following columns are not in dataframe: {missing_features}"
        )

    ################################################
    # Standard scaling
    if scaler is None:
        scaler = StandardScaler()

        numerical_values = scaler.fit_transform(
            dataframe[numerical_features]
        )
    else:
        numerical_values = scaler.transform(
            dataframe[numerical_features]
        )

    numerical_dataframe = pd.DataFrame(
        numerical_values,
        columns=numerical_features,
        index=dataframe.index
    )

    ################################################    
    # One-hot encoding
    if encoder is None:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        categorical_values = encoder.fit_transform(
            dataframe[categorical_features]
        )
    else:
        categorical_values = encoder.transform(
            dataframe[categorical_features]
        )

    categorical_columns = encoder.get_feature_names_out(
        categorical_features
    )

    categorical_dataframe = pd.DataFrame(
        categorical_values,
        columns=categorical_columns,
        index=dataframe.index
    )

    ################################################
    # Join transformed features
    remaining_features = dataframe.drop(
        columns=numerical_features + categorical_features
    )

    transformed_dataframe = pd.concat(
        [
            remaining_features,
            numerical_dataframe,
            categorical_dataframe
        ],
        axis=1
    )

    return transformed_dataframe, scaler, encoder

def removing_colunms_for_model(
        dataframe: pd.DataFrame,
        model_feature_list: list,
        making_predictions: bool = True
)-> pd.DataFrame:
    """
    Select the columns used for model training or prediction.

    Parameters
    ----------
        dataframe : pd.DataFrame
            Transactions whose model columns will be selected.

        model_feature_list : list[str]
            Ordered feature names used by the model.

        making_predictions : bool
            Whether to omit the target column. Default is True.

    Returns
    -------
        dataframe : pd.DataFrame
            Selected model inputs, including flag_fraude when preparing training data.

    Notes
    -----
    In training mode, flag_fraude is appended to the supplied feature list.

    """
    # Select model inputs and include the target only for training.
    if making_predictions:
        return dataframe[model_feature_list]
    else:
        model_feature_list.append('flag_fraude')
        return dataframe[model_feature_list]
