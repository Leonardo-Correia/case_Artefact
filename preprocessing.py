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
    Function created to transform dtypes of every feature to assure correct workflow.

    Parameters
    ----------
        dataframe : pd.DataFrame
            Dataframe to have features created.
        
        feature_list : list
            List with initial features list to workflow.

        making_predictions : bool 
            Boolean variable to decide weather the function will be used for predictions or not.
    Returns
        making_predictions : pd.DataFrame
            Dataframe with correct dtypes features
    -------
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


def create_features(dataframe: pd.DataFrame
    ) -> pd.DataFrame:

    """
    Function for Creating features that will be used in the model training and deploying 

    Parameters
    ----------
        dataframe : pd.DataFrame
            Dataframe to have features created.

    Returns
        dataframe : pd.Dataframe
            Dataframe with new features
    -------
    """

    # Temporal feature "month", "day_of_week", "is_weekend"
    dataframe['month'] = dataframe['datahora_transacao'].dt.month

    dataframe['day_of_week'] = dataframe['datahora_transacao'].dt.dayofweek

    dataframe['is_weekend'] = dataframe['datahora_transacao'].dt.dayofweek >= 5
    ################################################

    # Adding "number of fraudlent transactions of the last month" feature
    dataframe[f'fraudes_ultimos_{30}d'] = 0.0

    for client in dataframe['nome_cliente'].unique():
        
        df_client_temp = (
            dataframe[dataframe['nome_cliente'] == client]
            .sort_values('datahora_transacao')
            .copy()
        )

        fraude_rolling = (
            df_client_temp
            .set_index('datahora_transacao')['flag_fraude']
            .rolling(
                window=f'{30}D',
                closed='left'
            )
            .sum()
            .fillna(0)
            .to_numpy()
        )

        dataframe.loc[
            df_client_temp.index,
            f'fraudes_ultimos_{30}d'
        ] = fraude_rolling
    ################################################

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
            grouped_clients[column]
            .transform(lambda values: values.ne(values.shift()))
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

    # Retorna para a ordem original
    dataframe = (
        dataframe
        .sort_values("_original_order")
        .drop(columns="_original_order")
        .reset_index(drop=True)
    )

    return dataframe


def preprocessing_features(
    dataframe: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
    scaler: StandardScaler | None = None,
    encoder: OneHotEncoder | None = None
) -> tuple[pd.DataFrame, StandardScaler, OneHotEncoder]:
    """
    Applies StandardScaler to numerical features and OneHotEncoder
    to categorical features.

    If scaler and encoder are not provided, they are created and fitted.
    Otherwise, the previously fitted objects are only used to transform
    the input data.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe to be transformed.

    numerical_features : list[str]
        Numerical columns that will be standardized.

    categorical_features : list[str]
        Categorical columns that will be one-hot encoded.

    scaler : StandardScaler, optional
        Previously fitted scaler.

    encoder : OneHotEncoder, optional
        Previously fitted encoder.

    Returns
    -------
    transformed_dataframe : pd.DataFrame
        Dataset after encoding and scaling.

    scaler : StandardScaler
        Fitted scaler.

    encoder : OneHotEncoder
        Fitted encoder.
    """

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
    Function created for removing eny unecessary feature.

    Parameters
    ----------
        dataframe : pd.DataFrame
            Dataframe to have features removed.

        model_feature_list : list[str]
            List of features that the model uses for train and prediction

        making_predictions : bool
            Default = True; Boolean variable for defining if the function will be used for training or evaluation.
    Returns
        dataframe : pd.Dataframe
            Dataframe using removed selected features
    -------
    """
    if making_predictions:
        return dataframe[model_feature_list]
    else:
        model_feature_list.append('flag_fraude')
        return dataframe[model_feature_list]
