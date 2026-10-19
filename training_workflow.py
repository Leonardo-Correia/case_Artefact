"""Reproducible batch training of the project's four models."""
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

from preprocessing import (create_features, preprocessing_features,
    BEHAVIOR_NUMERICAL_FEATURES, MULTIVARIATE_NUMERICAL_FEATURES)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / 'artifacts' / 'training'
NUMERICAL = ['month', 'fraudes_ultimos_30d'] + BEHAVIOR_NUMERICAL_FEATURES + MULTIVARIATE_NUMERICAL_FEATURES
CATEGORICAL = ['tipo_estabelecimento', 'cidade', 'device_type', 'turno', 'valor_categoria']


def transform_for_model(frame, preprocessing):
    # Reuse the fitted scaler and encoder without fitting on the input batch.
    """
    Apply fitted preprocessing and imputation to engineered transactions.

    Parameters
    ----------
        frame : pd.DataFrame
            Transactions containing all engineered model inputs.

        preprocessing : dict
            Fitted scaler, encoder, imputer, and ordered model feature names.

    Returns
    -------
        transformed : np.ndarray
            Encoded, scaled, and imputed inputs in the fitted feature order.

    """
    transformed, _, _ = preprocessing_features(frame, NUMERICAL, CATEGORICAL,
        scaler=preprocessing['scaler'], encoder=preprocessing['encoder'])
    # Select the fitted feature order and apply the training imputer.
    return preprocessing['imputer'].transform(transformed[preprocessing['features']])


def run_training():
    # Create the artifact directory and prepare the historical features.
    """
    Train and persist the four models using random training, validation, and test splits.

    Parameters
    ----------
        None
            This function takes no explicit parameters.

    Returns
    -------
        metrics : pd.DataFrame
            Validation metrics and training duration for each fitted model.

    Notes
    -----
    The split is 84% training, 8% validation, and 8% test. Files are written
    to ARTIFACTS. Historical fraud features retain the existing label-availability
    assumptions; model scores are not calibrated fraud probabilities.

    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print('Criando features para 2025...', flush=True)
    raw = pd.read_csv(ROOT/'datasets/transacoes_historico_2025.csv', parse_dates=['datahora_transacao'])
    frame = create_features(raw)
    features = (ROOT/'predicting_constants/model_features.txt').read_text().splitlines()
    features = [feature.strip() for feature in features if feature.strip()]
    # Random batch holdouts: 84% training, 8% validation, 8% untouched test.
    train_valid, test = train_test_split(np.arange(len(frame)), test_size=.08, random_state=21)
    train, valid = train_test_split(train_valid, test_size=.08/.92, random_state=42, stratify=frame.iloc[train_valid].flag_fraude)
    # Fit preprocessing and imputation only on the training partition.
    train_frame, scaler, encoder = preprocessing_features(frame.iloc[train].copy(), NUMERICAL, CATEGORICAL)
    imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    X_train = imputer.fit_transform(train_frame[features])
    preprocessing = {'features':features, 'scaler':scaler, 'encoder':encoder, 'imputer':imputer}
    X_valid = transform_for_model(frame.iloc[valid].copy(), preprocessing)
    X_test = transform_for_model(frame.iloc[test].copy(), preprocessing)
    y_train = frame.iloc[train].flag_fraude.to_numpy().astype(int)
    y_valid = frame.iloc[valid].flag_fraude.to_numpy().astype(int)
    y_test = frame.iloc[test].flag_fraude.to_numpy().astype(int)
    joblib.dump(preprocessing, ARTIFACTS/'preprocessing.joblib', compress=3)
    # Persist the fitted preprocessing and reproducibility metadata.
    np.savez_compressed(ARTIFACTS/'split_indices.npz',train=train,validation=valid,test=test)
    metadata = {'python':platform.python_version(), 'sklearn':sklearn.__version__,
        'split':'random 84/8/8; test random_state=21; validation random_state=42, stratified',
        'rows':{'train':len(train),'validation':len(valid),'test':len(test)},
        'feature_count_before_imputation':len(features), 'feature_count_after_imputation':X_train.shape[1],
        'training_only_preprocessing':True,
        'note':'Historical fraud labels retain the availability assumptions of the original project. Scores are not calibrated fraud probabilities.'}
    (ARTIFACTS/'metadata.json').write_text(json.dumps(metadata,indent=2))
    models = {
        'logistic_regression':LogisticRegression(max_iter=1000,class_weight='balanced',random_state=21),
        'hist_gradient_boosting':HistGradientBoostingClassifier(learning_rate=.05,max_iter=500,max_leaf_nodes=100,l2_regularization=.0001,random_state=42),
        'random_forest':RandomForestClassifier(n_estimators=500,random_state=42,n_jobs=4),
        'mlp':MLPClassifier(hidden_layer_sizes=(128,64),activation='logistic',solver='adam',alpha=.0001,batch_size=256,learning_rate='adaptive',learning_rate_init=.001,max_iter=500,early_stopping=True,validation_fraction=.2,n_iter_no_change=20,random_state=42),
    }
    weight = compute_sample_weight(class_weight='balanced', y=y_train)
    # Configure the model candidates and class-balanced sample weights.
    predictions = pd.DataFrame({'row_index':valid,'month':frame.iloc[valid].datahora_transacao.dt.month.to_numpy(),
        'cidade':frame.iloc[valid].cidade.to_numpy(),'tipo_estabelecimento':frame.iloc[valid].tipo_estabelecimento.to_numpy(),
        'is_new_device':frame.iloc[valid].is_new_device.to_numpy(),'y_true':y_valid})
    test_predictions = pd.DataFrame({'row_index':test,'month':frame.iloc[test].datahora_transacao.dt.month.to_numpy(),
        'cidade':frame.iloc[test].cidade.to_numpy(),'tipo_estabelecimento':frame.iloc[test].tipo_estabelecimento.to_numpy(),
        'is_new_device':frame.iloc[test].is_new_device.to_numpy(),'y_true':y_test})
    metrics=[]
    for name, model in models.items():
        print(f'Treinando {name}...',flush=True)
        t=time.perf_counter()
        if name in ['hist_gradient_boosting','random_forest']:
            model.fit(X_train,y_train,sample_weight=weight)
        else:
            model.fit(X_train,y_train)
        probabilities=model.predict_proba(X_valid)[:,1]
        predictions[name]=probabilities
        test_predictions[name]=model.predict_proba(X_test)[:,1]
        # Threshold 0.5 is a descriptive reference, not the operational decision.
        selected=probabilities>=.5
        row={'model':name,'validation_roc_auc':roc_auc_score(y_valid,probabilities),
            'validation_average_precision':average_precision_score(y_valid,probabilities),
            'reference_threshold':.5,'validation_precision':precision_score(y_valid,selected,zero_division=0),
            'validation_recall':recall_score(y_valid,selected,zero_division=0),
            'validation_f1':f1_score(y_valid,selected,zero_division=0),'training_seconds':time.perf_counter()-t}
        metrics.append(row)
        joblib.dump(model,ARTIFACTS/f'{name}.joblib',compress=3)
        pd.DataFrame(metrics).to_csv(ARTIFACTS/'validation_metrics.csv',index=False)
        predictions.to_csv(ARTIFACTS/'validation_predictions.csv',index=False)
        test_predictions.to_csv(ARTIFACTS/'test_predictions.csv',index=False)
        print(json.dumps(row),flush=True)
    print(f'Treinamento concluído em {time.perf_counter()-started:.1f}s.',flush=True)
    # Fit each model, evaluate held-out scores, and save its artifacts.
    return pd.DataFrame(metrics)


if __name__=='__main__':
    run_training()
