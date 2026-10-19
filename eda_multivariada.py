"""Descriptive fraud analysis; no model fitting or feature selection on labels."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display


def segment_summary(data, columns):
    """
    Summarize transaction support, fraud rates, and Wilson 95% intervals by segment.

    Parameters
    ----------
        data : pd.DataFrame
            Transactions containing the grouping columns and flag_fraude.

        columns : list[str]
            Columns that define each segment.

    Returns
    -------
        summary : pd.DataFrame
            Segment counts, fraud rates, confidence bounds, and lift relative to the base rate.

    """
    # Aggregate transaction counts and fraud labels by segment.
    summary = data.groupby(columns, observed=True, dropna=False)['flag_fraude'].agg(
        transacoes='size', fraudes='sum', taxa='mean'
    ).reset_index()
    n, p, z = summary.transacoes, summary.taxa, 1.96
    # Calculate descriptive Wilson confidence intervals.
    center = (p + z*z/(2*n))/(1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1 + z*z/n)
    summary['ic95_inferior'], summary['ic95_superior'] = center-half, center+half
    summary['lift_base'] = p / data.flag_fraude.mean()
    # Compare each segment rate with the overall fraud rate.
    return summary


def plot_cross(data, row, column, min_support=100):
    """
    Plot fraud rates and transaction counts for two categorical features.

    Parameters
    ----------
        data : pd.DataFrame
            Transactions containing the selected features and fraud labels.

        row : str
            Feature displayed on the heatmap rows.

        column : str
            Feature displayed on the heatmap columns.

        min_support : int
            Minimum transaction count required to display a fraud rate. Default is 100.

    Returns
    -------
        summary : pd.DataFrame
            Segment statistics, including groups hidden from the rate heatmap.

    """
    # Summarize feature combinations and align the rate and count tables.
    summary = segment_summary(data, [row, column])
    rates = summary.pivot(index=row, columns=column, values='taxa')
    counts = summary.pivot(index=row, columns=column, values='transacoes').reindex_like(rates)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(3.5, len(rates)*.55)))
    # Hide low-support rates and display both heatmaps.
    eligible = rates.where(counts >= min_support)
    if eligible.notna().any().any():
        sns.heatmap(eligible*100, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0], cbar_kws={'label':'Fraudes (%)'})
    else:
        axes[0].text(.1, .5, 'Nenhum segmento com suporte suficiente')
    sns.heatmap(counts, annot=True, fmt='.0f', cmap='Blues', ax=axes[1], cbar=False)
    axes[0].set_title(f'Taxa de fraude — mínimo {min_support} compras')
    axes[1].set_title('Quantidade de transações')
    fig.suptitle(f'{row} × {column}')
    fig.tight_layout()
    plt.show()
    eligible_rows = summary[summary.transacoes >= min_support]
    if not eligible_rows.empty:
        highest = eligible_rows.sort_values('taxa', ascending=False).iloc[0]
        display(Markdown(
            f'**Leitura descritiva:** maior taxa elegível em {row}={highest[row]}, '
            f'{column}={highest[column]}: **{highest.taxa:.2%}**, '
            f'{int(highest.fraudes):,} fraudes em {int(highest.transacoes):,} compras '
            f'(IC 95%: {highest.ic95_inferior:.2%}–{highest.ic95_superior:.2%}; '
            f'lift {highest.lift_base:.2f}× a taxa global). '
            'O cruzamento descreve associação; não demonstra causalidade nem ganho preditivo.'
        ))
    return summary


def plot_month_segments(data, segment, min_support=100):
    # Extract the month and summarize each monthly segment.
    """
    Plot monthly fraud rates and transaction composition for a categorical feature.

    Parameters
    ----------
        data : pd.DataFrame
            Transactions containing timestamps, segment values, and fraud labels.

        segment : str
            Categorical feature used to compare monthly patterns.

        min_support : int
            Minimum monthly segment count required to display a rate. Default is 100.

    Returns
    -------
        summary : pd.DataFrame
            Monthly segment statistics and descriptive confidence intervals.

    """
    frame = data.assign(mes=data.datahora_transacao.dt.month)
    summary = segment_summary(frame, ['mes', segment])
    rates = summary.pivot(index='mes', columns=segment, values='taxa')
    counts = summary.pivot(index='mes', columns=segment, values='transacoes').reindex_like(rates)
    # Plot supported fraud rates alongside monthly transaction shares.
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    (rates.where(counts >= min_support)*100).plot(marker='o', ax=axes[0])
    (counts.div(counts.sum(axis=1), axis=0)*100).plot(ax=axes[1])
    axes[0].set(ylabel='Fraudes (%)', title=f'Taxa mensal por {segment}', xlabel='Mês')
    axes[1].set(ylabel='Participação nas transações (%)', title='Composição mensal da base', xlabel='Mês')
    fig.tight_layout()
    plt.show()
    # Describe the observed monthly rate variation.
    spread = rates.max()-rates.min()
    display(Markdown(
        f'**Estabilidade por segmento:** a maior amplitude mensal observada foi '
        f'{spread.max()*100:.2f} pontos percentuais ({segment}={spread.idxmax()}). '
        'Compare a taxa dentro de cada grupo com sua participação no volume: '
        'uma mudança na taxa global pode acompanhar mudanças na composição. '
        'Um único ano não permite confirmar sazonalidade recorrente.'
    ))
    return summary


def behavior_analysis(data, features):
    # Report the fraction of missing values for each behavioral feature.
    """
    Compare behavioral feature ranges with device novelty and report missing history.

    Parameters
    ----------
        data : pd.DataFrame
            Engineered transactions containing behavioral features and is_new_device.

        features : list[str]
            Behavioral columns to summarize and visualize.

    Returns
    -------
        summaries : dict[str, pd.DataFrame]
            Segment statistics for each behavioral feature.

    """
    missing = data[features].isna().mean().rename('fracao_ausente').to_frame()
    display(missing)
    summaries = {}
    # Use discrete values or descriptive quartiles while retaining missing history.
    for feature in features:
        values = data[feature]
        if values.nunique() <= 6:
            bins = values.astype('string').fillna('Sem histórico suficiente')
        else:
            # Quantiles are descriptive only; these bins are not model inputs.
            bins = pd.qcut(values, q=4, duplicates='drop').astype('string').fillna('Sem histórico suficiente')
        frame = data.assign(faixa_historico=bins)
        display(Markdown(f'**{feature}:** faixas descritivas cruzadas com dispositivo novo.'))
        summaries[feature] = plot_cross(frame, 'faixa_historico', 'is_new_device')
    return summaries
