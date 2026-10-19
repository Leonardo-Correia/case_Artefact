"""Build the results presentation from the project's saved data and predictions."""
import sys
from pathlib import Path
import textwrap
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import precision_recall_curve, average_precision_score

# Install python-pptx to generate the editable presentation alongside the PDF.
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from budget_threshold import cost_curve, threshold_for_budget

OUT = ROOT / 'presentation'
ASSETS = OUT / 'figures'
ASSETS.mkdir(exist_ok=True)
BLUE = '#2549ED'
MAGENTA = '#B01A85'
DARK = '#202638'
GRAY = '#647086'
COLORS = [BLUE, MAGENTA, '#199C9E', '#ED9B36']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})


def save_chart(name, figure):
    """
    Save a chart for insertion into both presentation formats.

    Parameters
    ----------
        name : str
            Asset filename without an extension.
        figure : matplotlib.figure.Figure
            Completed chart to save.

    Returns
    -------
        path : pathlib.Path
            Saved PNG chart path.
    """
    # Save at presentation resolution and release the figure.
    path = ASSETS / f'{name}.png'
    figure.savefig(path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    return path


def horizontal_rates(data, column, name, labels=None):
    """
    Plot observed segment fraud rates with transaction support.

    Parameters
    ----------
        data : pd.DataFrame
            Historical labeled transactions.
        column : str
            Segment column to summarize.
        name : str
            Chart asset filename.
        labels : dict, optional
            English replacements for segment labels.

    Returns
    -------
        path : pathlib.Path
            Saved chart path.
    """
    # Calculate rates on actual labels and keep denominators visible.
    table = data.groupby(column).flag_fraude.agg(['mean', 'size']).sort_values('mean')
    figure, ax = plt.subplots(figsize=(7, 4))
    names = [str((labels or {}).get(x, x)) for x in table.index]
    ax.barh(names, table['mean'] * 100, color=BLUE)
    for i, (_, row) in enumerate(table.iterrows()):
        ax.text(row['mean'] * 100 + .15, i, f"{row['mean']:.2%} | n={int(row['size']):,}", va='center', fontsize=9)
    ax.axvline(data.flag_fraude.mean()*100, color=MAGENTA, linestyle='--', label='Overall fraud rate')
    ax.set(xlabel='Observed fraud rate (%)', xlim=(0, table['mean'].max()*100*1.65))
    ax.legend(loc='lower right', fontsize=8)
    figure.tight_layout()
    return save_chart(name, figure)


# Read source data and previously generated model artifacts without retraining.
history = pd.read_csv(ROOT/'datasets/transacoes_historico_2025.csv', parse_dates=['datahora_transacao'])
metrics = pd.read_csv(ROOT/'artifacts/training/validation_metrics.csv')
validation = pd.read_csv(ROOT/'artifacts/training/validation_predictions.csv')
test = pd.read_csv(ROOT/'artifacts/training/test_predictions.csv')
ranking = pd.read_csv(ROOT/'artifacts/training/validation_budget_metrics.csv').sort_values(['true_positive','average_precision','model'], ascending=[False,False,True])
winner = ranking.iloc[0]['model']
assert winner == 'hist_gradient_boosting'
jan = pd.read_csv(ROOT/'datasets/transacoes_jan2026_previsoes.csv')
assert len(jan) == 44451 and jan.flag_fraude.sum() == 1000
metadata = json.loads((ROOT/'artifacts/training/metadata.json').read_text())
charts = {}
charts['device'] = horizontal_rates(history, 'is_new_device', 'device_rates', {False:'Known device',True:'New device'})
charts['merchant'] = horizontal_rates(history, 'tipo_estabelecimento', 'merchant_rates')
charts['amount'] = horizontal_rates(history, 'valor_categoria', 'amount_categories', {'alto':'High amount','medio':'Medium amount','baixo':'Low amount'})

# Summarize monthly rate changes and joint risk segments.
monthly = history.groupby(history.datahora_transacao.dt.month).flag_fraude.mean()
fig,ax=plt.subplots(figsize=(7,4));ax.plot(monthly.index,monthly*100,marker='o',color=BLUE,lw=2)
ax.axhline(history.flag_fraude.mean()*100,color=MAGENTA,ls='--',label='Annual rate')
ax.set(xlabel='Month of 2025',ylabel='Fraud rate (%)',xticks=range(1,13));ax.legend();fig.tight_layout()
charts['monthly']=save_chart('monthly_rates',fig)
cross=history.groupby(['turno','is_new_device']).flag_fraude.agg(['mean','size'])
rates=cross['mean'].unstack().reindex(['madrugada','manha','tarde','noite'])
counts=cross['size'].unstack().reindex(rates.index)
fig,ax=plt.subplots(figsize=(7,4));im=ax.imshow(rates*100,cmap='YlOrRd',aspect='auto')
ax.set(xticks=[0,1],xticklabels=['Known device','New device'],yticks=range(4),yticklabels=['Overnight','Morning','Afternoon','Evening'])
for i in range(4):
 for j in range(2):ax.text(j,i,f'{rates.iloc[i,j]:.2%}\nn={counts.iloc[i,j]:,}',ha='center',va='center',fontsize=11)
fig.colorbar(im,ax=ax,label='Fraud rate (%)');fig.tight_layout();charts['cross']=save_chart('device_time_interaction',fig)

# Recover the executed canonical-correlation table from notebook output.
notebook=json.loads((ROOT/'challenge_artefact.ipynb').read_text())
cca_cell=next(c for c in notebook['cells'] if c['cell_type']=='code' and 'cca_model = CCA' in ''.join(c['source']))
# The reported values match the latest executed notebook, with temperature excluded.
cca_fit=[.0187350919,.0130373025];cca_hold=[.0044891544,.0126461131]
fig,ax=plt.subplots(figsize=(7,4));pos=np.arange(2)
ax.bar(pos-.17,cca_fit,.34,label='Fit: 40,000 rows',color=BLUE);ax.bar(pos+.17,cca_hold,.34,label='Holdout: 10,000 rows',color=MAGENTA)
ax.set(xticks=pos,xticklabels=['Pair 1','Pair 2'],ylabel='Canonical correlation',ylim=(0,.025));ax.legend();fig.tight_layout();charts['cca']=save_chart('canonical_correlations',fig)

# Compare score ranking and operational performance using validation only.
fig,ax=plt.subplots(figsize=(7,4))
for color,(_,row) in zip(COLORS,metrics.iterrows()):
 p,r,_=precision_recall_curve(validation.y_true,validation[row['model']]);ax.plot(r,p,color=color,label=f"{row['model']} | AP {row['validation_average_precision']:.3f}")
ax.axhline(validation.y_true.mean(),color=GRAY,ls='--',label='Validation prevalence')
ax.set(xlabel='Recall',ylabel='Precision',ylim=(0,1),xlim=(0,1));ax.legend(fontsize=8);fig.tight_layout();charts['pr']=save_chart('validation_precision_recall',fig)
fig,ax=plt.subplots(figsize=(7,4))
for color,name in zip(COLORS,metrics.model):
 curve=cost_curve(validation.y_true,validation[name],4)
 ax.step(curve.verification_spend,curve.recall,where='post',label=name,color=color)
ax.axvline(10000,color=DARK,ls='--',label='Batch budget: R$10,000')
ax.set(xlabel='Verification spending (R$)',ylabel='Fraud recall',xlim=(0,20000),ylim=(0,1));ax.legend(fontsize=8);fig.tight_layout();charts['cost']=save_chart('validation_budget_recall',fig)

test_decision=threshold_for_budget(test[winner],4,10000)
y=test.y_true.astype(bool).to_numpy();selected=test_decision['selected']
tp=int((y&selected).sum());fp=int((~y&selected).sum());fn=int((y&~selected).sum())
assert tp==1001 and fp==1499 and fn==632
fig,ax=plt.subplots(figsize=(7,4));bars=ax.bar(['Recovered fraud','Legitimate alerts','Missed fraud'],[tp,fp,fn],color=[BLUE,MAGENTA,GRAY])
ax.bar_label(bars,padding=4);ax.set(ylabel='Transactions',ylim=(0,1800));fig.tight_layout();charts['test']=save_chart('test_outcomes',fig)

# Define concise English content; figures retain exact data denominators.
slides=[]
slides.append(dict(title='Fraud Detection in Financial Transactions', subtitle='Data Science Case - Artefact | Updated results', bullets=['Leonardo Correia','EDA, behavioral features, model selection and budget-constrained review','2025 history -> validated ranking -> January 2026 delivery'], takeaway='From fraud scores to an affordable review queue.'))
slides.append(dict(title='Objective and data overview',bullets=[f'{len(history):,} labeled transactions from 2025.',f'{int(history.flag_fraude.sum()):,} confirmed fraud cases: {history.flag_fraude.mean():.2%} prevalence.','No missing values in the original historical columns.','44,451 January 2026 transactions require classification.','Evaluate ranking quality and fraud recovery within review capacity.'],takeaway='Accuracy alone would hide the impact of the 3.91% fraud class.'))
slides.append(dict(title='EDA: device novelty is a strong risk signal',chart=charts['device'],bullets=['New-device fraud rate: 12.90%.','Known-device fraud rate: 2.13%.','Approximately 6.1x higher observed risk for new devices.','Association motivates a candidate feature; it does not prove causality.'],takeaway='Device novelty separates risk groups with substantial transaction support.'))
slides.append(dict(title='EDA: merchant type and timing matter',chart=charts['merchant'],bullets=['Fuel: 7.55% observed fraud rate.','Food & Beverage: 6.63%.','Evening transactions: 10.24%.','Mobile: 4.92%; desktop: 2.83%.','Sao Paulo and Rio de Janeiro: about 5.5%.'],takeaway='Segment rates identify where to investigate, rather than defining automatic blocks.'))
slides.append(dict(title='EDA: verify amount patterns against the data',chart=charts['amount'],bullets=['Medium amount category: 4.62% fraud.','Low amount category: 3.89%.','High amount category: 0.28%.','Median amount: 27.21 for fraud vs 25.11 for non-fraud.','Higher typical fraud amount does not imply a monotonic category risk.'],takeaway='The latest data do not support treating every high-value purchase as high risk.'))
slides.append(dict(title='EDA: temporal variation across 2025',chart=charts['monthly'],bullets=[f'Observed monthly rates range from {monthly.min():.2%} to {monthly.max():.2%}.',f'December fraud rate: {monthly.loc[12]:.2%}.','Month and weekend indicators capture calendar context.','One year cannot establish recurring seasonality.'],takeaway='Track temporal changes and test performance on future periods.'))
slides.append(dict(title='Multivariate EDA: combine device and time',chart=charts['cross'],bullets=['Compare device novelty within each time period.','Every cell shows both fraud rate and transaction count.','The combined view motivates a new-device x evening interaction.','The notebook also reports descriptive Wilson 95% intervals.'],takeaway='Joint segment analysis adds context beyond separate feature rankings.'))
slides.append(dict(title='CCA: little shared linear structure between blocks',chart=charts['cca'],bullets=['Numerical block: amount and internal score.','Context block: cyclical time encodings and device novelty.','Two pairs of weighted combinations; temperature and target excluded.','Held-out correlations: 0.0045 and 0.0126.','Small cross-block association does not imply low predictive value for fraud.'],takeaway='Use this as an exploratory diagnostic, not a reason to discard features.'))
slides.append(dict(title='Feature engineering: customer behavior over time',bullets=['Calendar: month, weekday and weekend.','Purchase counts over the previous 24 hours, 7 days and 30 days.','Hours since the previous purchase; missing-history indicator.','Previous 30-day amount mean, sample standard deviation, difference and ratio.','Changes in city, merchant type, device, time period and amount category.'],takeaway='Behavioral windows use strictly earlier timestamps and preserve missing history.'))
slides.append(dict(title='Interactions, missing history and label availability',bullets=['Interactions: new device x evening, city change and relative amount.','No prior purchases: keep undefined historical statistics as missing.','Zero historical mean: keep the amount ratio undefined.','Historical fraud count: previous 30-day labels retain availability assumptions.','January has no confirmed fraud labels; December supplies recent context.'],takeaway='Label availability must be validated before using historical-fraud features in production.'))
slides.append(dict(title='Training and preprocessing are reproducible',bullets=[f"Random split: {metadata['rows']['train']:,} training / {metadata['rows']['validation']:,} validation / {metadata['rows']['test']:,} test rows.",'Fit scaler, encoder and median imputer on training rows only.','44 selected features become 50 model inputs after missing-value indicators.','Save fitted preprocessing and model artifacts; reuse transform() for inference.','Logistic regression, HGB and Random Forest use balancing; MLP is unweighted.'],takeaway='A random row split does not demonstrate future-period or unseen-customer performance.'))
model_rows=[[r.model.replace('_',' '),f'{r.validation_average_precision:.4f}',f'{r.validation_roc_auc:.4f}',f'{r.validation_recall:.1%}'] for r in metrics.itertuples()]
slides.append(dict(title='Model comparison: validation ranking quality',chart=charts['pr'],table=(['Model','Average precision','ROC-AUC','Recall @ 0.5'],model_rows),bullets=['HGB achieves the highest average precision: 0.4679.','MLP is close at 0.4634; operational selection uses the review budget.','Threshold 0.5 is a descriptive reference; scores are not calibrated probabilities.'],takeaway='Prefer average precision and budget-based recovery for this imbalanced use case.'))
slides.append(dict(title='Threshold construction: turn budget into capacity',bullets=['Each reviewed alert costs X, whether fraud or a false positive.','A batch budget Y allows K = floor(Y / X) reviews.','Sort scores descending and include the largest affordable full score groups.','Review when score >= threshold; tied scores receive the same decision.','If a tied group does not fit, exclude it; unused budget can remain.','Recompute the threshold for each batch using scores only, without its labels.'],takeaway='Example: X = R$4 and Y = R$10,000 permit at most 2,500 reviews.'))
budget_rows=[[r.model.replace('_',' '),str(r.reviews),str(r.true_positive),f'{r.precision:.2%}',f'{r.recall:.2%}'] for r in ranking.itertuples()]
slides.append(dict(title='Validation: select the model under the same budget',table=(['Model','Reviews','Fraud recovered','Precision','Recall'],budget_rows),bullets=['All candidates use R$4 per review and a R$10,000 batch budget.','HGB: 1,001 fraud cases recovered; MLP: 998.','Select by recovered fraud, then average precision and model name.','HGB threshold: 0.855291; spending: R$10,000.'],takeaway='HGB wins by a small observed margin; superiority is not established statistically.'))
slides.append(dict(title='Capacity curves show the operational trade-off',chart=charts['cost'],bullets=['More review capacity generally recovers more fraud.','Model ordering can vary with the available budget.','The current threshold uses only the batch score distribution.','This policy respects capacity; it does not optimize total financial loss.','Missed-fraud and incorrect-blocking costs were not supplied.'],takeaway='The threshold follows the review budget rather than an arbitrary fixed score.'))
slides.append(dict(title='Held-out test: recovery under the review budget',chart=charts['test'],bullets=[f'{len(test):,} test transactions; {int(y.sum()):,} actual fraud cases.','Model chosen on validation, before test-label evaluation.',f"Batch threshold: {test_decision['threshold']:.6f}; 2,500 reviews.",f'Precision: {tp/selected.sum():.2%}; recall: {tp/y.sum():.2%}.','1,001 recovered fraud cases; 1,499 legitimate alerts; 632 missed fraud cases.'],takeaway='Test spending is R$10,000; threshold recalculation uses scores, not test labels.'))
slides.append(dict(title='January 2026: delivered classifications',bullets=['44,451 transactions, with original columns and row order preserved.','Budget: R$4,000 for the complete January batch; R$4 per review.','Capacity: 1,000 reviews, or 2.25% of the batch.','HGB batch threshold: 0.933746.','Final column flag_fraude: 1 for selected review alerts, 0 otherwise.','File: datasets/transacoes_jan2026_previsoes.csv'],takeaway='January flags are model decisions, not confirmed fraud; actual precision and recall are unknown.'))
slides.append(dict(title='Conclusions and next steps',bullets=['Device novelty and segment combinations motivate behavioral features.','HGB provides the best observed validation recovery within the configured budget.','Held-out test recall: 61.30%, with 40.04% precision at 2,500 reviews.','Deliver 1,000 January alerts within the R$4,000 review budget.','Next: temporal validation, label-availability audit and feature ablations.','Monitor drift, realized review precision and budget use; calibrate scores if needed.'],takeaway='The delivery is a reproducible, capacity-aware review prioritization workflow.'))
slides.append(dict(title='Evidence and reproducibility',bullets=['EDA: datasets/transacoes_historico_2025.csv and challenge_artefact.ipynb.','Training: training_workflow.py; preprocessing.py.','Model metrics: artifacts/training/validation_metrics.csv.','Budget selection: validation_budget_metrics.csv; budget_threshold.py.','Test evaluation: test_predictions.csv; labels used only for measurement.','Delivery: datasets/transacoes_jan2026_previsoes.csv.','Original presentation preserved; figures and source accompany this update.'],takeaway='All reported operational metrics come from persisted predictions; no models were retrained.'))


# Separate the ranking chart from the comparison table to keep labels legible.
comparison_index = next(i for i, item in enumerate(slides) if item['title'].startswith('Model comparison:'))
slides[comparison_index].pop('chart')
slides.insert(comparison_index + 1, dict(
    title='Precision-recall curves: assess the complete ranking',
    chart=charts['pr'],
    bullets=['HGB and MLP rank fraud similarly across much of the recall range.',
             'Average precision summarizes ranking across score cutoffs.',
             'The dashed line is the validation fraud prevalence.',
             'The budget determines the operating point used for model selection.'],
    takeaway='A high ROC-AUC is useful context; precision-recall shows the review trade-off more directly.'
))


def draw_text(slide, x, y, width, height, text, size=18, color=DARK, bold=False):
    """
    Add editable text to a PowerPoint slide.

    Parameters
    ----------
        slide : pptx.slide.Slide
            Destination slide.
        x, y, width, height : float
            Text-box position and dimensions in inches.
        text : str
            Text content.
        size : int
            Font size in points.
        color : str
            Hexadecimal font color.
        bold : bool
            Whether to use bold text.

    Returns
    -------
        None
            The destination slide is updated in place.
    """
    # Use an editable text box with a consistent presentation typeface.
    box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(width),Inches(height))
    box.text_frame.word_wrap=True
    for i,line in enumerate(text.split('\n')):
        para=box.text_frame.paragraphs[0] if i==0 else box.text_frame.add_paragraph()
        para.text=line;para.font.size=Pt(size);para.font.bold=bold;para.font.name='Arial'
        para.font.color.rgb=RGBColor.from_string(color.lstrip('#'))


# Render a PDF and a PowerPoint with the same slide content and chart assets.
prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
from pptx.enum.shapes import MSO_SHAPE
pdf_path=OUT/'Fraud_Detection_Updated_Results.pdf'
with PdfPages(pdf_path) as pdf:
 for number,spec in enumerate(slides,1):
    fig=plt.figure(figsize=(13.333,7.5),facecolor='white')
    fig.add_artist(plt.Rectangle((0,.875),1,.125,transform=fig.transFigure,color=BLUE))
    fig.text(.035,.921,spec['title'],color='white',fontsize=22,va='center')
    slide=prs.slides.add_slide(prs.slide_layouts[6])
    shape=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width,Inches(.94));shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string(BLUE[1:]);shape.line.fill.background()
    draw_text(slide,.35,.18,12.6,.6,spec['title'],24,'#FFFFFF')
    has_chart='chart' in spec;has_table='table' in spec
    if 'subtitle' in spec:
        fig.text(.05,.79,spec['subtitle'],fontsize=20,color=BLUE)
        draw_text(slide,.65,1.35,12,.6,spec['subtitle'],22,BLUE)
    if has_chart:
        image=plt.imread(spec['chart']);ax=fig.add_axes([.405,.23,.57,.59]);ax.imshow(image);ax.axis('off')
        slide.shapes.add_picture(str(spec['chart']),Inches(5.4),Inches(1.35),width=Inches(7.55),height=Inches(4.6))
    y=.77 if not 'subtitle' in spec else .66
    bullet_width=43 if has_chart else 106
    if has_table:
        cols,rows=spec['table'];table_y=.70 if not has_chart else .50
        if has_chart: # Place the full comparison table on the left, with compact commentary below.
            rect=[.035,.45,.36,.35];font=8
        else:rect=[.045,.41,.91,.35];font=12
        ax=fig.add_axes(rect);ax.axis('off');tbl=ax.table(cellText=rows,colLabels=cols,loc='center',cellLoc='left',colLoc='left');tbl.auto_set_font_size(False);tbl.set_fontsize(font);tbl.scale(1,2.05 if not has_chart else 1.8)
        for (r,c),cell in tbl.get_celld().items():
            cell.set_edgecolor('white');cell.set_facecolor(BLUE if r==0 else ('#EDF0FF' if r%2 else '#F6F7FB'))
            if r==0:cell.get_text().set_color('white')
        # Use native editable PowerPoint table cells.
        tx,ty,tw,th=(.45,1.45,4.8,2.6) if has_chart else (.6,1.45,12.1,2.65)
        table=slide.shapes.add_table(len(rows)+1,len(cols),Inches(tx),Inches(ty),Inches(tw),Inches(th)).table
        for r,row in enumerate([cols]+rows):
            for c,value in enumerate(row):
                cell=table.cell(r,c);cell.text=str(value);cell.fill.solid();cell.fill.fore_color.rgb=RGBColor.from_string(BLUE[1:] if r==0 else 'EDF0FF')
                for para in cell.text_frame.paragraphs:
                    para.font.size=Pt(10 if has_chart else 15);para.font.color.rgb=RGBColor.from_string('FFFFFF' if r==0 else DARK[1:])
        y=.37 if has_chart else .39
    bullet_size=14 if has_chart else 17
    if has_table:bullet_size=11 if has_chart else 14
    for bullet in spec['bullets']:
        lines=textwrap.wrap(bullet,width=bullet_width)
        text='\n'.join(lines)
        fig.text(.045,y,'\u25b8',color=MAGENTA,fontsize=bullet_size+2,va='top')
        fig.text(.065,y,text,fontsize=bullet_size,color=DARK,va='top',linespacing=1.35)
        box_y=(1-y)*7.5-.06
        draw_text(slide,.78,box_y,4.4 if has_chart else 11.9,.28+len(lines)*.25,'\u25b8 '+bullet,bullet_size+1)
        y-=len(lines)*(.031 if has_table else .039)+(.018 if has_chart else .025)
    fig.add_artist(plt.Rectangle((.035,.09),.93,.085,transform=fig.transFigure,color='#EDF0FF',zorder=0))
    wrapped='\n'.join(textwrap.wrap(spec['takeaway'],width=114))
    fig.text(.05,.133,wrapped,color=BLUE,fontsize=12,va='center',weight='bold')
    draw_text(slide,.6,6.25,12.1,.56,spec['takeaway'],15,BLUE,True)
    footer=f'Leonardo Correia  |  Fraud Detection  |  Updated results                                       {number} / {len(slides)}'
    fig.text(.04,.035,footer,color=GRAY,fontsize=9)
    draw_text(slide,.5,7.05,12,.25,footer,10,GRAY)
    pdf.savefig(fig)
    if number in [1,7,12,14,17]:fig.savefig(ASSETS/f'preview_slide_{number:02d}.png',dpi=95)
    plt.close(fig)
prs.save(OUT/'Fraud_Detection_Updated_Results.pptx')
(OUT/'Fraud_Detection_Updated_Results.md').write_text('\n\n'.join(f"## {i}. {s['title']}\n\n"+'\n'.join('- '+b for b in s['bullets'])+ ('\n\n'+pd.DataFrame(s['table'][1],columns=s['table'][0]).to_csv(index=False) if 'table'in s else '')+'\n\n'+s['takeaway'] for i,s in enumerate(slides,1)))
print(f'Created {len(slides)} slides: {pdf_path.name} and editable PowerPoint.')
print('Monthly rates:',monthly.to_dict())
