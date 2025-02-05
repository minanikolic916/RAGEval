from log_data.log_utils import read_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def visualize_log_eval_data(file_path:str, eval_method:str):
    #ucitavanje iz eval fajla
    metrics = read_data(file_path)
    df = pd.DataFrame(metrics)

    bar_width = 0.25
    colors = sns.color_palette("deep", 3)
    x = np.arange(len(df))

    #prikaz
    fig, ax = plt.subplots(figsize=(10, 8))
    labels_ragas = ['Faithfulness', 'Context precision', 'Response relevancy']
    labels_deepeval = ['Faithfulness', 'Contextual relevancy', 'Answer relevancy']
    if eval_method == "RAGAS":
        bars = [
            (labels_ragas[0], colors[0], df['faithfulness']),
            (labels_ragas[1], colors[1], df['llm_context_precision_without_reference']),
            (labels_ragas[2], colors[2], df['answer_relevancy'])
        ]
    else:
        bars = [
            (labels_deepeval[0], colors[0], df['faithfulness']),
            (labels_deepeval[1], colors[1], df['contextual_relevancy']),
            (labels_deepeval[2], colors[2], df['answer_relevancy'])]

    for i, (label, color, values) in enumerate(bars):
        bars_plot = ax.bar(x + (i - 1) * bar_width, values, width=bar_width, label=label, color=color)
        ax.bar_label(bars_plot, labels=[f'{v:.5f}' for v in values], padding=3, fontsize=8, color='black')
    
    df['combo_metrics'] = df['llm_evaluator'] + ' with '+ df['iterations'].astype(str) + ' folds'
    text_bellow_bars = df['combo_metrics']

    ax.set_xticks(x, text_bellow_bars)
    ax.set_xlabel('Evaluation metrics')
    ax.set_ylabel('Values')
    ax.set_title(f'RAG Evaluation with {eval_method}')
    ax.legend()

    plt.savefig(f"./visualizations/grafik_{eval_method}.png")


visualize_log_eval_data("./eval_data/eval_runs/EVAL_LOG_RAGAS.json", "RAGAS")
#visualize_log_eval_data("./eval_data/eval_runs/EVAL_LOG_DEEPEVAL.json")