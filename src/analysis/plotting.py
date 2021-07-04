from typing import Optional, List
import plotly.express as px


class Plotter:

    @staticmethod
    def plot_correlation(x_values: List[float], y_values: List[float], save_path: Optional[str] = None):
        fig = px.scatter(x=x_values, y=y_values)
        if save_path is not None:
            fig.write_html(save_path)
        fig.show()

    @staticmethod
    def plot_loss(values: List[float], save_path: Optional[str] = None, x_label: str = 'Epoch'):

        labels = {
            "value": "Loss",
            "index": x_label,
        }
        fig = px.line(values, title=f'Losses over {x_label}s', labels=labels)
        fig.layout.update(showlegend=False)
        if save_path is not None:
            fig.write_html(save_path)
        fig.show()

    @staticmethod
    def plot_distribution(values, title: str, x_title: str, bins: Optional[int] = None,
                          save_path: Optional[str] = None, hover_data=None):

        labels = {
            "value": x_title,
            "index": "count",
        }

        fig = px.histogram(values, title=title, nbins=bins, labels=labels, marginal='box', opacity=0.5,
                           hover_data=hover_data)
        fig.layout.update(showlegend=False)

        if save_path is not None:
            fig.write_html(save_path)
        fig.show()

    @staticmethod
    def plot_parallel_coordinates(values, color: str, dimensions: List[str], color_continuous_midpoint: int,
                                  save_path: Optional[str] = None):
        fig = px.parallel_coordinates(values, color=color, dimensions=dimensions,
                                      color_continuous_midpoint=color_continuous_midpoint)
        if save_path is not None:
            fig.write_html(save_path)
        fig.show()

    @staticmethod
    def plot_scatter(values, x: str, y: str, color: str, save_path: Optional[str] = None):
        fig = px.scatter(values, x, y, color=color, opacity=0.7)
        fig.update_traces(marker=dict(size=12))
        if save_path is not None:
            fig.write_html(save_path)
        fig.show()


if __name__ == '__main__':
    import pandas as pd
# csv_path = '/Users/chris/Google Drive/MARTIN/output/evaluation/bert2bert_big_results.csv'
# df_bert2bert_big = pd.read_csv(csv_path)
# df_bert2bert_big = df_bert2bert_big[['Cosine_similarity', 'Normal_length', 'Simple_length', 'SARI', 'METEOR']]
# df_bert2bert_big['Model'] = 0
# df_bert2bert_big['Model_name'] = 'Bert2Bert_big'
#
# csv_path = '/Users/chris/Google Drive/MARTIN/output/evaluation/bert2bert_small_results.csv'
# df_bert2bert_small = pd.read_csv(csv_path)
# df_bert2bert_small = df_bert2bert_small[['Cosine_similarity', 'Normal_length', 'Simple_length', 'SARI', 'METEOR']]
# df_bert2bert_small['Model'] = 1
# df_bert2bert_small['Model_name'] = 'Bert2Bert_small'
#
# csv_path = '/Users/chris/Google Drive/MARTIN/output/evaluation/newsela2newsela_small_results.csv'
# df_news2news_small = pd.read_csv(csv_path)
# df_news2news_small = df_news2news_small[['Cosine_similarity', 'Normal_length', 'Simple_length', 'SARI', 'METEOR']]
# df_news2news_small['Model'] = 2
# df_news2news_small['Model_name'] = 'Newsela2Newsela_small'
#
# csv_path = '/Users/chris/Google Drive/MARTIN/output/evaluation/newsela2newsela_big_results.csv'
# df_news2news_big = pd.read_csv(csv_path)
# df_news2news_big = df_news2news_big[['Cosine_similarity', 'Normal_length', 'Simple_length', 'SARI', 'METEOR']]
# df_news2news_big['Model'] = 3
# df_news2news_big['Model_name'] = 'Newsela2Newsela_big'
#
# csv_path = '/Users/chris/Google Drive/MARTIN/output/evaluation/roberta2roberta_big_results.csv'
# df_rob2rob_big = pd.read_csv(csv_path)
# df_rob2rob_big = df_rob2rob_big[['Cosine_similarity', 'Normal_length', 'Simple_length', 'SARI', 'METEOR']]
# df_rob2rob_big['Model'] = 4
# df_rob2rob_big['Model_name'] = 'Roberta2Roberta_big'
#
# csv_path = '/Users/chris/Google Drive/MARTIN/output/evaluation/roberta2roberta_small_results.csv'
# df_rob2rob_small = pd.read_csv(csv_path)
# df_rob2rob_small = df_rob2rob_small[['Cosine_similarity', 'Normal_length', 'Simple_length', 'SARI', 'METEOR']]
# df_rob2rob_small['Model'] = 5
# df_rob2rob_small['Model_name'] = 'Roberta2Roberta_small'
#
# df = pd.concat(
#     [df_bert2bert_big, df_bert2bert_small, df_news2news_small, df_news2news_big, df_rob2rob_big, df_rob2rob_small])
#
# labels = {
#     "Cosine_similarity": 'Cosine similarity',
#     "Normal_length": "Target length",
#     "Simple_length": "Prediction length",
#     "SARI": "SARI score"
# }
#
# fig = px.parallel_coordinates(df.sample(frac=1), labels=labels, color='Model', dimensions=["Cosine_similarity",
#                                                                                            "Normal_length",
#                                                                                            "Simple_length", "SARI"],
#                               color_continuous_midpoint=2.5)
# fig.show()
# fig.write_html('coordinates')

# =========================================================================

# csv_path = '/Users/chris/Google Drive/MARTIN/output/evaluation/bert2bert_small_results_config.csv'
# bert_small = pd.read_csv(csv_path)
# bert_small['Model'] = 'Bert2Bert'
# csv_path = '/Users/chris/Google Drive/MARTIN/output/evaluation/bert2bert_big_results_config.csv'
# bert_big = pd.read_csv(csv_path)
# bert_big['Model'] = 'Bert2Bert+'
# csv_path = '/Users/chris/Google Drive/MARTIN/output/evaluation/newsela2newsela_small_results_config.csv'
# newsela_small = pd.read_csv(csv_path)
# newsela_small['Model'] = 'Newsela2Newsela'
# csv_path = '/Users/chris/Google Drive/MARTIN/output/evaluation/newsela2newsela_big_results_config.csv'
# newsela_big = pd.read_csv(csv_path)
# newsela_big['Model'] = 'Newsela2Newsela+'
# csv_path = '/Users/chris/Google Drive/MARTIN/output/evaluation/roberta2roberta_big_results_config.csv'
# roberta_big = pd.read_csv(csv_path)
# roberta_big['Model'] = 'Roberta2Roberta+'
# csv_path = '/Users/chris/Google Drive/MARTIN/output/evaluation/roberta2roberta_small_results_config.csv'
# roberta_small = pd.read_csv(csv_path)
# roberta_small['Model'] = 'Roberta2Roberta'
#
# dff = pd.concat([bert_small, bert_big, newsela_small, newsela_big, roberta_small, roberta_big])
#
# labels = {
#     "SARI": 'SARI score',
#     "temperature": "Temperature",
# }
#
# fig = px.scatter(dff, y='SARI', x='temperature',labels=labels, color='Model', opacity=0.7)
#
# fig.update_traces(marker=dict(size=12))
# fig.show()
# fig.write_html('scatter.html')

# =======================================

# csv_path = '/Users/chris/Google Drive/MARTIN/data/_collections/proc_w_s_2_columns_shuffle_with_meta_data_without_low_similarity.csv'
# df = pd.read_csv(csv_path)
# df1 = df[['Normal_length', 'Simple_length']]
# df2 = df[['Cosine_similarity']]
# Plotter.plot_distribution(df1, title='Text length distribution', x_title='Length', bins=80, save_path='1.html')
# Plotter.plot_distribution(df2, title='Cosine similarity distribution', x_title='Cosine similarity', bins=80, save_path='2.html')
