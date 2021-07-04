import re
import pandas as pd


class TextPreprocessor:

    @staticmethod
    def clean_up_series(series: pd.Series) -> pd.Series:
        series = series.apply(lambda x: TextPreprocessor.clean_up(x))
        return series

    @staticmethod
    def normalize_series(series: pd.Series) -> pd.Series:
        series = series.apply(lambda x: TextPreprocessor.normalize_string(x))
        return series

    @staticmethod
    def clean_up(text):
        text = text.replace('–', '-')
        text = text.replace(' ,', ',')
        text = text.replace(' .', '.')
        text = text.replace(' ;', ';')
        text = text.replace('\n', '')
        text = re.sub(r' +', ' ', text)
        return text

    @staticmethod
    def normalize_string(text):
        text = re.sub(r'([.!?])', r' \1', text)
        text = re.sub(r'[^a-zA-Z.!?]+', r' ', text)
        text = TextPreprocessor.clean_up(text)
        return text


class DataPreparer:

    @staticmethod
    def remove_low_similarity_records(threshold: float, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df['Cosine_similarity'] > threshold]
        df = df[df['Normal_length'] > 7]
        df = df[df['Simple_length'] > 7]
        df = df[df['Normal_length'] <= 80]
        df = df[df['Simple_length'] <= 80]
        return df


if __name__ == '__main__':
    import os
    from src import Plotter

    # path = os.path.abspath(__file__ + '/../../../data/_collections/proc_w_s_2_columns_shuffle_with_meta_data.csv')
    # newpath = os.path.abspath(
    #     __file__ + '/../../../data/_collections/proc_w_s_2_columns_shuffle_with_meta_data_without_low_similarity.csv')
    # df = pd.read_csv(path)
    # print(df.shape)
    #
    # # Plotter.plot_distribution(df['Cosine_similarity'], title="OLD", x_title='x')
    #
    # newdf = DataPreparer.remove_low_similarity_records(0.6, df)
    # print(newdf.shape)
    #
    # newdf.to_csv(newpath, index=False)
    #
    # # Plotter.plot_distribution(newdf['Cosine_similarity'], title="NEW", x_title='x')
    #
    # df2 = DataPreparer.remove_low_similarity_records()

    old = os.path.abspath(__file__ + '/../../../data/_collections/proc_w_s_2_columns_shuffle_with_meta_data.csv')
    path = os.path.abspath(
        __file__ + '/../../../data/_collections/proc_w_s_2_columns_shuffle_with_meta_data_without_low_similarity.csv')
    # df = pd.read_csv(path)
    olda = pd.read_csv(old)
    print(olda.shape)
    print(olda['Normal_length'].max())

    new = DataPreparer.remove_low_similarity_records(0.6, olda)
    print(new.shape)
    # new = new[:100]
    # new.to_csv(path, index=False)

    frame = {'Simple': new['Simple_length'], 'Normal': new['Normal_length']}

    new_df = pd.DataFrame(frame)

    print(new_df.shape)
    Plotter.plot_distribution(new_df, title="NEW", x_title='x', bins=10)
    # Plotter.plot_correlation(new['Normal_length'], new['Cosine_similarity'])