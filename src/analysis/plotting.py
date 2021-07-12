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
                          save_path: Optional[str] = None, hover_data=None, show_legend: bool = False):

        labels = {
            "value": x_title,
            "index": "count",
        }

        fig = px.histogram(values, title=title, nbins=bins, labels=labels, marginal='box', opacity=0.5,
                           hover_data=hover_data)
        fig.layout.update(showlegend=show_legend)

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

