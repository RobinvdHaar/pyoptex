import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from .metric import Aopt, Dopt, Iopt
from .wrapper import default_fn, create_parameters
from ..utils.model import encode_names, model2names, encode_model, model2encnames
from ..utils.design import x2fx, encode_design
from ..constraints import no_constraints

def evaluate_metrics(Y, metrics, factors, fn):
    assert isinstance(Y, pd.DataFrame), 'Y must be a denormalized and decoded dataframe'
    Y = Y.copy()

    # Create the design parameters
    params = create_parameters(factors, fn)

    # Normalize Y
    for f in factors:
        Y[str(f.name)] = f.normalize(Y[str(f.name)])

    # Transform Y to numpy
    col_names = [str(f.name) for f in factors]
    Y = Y[col_names].to_numpy()

    # Encode the design
    Y = encode_design(Y, params.effect_types, params.coords)

    # Define the metric inputs
    X = params.fn.Y2X(Y)

    # Initialize the metrics
    for metric in metrics:
        metric.preinit(params)
        metric.init(Y, X, params)

    # Compute the metrics
    return [metric.call(Y, X, params) for metric in metrics]

def fraction_of_design_space(Y, factors, fn, iopt_N=10000, return_params=False):
    assert isinstance(Y, pd.DataFrame), 'Y must be a denormalized and decoded dataframe'
    Y = Y.copy()

    # Create the design parameters
    params = create_parameters(factors, fn)

    # Normalize Y
    for f in factors:
        Y[str(f.name)] = f.normalize(Y[str(f.name)])

    # Transform Y to numpy
    col_names = [str(f.name) for f in factors]
    Y = Y[col_names].to_numpy()

    # Encode the design
    Y = encode_design(Y, params.effect_types, params.coords)

    # Define the metric inputs
    X = params.fn.Y2X(Y)
    
    # Initialize Iopt
    iopt = Iopt(n=iopt_N, cov=fn.metric.cov)
    iopt.preinit(params)
    iopt.init(Y, X, params)

    # Compute information matrix
    if iopt.cov is not None:
        _, X = iopt.cov(Y, X)
    M = X.T @ params.Vinv @ X

    # Compute prediction variances
    pred_var = np.sum(iopt.samples.T * np.linalg.solve(M, np.broadcast_to(iopt.samples.T, (M.shape[0], *iopt.samples.T.shape))), axis=-2)
    pred_var = np.sort(pred_var)

    if return_params:
        return pred_var, params
    return pred_var

def plot_fraction_of_design_space(Y, factors, fn, iopt_N=10000):
    # Compute prediction variances
    pred_var, params = fraction_of_design_space(Y, factors, fn, iopt_N=iopt_N, return_params=True)

    # Create the figure
    fig = go.Figure()
    for i, pv in enumerate(pred_var):
        color = DEFAULT_PLOTLY_COLORS[i]
        name = ', '.join([f'plot {i+1} = {r:.3f}' for i, r in enumerate(params.ratios[i])])
        fig.add_trace(go.Scatter(x=np.linspace(0, 1, len(pv)), y=pv, marker_color=color, name=name))
        fig.add_hline(y=np.mean(pv), annotation_text=f'{np.mean(pv):.3f}', annotation_font_color=color, 
                        line_dash='dash', line_width=1, line_color=color, annotation_position='bottom right')

    # Set axis
    fig.update_layout(
        xaxis_title='Fraction of design space',
        yaxis_title='Relative prediction variance',
        legend_title_text='A-priori variance ratios',
        title='Fraction of design space plot',
        title_x=0.5
    )

    return fig

def estimation_variance_matrix(Y, factors, fn, return_params=False):
    assert isinstance(Y, pd.DataFrame), 'Y must be a denormalized and decoded dataframe'
    Y = Y.copy()
    
    # Create the design parameters
    params = create_parameters(factors, fn)

    # Normalize Y
    for f in factors:
        Y[str(f.name)] = f.normalize(Y[str(f.name)])

    # Transform Y to numpy
    col_names = [str(f.name) for f in factors]
    Y = Y[col_names].to_numpy()

    # Encode the design
    Y = encode_design(Y, params.effect_types, params.coords)

    # Define the metric inputs
    X = params.fn.Y2X(Y)

    # Compute information matrix
    if fn.metric.cov is not None:
        _, X = fn.metric.cov(Y, X)
    M = X.T @ params.Vinv @ X

    # Compute inverse of information matrix
    Minv = np.linalg.inv(M)

    # Create figure
    if return_params:
        return Minv, params
    return Minv

def plot_estimation_variance_matrix(Y, factors, fn, model=None):
    # Compute estimation variance matrix
    Minv, params = estimation_variance_matrix(Y, factors, fn, return_params=True)

    # Determine the encoded column names
    if model is None:
        encoded_colnames = np.arange(Minv.shape[-1])
    else:
        col_names = [str(f.name) for f in factors]
        encoded_colnames = model2encnames(model[col_names], params.effect_types)
        if len(encoded_colnames) < Minv.shape[-1]:
            encoded_colnames.extend([f'cov_{i}' for i in range(Minv.shape[-1] - len(encoded_colnames))])

    # Create the figure
    fig = make_subplots(rows=len(Minv), cols=1, row_heights=list(np.ones(len(Minv))/len(Minv)), 
        vertical_spacing=0.07,
        subplot_titles=[
            'A-priori variance ratios: ' + ', '.join([f'plot {i+1} = {r:.3f}' for i, r in enumerate(params.ratios[i])])
            for i in range(len(Minv))
        ]
    )
    for i in range(len(Minv)):
        fig.add_trace(go.Heatmap(
            z=np.flipud(Minv[i]), x=encoded_colnames, y=encoded_colnames[::-1], colorbar_len=0.75/len(Minv),
            colorbar_x=1, colorbar_y=1-(i+0.75/2+0.05*i)/len(Minv)
        ), row=i+1, col=1)
    fig.update_layout(
        title='Estimation covariance plot',
        title_x=0.5
    )

    # Return the plot
    return fig

def estimation_variance(Y, factors, fn):
    # Compute estimation variance matrix
    Minv = estimation_variance_matrix(Y, factors, fn)
    return np.stack([np.diag(Minv[i]) for i in range(len(Minv))])
