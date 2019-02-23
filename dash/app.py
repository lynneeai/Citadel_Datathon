# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Style
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# Data
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
trace_high = go.Scatter(
    x=df.Date,
    y=df['AAPL.High'],
    name = "High",
    line = dict(color = '#17BECF'),
    opacity = 0.8)

trace_low = go.Scatter(
    x=df.Date,
    y=df['AAPL.Low'],
    name = "Low",
    line = dict(color = '#7F7F7F'),
    opacity = 0.8)

trace_sum = go.Scatter(
    x=df.Date,
    y=df['AAPL.Low']+df['AAPL.High'],
    name = "Sum",
    line = dict(color = '#33CFA5'),
    opacity = 0.8)

data = [trace_high,trace_low, trace_sum]

# Update
updatemenus = list([
    dict(type="buttons",
         active=-1,
         buttons=list([
            dict(label = 'High',
                 method = 'update',
                 args = [{'visible': [True, False, False]},
                         {'title': 'High',
                          'annotations': []}]),
            dict(label = 'Low',
                 method = 'update',
                 args = [{'visible': [False, True, False]},
                         {'title': 'Low',
                          'annotations': []}]),
            dict(label = 'Sum',
                 method = 'update',
                 args = [{'visible': [False, False, True]},
                         {'title': 'Sum',
                          'annotations': []}]),
            dict(label = 'All',
                 method = 'update',
                 args = [{'visible': [True, True, True]},
                         {'title': 'All',
                          'annotations': []}])
        ]),
    )
])

# === Slider ===
layout = dict(
    title='Energy Time Series',
    xaxis=dict(
        rangeslider=dict(
            visible = True
        ),
        type='date'
    ),
    showlegend=True,
    updatemenus=updatemenus,
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background']
)

# App Layout
app.layout = html.Div(
    style={'backgroundColor': colors['background']}, 
    children=[
        html.H1(
            children='The Future of Energy Simulator',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),

        html.Div(
            children='Energy: A life and death topic',
            style={
            'textAlign': 'center',
            'color': colors['text']
            }
        ),

        # Checklist
        dcc.Checklist(
            id='energy-checklist',
            options=[
                {'label': 'Solar', 'value': 'AAPL.High'},
                {'label': 'Hydro', 'value': 'AAPL.Low'},
                {'label': 'Sum', 'value': 'S'}
            ],
            values=['AAPL.High', 'AAPL.Low']
        ),

        # Graph
        dcc.Graph(
            id='simulator-graph',
            figure={
                'data': data,
                'layout': layout
            }
        ),

        # Slider
        dcc.Slider(
            id='solar-slider',
            min=0,
            max=10,
            step=0.01,
            marks={i: str(i) for i in range(1, 10)},
            value=0,
        )
    ]
)

line_color = ['#17BECF', '#7F7F7F', '#33CFA5']

# Callback
@app.callback(
    dash.dependencies.Output('simulator-graph', 'figure'),
    [dash.dependencies.Input('energy-checklist', 'values')])
def update_graph(check_values):

    return {
        'data': [go.Scatter(
                    x=df.Date,
                    y=df[check_values[i]],
                    name = check_values[i],
                    line = dict(color = line_color[i]),
                    opacity = 0.8) for i in range(len(check_values))
                ]+[go.Scatter(
                    x=df.Date,
                    y=sum(df[check_values[i]] for i in range(len(check_values))),
                    name = "Sum",
                    line = dict(color = '#33CFA5'),
                    opacity = 0.8)
                ],
        'layout': layout
    }

# Main
if __name__ == '__main__':
    app.run_server(debug=True)
