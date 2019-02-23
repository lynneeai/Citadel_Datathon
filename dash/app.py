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
#df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
df = pd.read_csv('../data/fake_data.csv')

# Static data info
years = list(df['Date'])
n = 2 # number of renewable energies = 5
renew_energies = list(df.columns)[1:n+1] # n renewable energies

# update curve
trace_high = go.Scatter(
    x=df.Date,
    y=df['solar'],
    name = "High",
    line = dict(color = '#17BECF'),
    opacity = 0.8)

trace_low = go.Scatter(
    x=df.Date,
    y=df['Hydro'],
    name = "Low",
    line = dict(color = '#7F7F7F'),
    opacity = 0.8)

trace_sum = go.Scatter(
    x=df.Date,
    y=df['Hydro']+df['solar'],
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
                {'label': 'Solar', 'value': 'solar'},
                {'label': 'Hydro', 'value': 'Hydro'},
                {'label': 'Sum', 'value': 'S'}
            ],
            values=['solar', 'Hydro']
        ),

        # Graph
        dcc.Graph(
            id='simulator-graph',
            figure={
                'data': data,
                'layout': layout
            }
        ),

        # Text
        dcc.Input(
            id='solar-year',
            value='Year', 
            type='text'
        ),

        # Slider
        dcc.Slider(
            id='solar-slider',
            min=0,
            max=10,
            step=0.01,
            marks={i: str(i) for i in range(1, 10)},
            value=0
        ),

    ]
)

line_color = ['#17BECF', '#7F7F7F', '#33CFA5']

# Callback
@app.callback(
    dash.dependencies.Output('simulator-graph', 'figure'),
    [dash.dependencies.Input('energy-checklist', 'values')])
def update_graph(check_values):

    # copy of data
    dff = df.copy()
    years = [1969]*n
    rates = [0.05]*n

    # compute adjusted
    starts = [years.index(i) for i in years]
    lists = [dff[i].tolist() for i in renew_energies]
    lists = [(lists[j][0:starts[j]] + [ i * (1 + rates[j]) for i in lists[j][starts[j]:]]) for j in range((n))]
    idx = 0

    # update values
    for e in renew_energies:
        dff.drop([e], axis=1, inplace = True)
        dff[e] = lists[idx]
        idx += 1

    # return new graph
    return {
        'data': [go.Scatter(
                    x=dff.Date,
                    y=dff[check_values[i]],
                    name = check_values[i],
                    line = dict(color = line_color[i]),
                    opacity = 0.8) for i in range(len(check_values))
                ]+[go.Scatter(
                    x=dff.Date,
                    y=sum(dff[check_values[i]] for i in range(len(check_values))),
                    name = "Sum",
                    line = dict(color = '#33CFA5'),
                    opacity = 0.8)
                ],
        'layout': layout
    }

# Main
if __name__ == '__main__':
    app.run_server(debug=True)
