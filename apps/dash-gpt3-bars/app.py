import black
import os
from textwrap import dedent
import pandas as pd
import os.path
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dash import no_update
from dash.dependencies import Input, Output, State
import plotly.express as px
import openai
import csv
import datetime

def Header(name, app):
    title = html.H1(name, style={"margin-top": 5})
    logo = html.Img(
        src=app.get_asset_url("dash-logo.png"), style={"float": "right", "height": 60}
    )
    return dbc.Row([dbc.Col(title, md=8), dbc.Col(logo, md=4)])


# Authentication
openai.api_key = os.getenv("OPENAI_KEY")


# Define the prompt
desc = "Our zoo has three twenty giraffes, fourteen orangutans, 3 monkeys more than the number of giraffes we have."
code_exp = "px.bar(x=['giraffes', 'orangutans', 'monkeys'], y=[20, 14, 23], labels=dict(x='animals', y='count'), title='Our Zoo')"
formatted_exp = black.format_str(code_exp, mode=black.FileMode(line_length=50))

# Create
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

content_style = {"height": "475px"}

controls = [
    dbc.InputGroup(
        [
            dbc.Input(
                id="input-text",
                placeholder="Specify what you want GPT-3 to generate...",
            ),
            dbc.InputGroupAddon(
                dbc.Button("Submit", id="button-submit", color="primary"),
                addon_type="append",
            ),
        ]
    )
]
output_graph = [
    dbc.CardHeader("Plotly Express Graph"),
    dbc.CardBody(dcc.Graph(id="output-graph", style={"height": "400px"})),
]
output_code = [
    dbc.CardHeader("GPT-3 Generated Code"),
    dcc.Markdown(id="output-code", style={"margin": "50px 5px"}),
]

#Survey Section 
#Aims to collect data/metrics from the user
survey = [
    dbc.CardHeader("When you are done, please let us know how we did."),
    html.Div([
        dbc.Label("Check the boxes that you agree with."),
        dbc.Checklist(
            options=[
                {"label": "The graphs provided were accurate", "value": 1},
                {"label": "The code provided was accurate", "value": 2},
                {"label": "The product was helpful", "value": 3},
                {"label": "The product was fast", "value": 4},
                {"label": "You would recommend this product", "value": 5},
            ],
            id="survey-input",
            value = 'show',
            )
        ]
    ),
]

explanation = f"""
*GPT-3 can generate Plotly graphs from a simple description of what you want!
We only needed to give the following prompt to GPT-3:*

Description: **{desc}**

Code:
```
{code_exp}
```
"""
explanation_card = [
    dbc.CardHeader("What am I looking at?"),
    dbc.CardBody(dcc.Markdown(explanation)),
]



app.layout = dbc.Container(
    [
        Header("Dash GPT-3 Chart Generation", app),
        html.Hr(),
        html.Div(controls, style={"padding-bottom": "15px"}),
        dbc.Spinner(
            dbc.Row(
                [
                    dbc.Col(dbc.Card(comp, style=content_style))
                    for comp in [output_graph, output_code]
                ],
                style={"padding-bottom": "15px"},
            )
        ),
        dbc.Card(explanation_card),
        #Added Survey Card here
        dbc.Card([dbc.Form(survey),
        #Submit button added for survey
        html.Button('Submit Survey', id='submit-val', n_clicks=0),
        html.P(id="survey-output")],style = {'display':'block'},id = 'card_id'
        ),
    ],
    fluid=False,
)



#Survey call back
@app.callback(
    Output("survey-output", "children"),
    [Input("survey-input", "value"),
    Input('submit-val', 'n_clicks'),
    ]
)
#Function to collect and store metrics
#Saves metrics as CSV to local 
def report_metrics(survey_value,n_clicks):
    #If submit button is clicked -> then send results to csv
    if n_clicks > 0:
        d = {"Graph_Acc?": 1,
        "Code_Acc?": 2,
        "Helpfulness?": 3,
        "fast?": 4,
        "Recommendation?": 5}
        x = []
        for k, v in d.items():
            if v in survey_value:
                x.append(1)
            else:
                x.append(0)

        #Create df
        df = pd.DataFrame(columns = d.keys())

        #Set row name to time and date of survey completion
        time = datetime.datetime.now()
        df.loc[time] = x

        #Save to csv 
        my_file = Path('/Users/jamesalfano/Documents/gpt3bar_Metrics.csv')

        #If file exists -> dont use header
        if my_file.is_file():
            df.to_csv('/Users/jamesalfano/Documents/gpt3bar_Metrics.csv',mode = 'a',header = None)
        #If file does not exist ->  Create header
        else:
             df.to_csv('/Users/jamesalfano/Documents/gpt3bar_Metrics.csv',mode = 'a')



#Callback to remove survey when finished
@app.callback(Output('card_id', 'style'),[Input('survey-output', 'value'),Input('submit-val', 'n_clicks')])
def toggle_survey(toggle_value,n_clicks):
    if n_clicks > 0:
        return {'display': 'none'}


@app.callback(
    [Output("output-graph", "figure"), Output("output-code", "children")],
    [Input("button-submit", "n_clicks"), Input("input-text", "n_submit")],
    [State("input-text", "value")],
)
def generate_graph(n_clicks, n_submit, text):
    if text is None:
        return dash.no_update, dash.no_update

    prompt = dedent(
        f"""
        description: {desc}
        code:
        {code_exp}

        description: {text}
        code:
        """
    ).strip("\n")

    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=100,
        stop=["description:", "code:"],
        temperature=0,
    )
    output = response.choices[0].text.strip()

    code = f"import plotly.express as px\nfig = {output}\nfig.show()"
    formatted = black.format_str(code, mode=black.FileMode(line_length=50))

    try:
        fig = eval(output).update_layout(margin=dict(l=35, r=35, t=35, b=35))
    except Exception as e:
        fig = px.line(title=f"Exception: {e}. Please try again!")

    return fig, f"```\n{formatted}\n```"


if __name__ == "__main__":
    app.run_server(debug=False)
