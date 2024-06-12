import dash
from dash import dcc, html, Input, Output, State
#import dash_table
from dash import dash_table
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import io
import base64
import numpy as np

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("C4.5 Decision Tree with Custom Dataset"),
    
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    
    dcc.Checklist(
        id='header',
        options=[{'label': 'Header', 'value': 'header'}],
        value=['header']
    ),
    
    html.Div(id='select-label-container'),
    
    dcc.Slider(
        id='train-ratio',
        min=0.5,
        max=0.9,
        step=0.1,
        value=0.7,
        marks={i: str(i) for i in np.arange(0.5, 1.0, 0.1)}
    ),
    html.Button('Train Model', id='train-button', n_clicks=0),
    
    html.Div(id='custom-inputs-container'),
    
    html.Button('Predict Custom Data', id='predict-button', n_clicks=0),
    
    dcc.Tabs([
        dcc.Tab(label='Tree Plot', children=[
            dcc.Graph(id='tree-plot')
        ]),
        dcc.Tab(label='Model Summary', children=[
            html.Pre(id='model-summary')
        ]),
        dcc.Tab(label='Confusion Matrix', children=[
            html.Pre(id='conf-matrix')
        ]),
        dcc.Tab(label='Custom Prediction', children=[
            html.Pre(id='custom-prediction')
        ]),
    ])
])

def parse_contents(contents, filename, header):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'xls' in filename:
        df = pd.read_excel(io.BytesIO(decoded), header=0 if 'header' in header else None)
    else:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=0 if 'header' in header else None)
    return df

@app.callback(
    Output('select-label-container', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('header', 'value')
)
def update_label_dropdown(contents, filename, header):
    if contents is None:
        return []
    df = parse_contents(contents, filename, header)
    return dcc.Dropdown(
        id='select-label',
        options=[{'label': col, 'value': col} for col in df.columns],
        placeholder="Select Label Column"
    )

@app.callback(
    Output('custom-inputs-container', 'children'),
    Input('train-button', 'n_clicks'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('header', 'value'),
    State('select-label', 'value'),
    State('train-ratio', 'value')
)
def generate_custom_inputs(n_clicks, contents, filename, header, label, train_ratio):
    if n_clicks == 0:
        return []
    df = parse_contents(contents, filename, header)
    features = df.drop(columns=[label]).columns
    return [dcc.Input(id=feature, type='number', placeholder=f'Enter value for {feature}') for feature in features]

@app.callback(
    Output('tree-plot', 'figure'),
    Output('model-summary', 'children'),
    Output('conf-matrix', 'children'),
    Output('custom-prediction', 'children'),
    Input('train-button', 'n_clicks'),
    Input('predict-button', 'n_clicks'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('header', 'value'),
    State('select-label', 'value'),
    State('train-ratio', 'value'),
    State({'type': 'input', 'index': dash.dependencies.ALL}, 'value')
)
def update_model(train_clicks, predict_clicks, contents, filename, header, label, train_ratio, custom_inputs):
    if contents is None:
        return {}, '', '', ''
    
    df = parse_contents(contents, filename, header)
    X = df.drop(columns=[label])
    y = df[label]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=123)
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    fig = px.imshow(conf_matrix, text_auto=True, labels=dict(x="Predicted", y="Actual"))
    
    summary = f"Classification Report:\n\n{report}"
    
    if custom_inputs is not None and len(custom_inputs) == len(X.columns):
        input_data = np.array(custom_inputs).reshape(1, -1)
        custom_pred = model.predict(input_data)
        custom_prediction = f"Custom Data Prediction: {custom_pred[0]}"
    else:
        custom_prediction = ''
    
    return fig, summary, str(conf_matrix), custom_prediction

if __name__ == '__main__':
    app.run_server(debug=True)
