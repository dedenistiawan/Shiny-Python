import pandas as pd
import numpy as np
from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_widget
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import io
import base64

app_ui = ui.page_fluid(
    ui.h1("C4.5 Decision Tree with Custom Dataset"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_file("file", "Upload Excel File", accept=[".xls", ".xlsx"]),
            ui.input_checkbox("header", "Header", True),
            ui.output_ui("select_label"),
            ui.input_slider("train_ratio", "Training Set Ratio:", 0.5, 0.9, 0.7, step=0.1),
            ui.input_action_button("train_button", "Train Model"),
            ui.output_ui("custom_inputs"),
            ui.input_action_button("predict_button", "Predict Custom Data")
        ),
        ui.panel_main(
            ui.navset_tab(
                ui.nav_panel("Tree Plot", output_widget("tree_plot")),
                ui.nav_panel("Model Summary", ui.output_text_verbatim("model_summary")),
                ui.nav_panel("Confusion Matrix", ui.output_text_verbatim("conf_matrix")),
                ui.nav_panel("Custom Prediction", ui.output_text_verbatim("custom_prediction"))
            )
        )
    )
)

def server(input, output, session):
    @reactive.Effect
    @reactive.event(input.file)
    def update_label_dropdown():
        if input.file() is None:
            return
        df = parse_contents(input.file(), input.header())
        output.select_label.set_render(
            ui.input_select("label", "Select Label Column:", choices=list(df.columns))
        )

    @reactive.Effect
    @reactive.event(input.train_button)
    def update_custom_inputs():
        if input.file() is None or input.label() is None:
            return
        df = parse_contents(input.file(), input.header())
        features = df.drop(columns=[input.label()]).columns
        output.custom_inputs.set_render(
            ui.TagList(
                *[ui.input_numeric(f"input_{feature}", f"Enter value for {feature}", 0) for feature in features]
            )
        )

    @reactive.Calc
    def train_model():
        if input.file() is None or input.label() is None:
            return None
        df = parse_contents(input.file(), input.header())
        X = df.drop(columns=[input.label()])
        y = df[input.label()]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=input.train_ratio(), random_state=123)
        
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {"model": model, "conf_matrix": conf_matrix, "report": report, "features": X.columns}

    @output
    @render_widget
    def tree_plot():
        model_data = train_model()
        if model_data is None:
            return go.Figure()
        
        model = model_data["model"]
        fig = go.Figure()
        plot_tree(model, feature_names=model_data["features"], filled=True)
        return fig

    @output
    @render.text
    def model_summary():
        model_data = train_model()
        if model_data is None:
            return ""
        report = model_data["report"]
        return classification_report(report)

    @output
    @render.text
    def conf_matrix():
        model_data = train_model()
        if model_data is None:
            return ""
        return str(model_data["conf_matrix"])

    @output
    @render.text
    @reactive.event(input.predict_button)
    def custom_prediction():
        model_data = train_model()
        if model_data is None:
            return ""
        input_data = np.array([input[f"input_{feature}"]() for feature in model_data["features"]]).reshape(1, -1)
        prediction = model_data["model"].predict(input_data)
        return f"Custom Data Prediction: {prediction[0]}"

def parse_contents(file, header):
    content = file[0]["datapath"]
    df = pd.read_excel(content, header=0 if header else None)
    return df

app = App(app_ui, server)

# Run the app
if __name__ == '__main__':
    app.run()
