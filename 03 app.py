# Core
from shiny import ui, render, App

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_text("txt_in", "Type something here:"),
        open="always",
    ),
    ui.card(
        ui.output_code("result"),
    )
)

def server(input, output, session):
    @render.code
    def result():
        return f"You entered '{input.txt_in()}'."

app = App(app_ui, server)
app.run()