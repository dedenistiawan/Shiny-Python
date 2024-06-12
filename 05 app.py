# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:55:28 2022

@author: dfs20
"""
# explicit import of jinja2 required for shinylive 0.0.7
import jinja2

from shiny import App, ui, render
import pandas as pd
import plotnine as gg

# read data from file
# jr = pd.read_csv("jr_shiny.csv")

## BEGIN: shinylive specific modification for loading 
## data file on hosted app
from pathlib import Path
jr = pd.read_csv(Path(__file__).parent / "jr_shiny.csv")
## END

jr = jr.astype({"day": "object"})

# dictionary of choices for input_select
# x axis of our graph of form {"value": "UI label"}
choices_select = {
    "year": "Year",
    "day": "Day",
    "hour": "Hour",
    "media_type": "Media"
}

# dictionary of choices for checkbox group
# factors for table of form {"value": "UI label"}
choices_check = {
    "created_at": "Date",
    "text": "Text",
    "retweet_count": "Retweets",
    "favorite_count": "Likes"
}

app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.panel_sidebar(
            # user inputs in the sidebar
            ui.input_select(
                id="x", label="X-axis Variable", choices=choices_select, selected="year"
            ),
            ui.input_numeric(
                id="num",
                label="How many tweets do you want to view?",
                value=0,
                min=0,
                max=50
            ),
            ui.panel_conditional(
                # a client-side condition for whether to display this panel
                "input.num > 0 && input.num <= 50", 
                ui.input_checkbox_group(
                    id="cols",
                    label="Select which variables you want to view:",
                    choices=choices_check,
                    selected=(["created_at", "text"]),
                )
            )
        ),
        ui.panel_main(
            ui.output_plot("plot"), ui.output_text("text"), ui.output_table("table")
        )
    )
)



def server(input, output, session):
    @output
    @render.plot
    def plot(): # function name matches the id="plot" in the outputs
        # access input id="x" value with input.x()
        avg_int = jr.groupby(input.x(), as_index=False).agg(
            {"retweet_count": "mean", "favorite_count": "mean"}
        )
        avg_int = pd.melt(avg_int, id_vars=input.x())
        plot = (
            gg.ggplot(avg_int, gg.aes(input.x(), "value", fill="variable"))
            + gg.geom_col(position="dodge")
            + gg.ylab("Average Interactions")
            + gg.scale_fill_brewer(
                type="qual",
                palette="Dark2",
                name="Interaction",
                labels=(["Like", "Retweet"]),
            )
            + gg.theme_classic()
        )
        if input.x() == "day":
            return plot + gg.scale_x_discrete(
                labels=(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            )
        else:
            return plot

    @output
    @render.text
    def text():
        if input.cols() == () or input.num() <= 0 or input.num() > 50:
            return ""
        elif input.num() == 1:
            return "Displaying the most recent @jumping_uk tweet:"
        else:
            return f"Displaying the {input.num()} most recent @jumping_uk tweets:"

    @output
    @render.table
    def table():
        cols = jr.filter(input.cols())
        cols.rename(
            columns={
                "created_at": "Date",
                "text": "Text",
                "retweet_count": "Retweets",
                "favorite_count": "Likes",
            },
            inplace=True,
        )
        pd.set_option("colheader_justify", "left")
        first_n = cols.head(input.num())
        if input.num() <= 0 or input.num() > 50:
            return None
        else:
            return first_n


app = App(app_ui, server)
app.run()