from shiny import App, render, ui
import matplotlib.pyplot as plt
import numpy as np

# Nest Python functions to build an HTML interface
app_ui = ui.page_fluid( # Layout the UI with Layout Functions
 
  #  Add Inputs with ui.input_*() functions 
  ui.input_slider(
    "n", "Sample Size", 0, 1000, 20
  ),
  # Add Outputs with ui.ouput_*() functions
  ui.output_plot("dist")
)

def server(input, output, session):
  
  # For each output, define a function that generates the output
  @render.plot # Specify the type of output with a @render. decorator
  def dist(): # Use the output id as the function name
    
    # Call the values of UI inputs with input.<id>()
    x = np.random.randn(input.n()) 
    plt.hist(x, range=[-3, 3])

# Call App() to combine app_ui and server() into an interactive app
app = App(app_ui, server)
app.run()