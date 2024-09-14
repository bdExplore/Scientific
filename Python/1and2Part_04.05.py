import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd

x = np.arange(0, 5, 0.1)
fig = px.scatter(x=x, y=x)
fig.show()