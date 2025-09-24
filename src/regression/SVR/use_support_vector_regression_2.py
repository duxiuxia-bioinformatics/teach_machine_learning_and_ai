# https://towardsdatascience.com/support-vector-regression-svr-one-of-the-most-flexible-yet-robust-prediction-algorithms-4d25fbdaca60/

# ===========
# Setup
# Data Manipulation
import pandas as pd # for data manipulation
import numpy as np # for data manipulation

# Sklearn
from sklearn.linear_model import LinearRegression # for building a linear regression model
from sklearn.svm import SVR # for building SVR model
from sklearn.preprocessing import MinMaxScaler

# Visualizations
import plotly.graph_objects as go # for data visualization
import plotly.express as px # for data visualization

# ==================
# Read in data
df = pd.read_csv('Real estate.csv', encoding='utf-8')

# Use MinMax scaling on X2 and X3 features
scaler=MinMaxScaler()
df['X2 house age (scaled)']=scaler.fit_transform(df[['X2 house age']])
df['X3 distance to the nearest MRT station (scaled)']=scaler.fit_transform(df[['X3 distance to the nearest MRT station']])

# Print Dataframe
df

# ==========================
# SVR vs. simple linear regression - 1 independent variable

# Create a scatter plot
fig = px.scatter(df, x=df['X3 distance to the nearest MRT station'], y=df['Y house price of unit area'],
                 opacity=0.8, color_discrete_sequence=['black'])

# Change chart background color
fig.update_layout(dict(plot_bgcolor = 'white'))

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                 showline=True, linewidth=1, linecolor='black')

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                 showline=True, linewidth=1, linecolor='black')

# Set figure title
fig.update_layout(title=dict(text="House Price Based on Distance from the Nearest MRT",
                             font=dict(color='black')))

# Update marker size
fig.update_traces(marker=dict(size=3))

fig.show()

# ====================
# ------- Select variables -------
# Note, we need X to be a 2D array, hence reshape
X=df['X3 distance to the nearest MRT station'].values.reshape(-1,1)
y=df['Y house price of unit area'].values

# ------- Linear regression -------
model1 = LinearRegression()
lr = model1.fit(X, y)

# ------- Support Vector regression -------
model2 = SVR(kernel='rbf', C=1, epsilon=10) # set kernel and hyperparameters
svr = model2.fit(X, y)

# ------- Predict a range of values based on the models for visualization -------
# Create 100 evenly spaced points from smallest X to largest X
x_range = np.linspace(X.min(), X.max(), 100)

# Predict y values for our set of X values
y_lr = model1.predict(x_range.reshape(-1, 1)) # Linear regression
y_svr = model2.predict(x_range.reshape(-1, 1)) # SVR

# =======================
# visualize the two models
# Create a scatter plot
fig = px.scatter(df, x=df['X3 distance to the nearest MRT station'], y=df['Y house price of unit area'],
                 opacity=0.8, color_discrete_sequence=['black'])

# Add a best-fit line
fig.add_traces(go.Scatter(x=x_range, y=y_lr, name='Linear Regression', line=dict(color='limegreen')))
fig.add_traces(go.Scatter(x=x_range, y=y_svr, name='Support Vector Regression', line=dict(color='red')))
fig.add_traces(go.Scatter(x=x_range, y=y_svr+10, name='+epsilon', line=dict(color='red', dash='dot')))
fig.add_traces(go.Scatter(x=x_range, y=y_svr-10, name='-epsilon', line=dict(color='red', dash='dot')))

# Change chart background color
fig.update_layout(dict(plot_bgcolor = 'white'))

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                 showline=True, linewidth=1, linecolor='black')

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                 showline=True, linewidth=1, linecolor='black')

# Set figure title
fig.update_layout(title=dict(text="House Price Based on Distance from the Nearest MRT with Model Predictions (epsilon=10, C=1)",
                             font=dict(color='black')))
# Update marker size
fig.update_traces(marker=dict(size=3))

fig.show()

# ==========
# SV vs. multiple linear regression - 2 independent variables
# Create a 3D scatter plot
fig = px.scatter_3d(df,
                    x=df['X3 distance to the nearest MRT station (scaled)'],
                    y=df['X2 house age (scaled)'],
                    z=df['Y house price of unit area'],
                    opacity=0.8, color_discrete_sequence=['black'],
                    height=900, width=1000
                   )

# Set figure title
fig.update_layout(title_text="Scatter 3D Plot",
                  scene_camera_eye=dict(x=1.5, y=1.5, z=0.25),
                  scene_camera_center=dict(x=0, y=0, z=-0.2),
                  scene = dict(xaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='lightgrey'),
                               yaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='lightgrey'
                                          ),
                               zaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='lightgrey')))

# Update marker size
fig.update_traces(marker=dict(size=2))

fig.show()

# ----------- Select variables -----------
X=df[['X3 distance to the nearest MRT station (scaled)','X2 house age (scaled)']]
y=df['Y house price of unit area'].values

# ----------- Model fitting -----------
# Define models and set hyperparameter values
model1 = LinearRegression()
model2 = SVR(kernel='rbf', C=100, epsilon=1)

# Fit the two models
lr = model1.fit(X, y)
svr = model2.fit(X, y)

# ----------- For creating a prediciton plane to be used in the visualization -----------
# Set Increments between points in a meshgrid
mesh_size = 0.05

# Identify min and max values for input variables
x_min, x_max = X['X3 distance to the nearest MRT station (scaled)'].min(), X['X3 distance to the nearest MRT station (scaled)'].max()
y_min, y_max = X['X2 house age (scaled)'].min(), X['X2 house age (scaled)'].max()

# Return evenly spaced values based on a range between min and max
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)

# Create a meshgrid
xx, yy = np.meshgrid(xrange, yrange)

# ----------- Create a prediciton plane  -----------
# Use models to create a prediciton plane --- Linear Regression
pred_LR = model1.predict(np.c_[xx.ravel(), yy.ravel()])
pred_LR = pred_LR.reshape(xx.shape)

# Use models to create a prediciton plane --- SVR
pred_svr = model2.predict(np.c_[xx.ravel(), yy.ravel()])
pred_svr = pred_svr.reshape(xx.shape)

# Note, .ravel() flattens the array to a 1D array,
# then np.c_ takes elements from flattened xx and yy arrays and puts them together,
# this creates the right shape required for model input

# prediction array that is created by the model output is a 1D array,
# Hence, we need to reshape it to be the same shape as xx or yy to be able to display it on a graph

# Create a 3D scatter plot with predictions
fig = px.scatter_3d(df, x=df['X3 distance to the nearest MRT station (scaled)'], y=df['X2 house age (scaled)'], z=df['Y house price of unit area'],
                    opacity=0.8, color_discrete_sequence=['black'],
                    width=1000, height=900
                   )

# Set figure title and colors
fig.update_layout(title_text="Scatter 3D Plot with Linear Regression Prediction Surface",
                  scene_camera_eye=dict(x=1.5, y=1.5, z=0.25),
                  scene_camera_center=dict(x=0, y=0, z=-0.2),
                  scene = dict(xaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='lightgrey'),
                               yaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='lightgrey'
                                          ),
                               zaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='lightgrey')))
# Update marker size
fig.update_traces(marker=dict(size=2))

# Add prediction plane
fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred_LR, name='LR',
                          colorscale=px.colors.sequential.Sunsetdark, showscale=False))

fig.show()

# Create a 3D scatter plot with predictions
fig = px.scatter_3d(df, x=df['X3 distance to the nearest MRT station (scaled)'], y=df['X2 house age (scaled)'], z=df['Y house price of unit area'],
                    opacity=0.8, color_discrete_sequence=['black'],
                    width=1000, height=900
                   )

# Set figure title and colors
fig.update_layout(title_text="Scatter 3D Plot with SVR Prediction Surface",
                  scene_camera_eye=dict(x=1.5, y=1.5, z=0.25),
                  scene_camera_center=dict(x=0, y=0, z=-0.2),
                  scene = dict(xaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='lightgrey'),
                               yaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='lightgrey'
                                          ),
                               zaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='lightgrey')))
# Update marker size
fig.update_traces(marker=dict(size=2))

# Add prediction plane
fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred_svr, name='SVR',
                          colorscale=px.colors.sequential.Sunsetdark,
                          showscale=False))

fig.show()

xx = 1