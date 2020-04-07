import plotly.graph_objects as go

pos_train = 9000
pos_test=1800
neg_test=150000
total = pos_test+pos_train+neg_test
fig = go.Figure(data=go.Scatter(
    x=[1, 2, 3],
    y=[10, 10, 10],
    mode='markers',
    marker=dict(size=[1000*9000/total, 1000*1800/total, 1000*50000/total],
                color=[0, 1, 2])
))

fig.show()
