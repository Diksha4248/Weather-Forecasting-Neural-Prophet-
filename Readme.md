# NeuralProphet
NeuralProphet is a python library for modeling time-series data based on neural networks. It’s built on top of PyTorch and is heavily inspired by [Facebook Prophet](https://github.com/facebook/prophet)
and [AR-Net](https://github.com/ourownstory/AR-Net) libraries.

## NeuralProphet vs. Prophet
According to NeuralProphet’s documentation, the added features are:

* Using PyTorch’s Gradient Descent optimization engine making the modeling process much faster than Prophet
* Using AR-Net for modeling time-series autocorrelation (aka serial correlation)
* Custom losses and metrics
* Having configurable non-linear layers of feed-forward neural networks,
* etc.

### Minimal example
```python
from neuralprophet import NeuralProphet
```
After importing the package, you can use NeuralProphet in your code:
```python
m = NeuralProphet()
metrics = m.fit(df)
forecast = m.predict(df)
```
You can visualize your results with the inbuilt plotting functions:
```python
fig_forecast = m.plot(forecast)
fig_components = m.plot_components(forecast)
fig_model = m.plot_parameters()
```
If you want to forecast into the unknown future, extend the dataframe before predicting:
```python
m = NeuralProphet().fit(df, freq="D")
df_future = m.make_future_dataframe(df, periods=30)
forecast = m.predict(df_future)
fig_forecast = m.plot(forecast)
```
