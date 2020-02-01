import tensorflow as tf

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series) # Create a Dataset from series
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast