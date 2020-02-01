import tensorflow as tf

def create_window_dataset(series, window_size, batch_size=32,
                   shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    # "This dataset fills a buffer with buffer_size elements, then randomly samples elements from this buffer"
    dataset = dataset.shuffle(shuffle_buffer)
    # Split into features (window[:-1]) and labels (window[-1])
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    # Prepare later batch while training current batch. This helps to improve performance at the cost
    # of using more memory
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def create_seq2seq_window_dataset(series, window_size, batch_size=32,
                           shuffle_buffer=1000):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    # Split the dataset so that every input features having corresponding labels (the value of the next day)
    # For example: X:[[[1],[2],[3]]], Y:[[[2],[3],[4]]]
    ds = ds.map(lambda w: (w[:-1], w[1:])) 
    return ds.batch(batch_size).prefetch(1)