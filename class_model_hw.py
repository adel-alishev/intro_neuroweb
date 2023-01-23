import tensorflow as tf
import numpy as np
print(tf.__version__)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from sklearn.metrics import accuracy_score

np.random.seed(10)

colors = ['red', "blue"]
labels_cmap = ListedColormap(colors, 2)
colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]  # R -> W -> B
main_cmap = LinearSegmentedColormap.from_list("main_scheme", colors, N=300)


def show_data(X, y):
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], s=120, color=labels_cmap(y))
    plt.show()
def show_descision_boundary(clf, limits, binary=False, X=None, y=None, n_lines=10, show_lines=False,
                            figsize=(5, 5), ax=None):
    if limits is None:
        if X is not None:
            xs = [X[:, 0].min() - .3, X[:, 0].max() + .3]
            ys = [X[:, 1].min() - .3, X[:, 1].max() + .3]
        else:
            xs = [-1, 1]
            ys = [-1, 1]
    else:
        xs, ys = limits

    x_min, x_max = xs
    y_min, y_max = ys

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

    if binary:
        Z = clf.predict(np.c_[xx.ravel(), xx.ravel()])
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        norm = Normalize(vmin=0., vmax=1.)
    else:
        Z = clf(np.c_[xx.ravel(), xx.ravel()])
        Z = clf(np.c_[xx.ravel(), yy.ravel()])
        # if clf.prob_output:
        #    norm = Normalize(vmin=0.,vmax=1.)
        # else:
        norm = Normalize(vmin=-10., vmax=10., clip=True)
        Z = Z.numpy()

    Z = Z.reshape(xx.shape)
    Z = Z.astype(np.float32)

    ax.contourf(xx, yy, Z, n_lines, alpha=0.4, cmap=main_cmap, norm=norm)
    if show_lines:
        cp = ax.contour(xx, yy, Z, n_lines)
        ax.clabel(cp, inline=True,
                  fontsize=10, colors="green")

    if y is not None:
        X = np.array(X)
        y = np.array(y)
        ax.scatter(X[:, 0], X[:, 1], s=120, color=labels_cmap(y),
                   zorder=4)
    plt.show()
def eval_model(model, X, y):
    accuracy = model.evaluate(X, y)[1]
    if accuracy == 1.0:
        print("Perfect!")
    elif  accuracy > 0.9:
        print("Well done! Can you make 100%?")
    else:
        print("Don't give up!")
    return accuracy
# X = np.random.rand(200, 2) - 0.5
# y = ((X[:, 0] < 0) ^ (X[:, 1] < 0))
# X = X*2
# show_data(X, y)
# from sklearn.datasets import make_moons
#
# X, y = make_moons(noise=0.04)
# show_data(X, y)
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

X, y = make_circles(factor=0.5, noise=0.05)
show_data(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}, Labels ratio: {y_train.mean():.2f}")

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8, input_shape=(2,), activation='relu'))
model.add(tf.keras.layers.Dense(1))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
from pathlib import Path
path = Path("model_1")
path.mkdir(exist_ok=True) # создаем папку на диске
# cpt_filename = "{epoch:02d}_checkpoint_{val_loss:.2f}"
cpt_filename = "checkpoint.hdf5"
cpt_path = str(path / cpt_filename)

checkpoint = tf.keras.callbacks.ModelCheckpoint(cpt_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=100, verbose=1, batch_size=64,
          callbacks=[checkpoint])
print(history)

limits = [[-0.6, 0.6], [-0.6, 0.6]]
eval_model(model, X, y)
show_descision_boundary(limits=None, clf=model, binary=False,
                                X=X,
                                y=y,
                                n_lines=50,
                                show_lines=False)
history_df = pd.DataFrame(history.history)
plt.plot(history_df.loss)
plt.show()