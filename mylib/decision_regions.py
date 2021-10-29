import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, learner, resolution=0.1, title="Decision regions", ax=None):
    D = np.concatenate([X, y], axis=1)
    
    if ax is None:
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(1,1,1)
    # Create color maps
    # cmap_light = ListedColormap(['#fb7b30', '#abcd3f', '#37a6ee'])
    # cmap_bold = ListedColormap(['red', 'green', 'blue'])
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    
    # Plot the decision boundary.
    x_min, x_max = D[:,0].min() - .15, D[:,0].max() + .15
    y_min, y_max = D[:,1].min() - .15, D[:,1].max() + .15
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    Z = learner.predict(np.array([xx.ravel(), yy.ravel()]).T)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    ax.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot the points
    ax.scatter(
        x=D[:, 0],
        y=D[:, 1], c=D[:,-1], cmap=cmap_bold,
        edgecolor='k', s=60)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # plt.legend(loc='best')
    plt.title(title)

    # plt.show()
    