import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


def bar_plot_col_counts(df, col_name, ascending=True, dropna=True, title=''):
    '''
    INPUT:
    df - a dataframe holding all the variables of interest
    col_name - the column containing the data to plot
    
    OUTPUT:
    a bar chart with value counts for the column supplied
    '''
    df[col_name].value_counts(ascending=ascending, dropna=dropna).plot(kind='bar', title=('Value counts for {}'.format(col_name)));
    plt.title(title)
    
    
# Copied from helper functions used in udacity lessons
def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA in scikit learn
            
    OUTPUT:
            None
    '''
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=8)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
    
    
# given pca, plot the variance ratios
def plot_pca_variance_ratio(pca):
    '''
    INPUT:
    pca - pca model to plot components and explained variance for
    
    OUTPUT:
    a plot of cumulative explained variance by component
    '''
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.show();
    
    
def plot_feature_importance(df, model, threshold):
    '''
    INPUT:
    df - a dataframe related to the data the model was trained on
    model - a regression model
    threshold - the importance threshold  above which to plot data
    
    OUTPUT:
    a bar chart showing the importance of the features used in the model
    '''
    feats = {} # a dict to hold feature_name: feature_importance
    for feature, importance in zip(df.columns, model.feature_importances_):
        if importance > threshold:
            feats[feature] = importance #add the name/value pair 

    print('Number of features above the threshold of {} is {}'.format(threshold, len(feats)))
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
    importances.sort_values(by='Importance').plot(kind='bar', rot=90);
    
    
def plot_kmean_score_vs_k(x_pca, num_iterations=20):
    '''
    INPUT:
    x_pca - a dataframe pca data
    num_iterations - the number of iterations to fit KMeans clusters over
    
    OUTPUT:
    a chart showing KMeans score vs K
    '''
    # Over a number of different cluster counts...
    # run k-means clustering on the data and...
    # compute the average within-cluster distances.
    kmean_scores = []
    num_clusters = []
    for i in range(num_iterations):
        kmeans_i = KMeans(i+1)
        model_i = kmeans_i.fit(x_pca)
        score_i = np.abs(kmeans_i.score(x_pca))
        kmean_scores.append(score_i)
        num_clusters.append(i+1)

    # Investigate the change in within-cluster distance across number of clusters.
    # HINT: Use matplotlib's plot function to visualize this relationship.
    # now plot the scores
    plt.plot(num_clusters, kmean_scores, linestyle='--', marker='o', color='b');
    plt.xlabel('K');
    plt.ylabel('KMean Score');
    plt.title('KMean Score vs. K');