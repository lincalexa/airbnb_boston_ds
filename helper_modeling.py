import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error



# Copied from helper functions used in udacity lessons
# Investigate the variance accounted for by each principal component.
# modified from helper functions from lessons
def show_variance_by_dimension(pca, df):
    '''
    INPUT:
    pca - pca model
    df - a DataFrame used to create the pca
    
    OUTPUT:
    df - a DataFrame containing PCA dimensions and explained variance
    '''    
    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = df.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    # Return a concatenated DataFrame
    pca_result = pd.concat([variance_ratios, components], axis = 1)
    print('Sum of result Explained Variance: {}'.format(pca_result['Explained Variance'].sum()))

    return pca_result

# given a number of components, either fit and transform, or just transform the data with PCA
def do_pca(df, num_components, only_transform=False):
    '''
    INPUT:
    df - a DataFrame with data to be modeled
    num_components - the number of components to use
    only_transform - True/False indicator for transform or transform_fit
    
    OUTPUT:
    pca - a pca model
    x_pca - data transformed (and optionally fitted) to pca model
    '''    
    pca = PCA(num_components, random_state=42)
    if only_transform == True:
        x_pca = pca.transform(df)
        print('pca transformed')
    else:
        x_pca = pca.fit_transform(df)
        print('pca fitted and transformed')
    return pca, x_pca


# need to combine column names with weights and plot
def sort_pca_by_weights(comp_num, df_scaled, pca_n, print_size=10, print_all=False):
    '''
    INPUT:
    comp_num - component number
    df_scaled - a scaled DataFrame
    pca_n - pca model
    print_size - the number of features to print
    print_all - optionally print all features
    
    OUTPUT:
    print of PCA Weights 
    '''    
    pca_com = pd.DataFrame(pca_n.components_[comp_num], columns=['pca_weights'])
    feature_names = pd.DataFrame(df_scaled.columns, columns=['feature_name'])
    pca_weights = pd.concat([pca_com, feature_names], axis=1)
    pca_sorted = pca_weights.sort_values(by=['pca_weights'])
    print('PCA By Weights Head:')
    print(pca_sorted.head(print_size))
    print('\nPCA By Weights Tail:')
    print(pca_sorted.tail(print_size))
    if print_all == True:
        print('\nPCA By Weights:')
        print(pca_sorted)

        
# fit and predict kmeans for PCA
def fit_kmeans_predict(num_clusters, x_pca, model_only=False):
    '''
    INPUT:
    num_clusters - the number of clusters for PCA
    x_pca - data fitted to the PCA
    model_only - True/False indicator to model only or model and predict
    
    OUTPUT:
    kmeans_model - the kmeans model for the pca
    kmeans_pred - if model_only=False, the kmeans predictions
    '''    

    kmeans = KMeans(num_clusters)
    kmeans_model = kmeans.fit(x_pca)
    if model_only == False:
        kmeans_pred = kmeans_model.predict(x_pca)
        print('kmeans fitted and predicted')
        return kmeans_model, kmeans_pred
    else:
        print('kmeans fitted')
        return kmeans_model

    
# Based on this article: 
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
def rf_tune_parameters(model):
    '''
    INPUT:
    model - a dataframe with listing data to be cleaned
    
    OUTPUT:
    rf_tuned - RandomizedSearchCV model parameter tuning results
    '''    

    n_estimators = [5, 10, 20, 50, 100, 200, 500]
    max_features = ['auto', 'sqrt']
    max_depth = [5,10,15,20,25,30,35,40,45,50, None]
    min_samples_split = [3, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf_tuned = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    return rf_tuned
