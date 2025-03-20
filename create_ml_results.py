#-------------------------------------------------------------------------------
#
# Code for the paper:
# "AI brings a fresh approach to relative valuation"
# by Geertsema, Lu, Stouthuysen (2025)
#
#-------------------------------------------------------------------------------

#    Copyright (C) Paul Geertsema 2022 - 2025
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


# ==============================================================================
# ==============================================================================
#
# This first section of code is a verbatim copy of the code available at
# https://github.com/pgeertsema/AXIL_paper
# in particular, the contents of https://github.com/pgeertsema/AXIL_paper/blob/main/axil.py 
# based on the following paper: "Instance-based Explanations for Gradient Boosting Machine Predictions with AXIL Weights"
# by Geertsema and Lu (2023), see https://arxiv.org/abs/2301.01864 
# it is included here minimise dependencies and complexity for what is 
# an application aimed at exposition.
#
# ==============================================================================
# ==============================================================================

# module docstring 
"""AXIL - Additive eXplanations using Instance Loadings"""

#---------------------------------------------------------------
# imports
#---------------------------------------------------------------

import numpy as np
import lightgbm as lgb
import os


#---------------------------------------------------------------
# Leaf Coincidence Matrix (LCM) utility function
#---------------------------------------------------------------


def LCM(vector1, vector2):
    '''
    utility function to create leaf coincidence matrix L from leaf membership vectors vector1 (train) and vector2 (test)
    element l_s,f in L takes value of 1 IFF observations v1 and v2 are allocated to the same leaf (for v1 in vector1 and v2 in vector2)
    that is, vector1[v1] == vector2[v2], otherwise 0

    Input arguments vector1 and vector2 are list-like, that is, support indexing and len()
    Output L is a python matrix of dimension (len(vector1), len(vector2))
    '''
    # relies on numpy broadcasting to generate self equality matrix
    vector1 = np.array(vector1)[:, None]
    vector2 = np.array(vector2)
    return vector1 == vector2

# LCM([201,893,492,131,478,653,152],[58,131,307,653,492])*1



#---------------------------------------------------------------
# functionality starts here
#---------------------------------------------------------------

class Explainer:

    # internal variables
    model = None
    learning_rate = None
    train_data = None
    P_list = None
    lm_train = []
    trained = False

    # constructor
    def __init__(self, model, learning_rate=None):

        if not isinstance(model, lgb.basic.Booster):
            print("Sorry, the model needs to be a LightGBM regression model (type 'lightgbm.basic.Booster').")
            return None
       
        self.model = model
        if learning_rate == None:

            # get from text file of model (a hack, but currently the only way)
            self.model.save_model("temp_booster.txt")
            found = False
            with open("temp_booster.txt", "r") as f:
                for line in f:
                    if "[learning_rate: " in line:
                        lr = line.replace("[learning_rate: ", "").replace("]", "")
                        try:
                            self.learning_rate = float(lr)
                        except:
                            # catch error if unable to convert lr to float
                            print("Sorry, unable to figure out the learning rate from the model provided. Please supply the 'learning_rate' parameter.")
                            return None
                        else:
                            found = True
                # delete temporary model file
                f.close()
                os.remove("temp_booster.txt")

            if not found:
                print("Sorry, unable to figure out the learning rate from the model provided. Please supply the 'learning_rate' parameter.")
                return None

        else:

            # or, if supplied by the user
            self.learning_rate = learning_rate           

        return


    # fit the model to training data
    def fit(self, X):

        if isinstance(X, lgb.basic.Dataset):
            print("Sorry, you need to provide the raw data (type lightgbm.basic.Dataset), just like for LightGBM predict()")
            return None

        # number of observations in data
        N = len(X)

        self.train_data = X

        # obtain instance leaf membership information from trained LightGBM model (argument pred_leaf=True)
        instance_leaf_membership = self.model.predict(data=X, pred_leaf=True)

        # the first "tree" mimics a single leaf, so that it effectively calculates the training data sample average
        lm = np.concatenate((np.ones((1, N)), instance_leaf_membership.T), axis = 0) + 1

        # number of trees in model
        num_trees = self.model.num_trees()

        # useful matrices
        ones = np.ones((N,N))
        I = np.identity(N)

        # Clear list of P matrices (to be used for calculating AXIL weights)
        P_list = []

        # iterations 0 model predictions (simply average of training data)
        # corresponds to "tree 0"
        P_0 = (1/N) * ones
        P_list.append(P_0)
        G_prev = P_0

        # do iterations for trees 1 to num_trees (inclusive)
        # note, LGB trees ingnores the first (training data mean) predictor, so offset by 1
        for i in range(1, num_trees+1):

            D = LCM(lm[i], lm[i])
            P = self.learning_rate * ( (D / (ones @ D)) @ (I-G_prev) )
            P_list.append(P)
            G_prev +=  P


        self.trained = True
        self.P_list = P_list
        self.lm_train = lm
        return

    def transform(self, X):

        if not self.trained:
            print("Sorry, you first need to fit to training data. Use the fit() method.")
            return None

        # list of P matices
        P = self.P_list

        # number of instances in training data used to estimate P
        N, _ = P[0].shape

        # number of instances in this data
        S = len(X)

        # model instance membership of tree leaves 
        instance_leaf_membership = self.model.predict(data=X, pred_leaf=True)

        lm_test = np.concatenate((np.ones((1, S)), instance_leaf_membership.T), axis = 0) + 1

        # number of trees in model
        num_trees = self.model.num_trees()

        # ones matrix with same dimensions as P
        ones_P = np.ones((N, N))
        
        # ones matrix with same dimensions as L
        ones_L = np.ones((N, S))

        # first tree is just sample average
        L = ones_L
        K = (P[0].T @ (L / (ones_P @ L)))

        # execute for 1 to num_trees (inclusive)
        for i in range(1, num_trees+1):
            L = LCM(self.lm_train[i], lm_test[i])
            K += (P[i].T @ (L / (ones_P @ L)))


        return K
    
    def reset(self):
        # reset to initialised state
        self.train_data = None
        self.P_list = None
        self.lm_train = []
        self.trained = False



# ==============================================================================
# ==============================================================================
#
# This section contains the actual example code that generates the images and 
# data reported in # "AI brings the finance world a fresh approach to relative valuation"
# by Geertsema, Lu, Stouthuysen (2025)
#
# ==============================================================================
# ==============================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import shap
import matplotlib.cm as cm
import scipy.cluster.hierarchy as hc

# change as needed
SOURCE = r"C:\Data\HBR\Work"
RESULTS = r"C:\Users\Paul\Dropbox\HBR_peers\results"
os.chdir(RESULTS)

#---------------------------------------------------------------
# Data
#---------------------------------------------------------------

TARGET = "target"
data = pd.read_stata(SOURCE+"\\combined.dta", convert_dates=False)

exclude_cols = [TARGET, "permno", "ticker", "target", "marketcap"] 

#---------------------------------------------------------------
# Model parameters
#---------------------------------------------------------------

# given the very small size of the dataset, we opt for a 
# very restrained GBM with only 5 trees
# We set min_data to 5 so that each leaf prediction is based
# on a minimum of 5 observations 

# results are in-sample, as we are not attempting prediction per se
# but rather, we are interested in recovering the peer-relationships
# between firms

TREES = 5
LR = 0.1

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "verbose": 0,  
    "min_data": 5,
    "learning_rate": LR,    
}

# Define the example firm we want to focus on
# "MA" is the ticker for Mastercard
example_firm = "MA" 

#---------------------------------------------------------------
# Process by month
#---------------------------------------------------------------

# get all unique months and companies
months = sorted(data["mth"].unique())
total_months = len(months)
all_companies = sorted(data["ticker"].unique())
num_companies = len(all_companies)

# create mapping from ticker to index for consistent ordering
ticker_to_idx = {ticker: idx for idx, ticker in enumerate(all_companies)}

# storage for monthly results
all_shap_values = []              # average SHAP values across all observations
all_axil_loadings = []            # AXIL loadings matrix for each month
example_firm_shap_values = []     # SHAP values for example_firm only
example_firm_axil_loadings = []   # AXIL loadings for example_firm only

# storage for actual and predicted target values for all firms
all_actual = []
all_predicted = []

# storage for monthly data to calculate feature averages efficiently
monthly_data_frames = []

# we train a new model of each of the 36 months
# extract SHAP values and AXIL weights (peer-weights) for each month
# which we will then average before presenting results
# by averaging across 36 months we ensure a more stable and representative
# representation of relationships

# loop by month
for i, month in enumerate(months):
    print(f"Processing month {i+1} of {total_months} ({month})")
    
    # dilter data for this month
    month_data = data[data["mth"] == month]
    
    # sort by ticker to ensure consistent company ordering
    month_data = month_data.sort_values("ticker").reset_index(drop=True)
    
    # store the month's data for later processing
    monthly_data_frames.append(month_data.copy())
    
    X_train = month_data.drop(columns=exclude_cols + ["mth"])
    y_train = month_data[TARGET]
    month_tickers = month_data["ticker"].values
    
    # train model for this month
    np.random.seed(42)  
    lgb_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, lgb_data, num_boost_round=TREES)
    
    # calculate SHAP values
    SHAP_explainer = shap.Explainer(model)
    shap_values = SHAP_explainer(X_train)
    
    # get predicted values for all firms in this month
    predictions = model.predict(X_train)
    
    # store actual and predicted values for all firms in this month
    for i in range(len(y_train)):
        all_actual.append(y_train.iloc[i])
        all_predicted.append(predictions[i])
        
    # find index of example firm in this month's data for SHAP and AXIL analysis
    example_firm_idx = month_data[month_data["ticker"] == example_firm].index[0]
    
    # calculate absolute SHAP values
    # this way we focus on magniture of impact, rather than direction
    abs_shap = np.abs(shap_values.values)
    
    # scale SHAP values to add to 100% for each observation
    # this makes shap values comparable accross different months
    scaled_shap = np.zeros_like(abs_shap)
    for i in range(abs_shap.shape[0]):
        row_sum = np.sum(abs_shap[i, :])
        if row_sum > 0:  
            scaled_shap[i, :] = (abs_shap[i, :] / row_sum) * 100
    
    # store average percentage SHAP values for this month
    month_avg_shap = np.mean(scaled_shap, axis=0)
    all_shap_values.append(month_avg_shap)
    
    # store SHAP values specifically for example_firm
    example_firm_shap = scaled_shap[example_firm_idx]
    example_firm_shap_values.append(example_firm_shap)
    
    # calculate AXIL loadings (this relies on the Explainer class 
    # that is part of the AXIL.py file we copied above)
    AXIL_explainer = Explainer(model)
    AXIL_explainer.fit(X_train)
    loadings = AXIL_explainer.transform(X_train)
    
    # store the loadings - companies should be in same order as sorted by ticker
    all_axil_loadings.append(loadings)
    
    # get index for example_firm in the sorted tickers
    example_firm_col_idx = list(month_tickers).index(example_firm)
    
    # store AXIL loadings specifically for example_firm (column of loadings matrix)
    example_firm_loadings = loadings[:, example_firm_col_idx]
    example_firm_axil_loadings.append(example_firm_loadings)

# calculate average feature values for each company across all months
print("Calculating average feature values across all months...")
all_months_data = pd.concat(monthly_data_frames)
feature_cols = [col for col in all_months_data.columns if col not in exclude_cols and col != "mth"]
avg_company_feature_values = all_months_data.groupby("ticker")[feature_cols].mean()

#---------------------------------------------------------------
# Create scatter plot of actual vs. predicted for all firms
#---------------------------------------------------------------

# convert to numpy arrays
all_actual = np.array(all_actual)
all_predicted = np.array(all_predicted)

# create scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(all_actual, all_predicted, alpha=0.5, s=10)  # Smaller point size for many points

# add line of best fit
z = np.polyfit(all_actual, all_predicted, 1)
p = np.poly1d(z)

# create x range for plotting the line of best fit
x_range = np.linspace(min(all_actual), max(all_actual), 100)
plt.plot(x_range, p(x_range), "r--", alpha=0.7, linewidth=2)

# add labels
plt.xlabel('Actual Target', fontsize = 18)
plt.ylabel('Predicted Target', fontsize = 18)

plt.grid(True, alpha=0.3)
plt.tight_layout()

# save the plot
plt.savefig("scatter_plot.pdf", bbox_inches="tight")
plt.close()
print(f"Created scatter_plot.pdf")

#---------------------------------------------------------------
# Average results across months
#---------------------------------------------------------------

# average SHAP values across months
avg_shap_values = np.mean(np.array(all_shap_values), axis=0)
feature_names = X_train.columns  # Features should be the same across all months

# average AXIL loadings across months
avg_axil_loadings = np.mean(np.array(all_axil_loadings), axis=0)

# average SHAP values specifically for example_firm across months
avg_example_firm_shap = np.mean(np.array(example_firm_shap_values), axis=0)

# average AXIL loadings specifically for example_firm across months
avg_example_firm_axil = np.mean(np.array(example_firm_axil_loadings), axis=0)

#---------------------------------------------------------------
# Output CSV files for example_firm
#---------------------------------------------------------------

# create and save sorted SHAP values for example_firm
example_firm_shap_df = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Value': avg_example_firm_shap
})
example_firm_shap_df = example_firm_shap_df.sort_values('SHAP_Value', ascending=False)
example_firm_shap_df.to_csv(f"{example_firm}_SHAP_values.csv", index=False)

# create and save sorted AXIL weights for example_firm
example_firm_axil_df = pd.DataFrame({
    'Ticker': all_companies,
    'AXIL_Weight': avg_example_firm_axil
})
example_firm_axil_df = example_firm_axil_df.sort_values('AXIL_Weight', ascending=False)
example_firm_axil_df.to_csv(f"{example_firm}_AXIL_weights.csv", index=False)

#---------------------------------------------------------------
# Generate outputs using averages
#---------------------------------------------------------------

# SHAP bar plot
feature_importance = [(feature, importance) for feature, importance in zip(feature_names, avg_shap_values)]
feature_importance.sort(key=lambda x: x[1], reverse=True)

plt.figure(figsize=(10, 6))
features = [x[0] for x in feature_importance]
contributions = [x[1] for x in feature_importance]
y_pos = np.arange(len(features))

plt.barh(y_pos, contributions)
plt.yticks(y_pos, features)
plt.xlabel('Average SHAP Contribution (% of total impact)')
plt.title('Feature Importance (Scaled to 100%)')
plt.tight_layout()
plt.savefig("SHAP_magnitudes_"+TARGET+".pdf", bbox_inches="tight")
plt.close()

# create a DataFrame for the averaged AXIL loadings
avg_axil_df = pd.DataFrame(
    avg_axil_loadings, 
    index=all_companies, 
    columns=all_companies
)

# export AXIL weights
avg_axil_df.to_csv("AXIL_weights.csv")

# AXIL weights heatmap
plt.close()
ax = sns.heatmap(avg_axil_df, xticklabels=1, yticklabels=1, cmap="Blues", cbar=False)
ax.set_box_aspect(1)
ax.yaxis.tick_right()
ax.set(ylabel=None)
ax.tick_params(axis='both', which='major', labelsize=8)
plt.yticks(rotation=0)
ax.figure.tight_layout()
plt.savefig("heatmap_all_"+TARGET+".pdf")
plt.close()

# AXIL weights clustered heatmap
plt.close()
linkage = hc.linkage(avg_axil_df, method='average')
ax = sns.clustermap(avg_axil_df, row_linkage=linkage, col_linkage=linkage, xticklabels=1, yticklabels=1, cmap="Blues", cbar_pos=None)
ax.figure.tight_layout()
ax.ax_heatmap.set_ylabel("")
plt.savefig("clustermap_all_"+TARGET+".pdf")
plt.close()

#---------------------------------------------------------------
# Generate network graph
#---------------------------------------------------------------

# AXIL weights spring network
labels = {i: ticker for i, ticker in enumerate(all_companies)}

graph = avg_axil_loadings.copy()
np.fill_diagonal(graph, 0)
G = nx.from_numpy_array(graph)

# normalize columns so incoming edges to each node sum to 1 (for comparability)
normalized_graph = graph.copy()
for col in range(normalized_graph.shape[1]):
    col_sum = normalized_graph[:, col].sum()
    if col_sum > 0:
        normalized_graph[:, col] = normalized_graph[:, col] / col_sum

# create a graph with the normalized edges
G_normalized = nx.from_numpy_array(normalized_graph)

# get all edges and weights from normalized graph
edges_to_draw = list(G_normalized.edges())

# create a list of (edge, weight) tuples
edge_weight_pairs = [(edge, G_normalized[edge[0]][edge[1]]['weight']) for edge in edges_to_draw]

# sort by weight (ascending) so stronger edges are drawn on top
edge_weight_pairs.sort(key=lambda x: x[1])

# extract sorted edges and weights
edges_to_draw = [pair[0] for pair in edge_weight_pairs]
weights = [pair[1] for pair in edge_weight_pairs]

min_weight = min(weights) if weights else 0
max_weight = max(weights) if weights else 0

# scale weights
scaled_weights = [0.8 * (w - min_weight) / (max_weight - min_weight) for w in weights]

# Find the node index for example_firm
example_firm_idx = None
for idx, ticker in labels.items():
    if ticker == example_firm:
        example_firm_idx = idx
        break

# prepare node colors
node_colors = ['lightskyblue'] * len(G.nodes())
if example_firm_idx is not None:
    node_colors[example_firm_idx] = 'indianred'

# prepare edge colors - red for example_firm edges, grey for others
edge_colors = []
for i, edge in enumerate(edges_to_draw):
    if example_firm_idx is not None and (edge[0] == example_firm_idx or edge[1] == example_firm_idx):
        # use Reds colormap for example_firm edges
        edge_colors.append(cm.Reds(scaled_weights[i]))
    else:
        # use Greys colormap for other edges
        edge_colors.append(cm.Greys(scaled_weights[i]))

# scale edge width with weight
edge_widths = [8 * w for w in scaled_weights]

# create "spring" layout
# this start with a fixed repulsive force between all nodes
# then adds an attractive force relative to the strength of the peer-weight association
pos = nx.spring_layout(G_normalized, iterations=10_000, seed=42, k=0.02)

# draw the network
plt.figure(figsize=(12, 12))
nx.draw(G, pos, with_labels=True, labels=labels, font_weight='medium',
        node_color=node_colors,  
        edgelist=edges_to_draw,  
        edge_color=edge_colors,
        width=edge_widths,
        font_size=18) 

plt.savefig("spring_" + TARGET + ".pdf")
plt.close()

#---------------------------------------------------------------
# Create table of top peers with their top feature values
#---------------------------------------------------------------

print("\nCreating table of top 5 peers and their feature values...")

# identify top 5 peers of example_firm (excluding the firm itself)
top_peers = example_firm_axil_df['Ticker'].values
top_peers = [peer for peer in top_peers if peer != example_firm][:5]
print(f"Top 5 peers of {example_firm}: {', '.join(top_peers)}")

# identify top 5 features for example_firm based on SHAP values
top_features = example_firm_shap_df['Feature'].values[:5]
print(f"Top 5 features for {example_firm}: {', '.join(top_features)}")

# create DataFrame for the peer feature table
peer_features_df = pd.DataFrame(index=[example_firm] + top_peers, columns=top_features)

# populate with feature values (averaged across all months)
for ticker in [example_firm] + top_peers:
    for feature in top_features:
        try:
            peer_features_df.loc[ticker, feature] = avg_company_feature_values.loc[ticker, feature]
        except (KeyError, ValueError):
            peer_features_df.loc[ticker, feature] = np.nan

# format the table for better readability
formatted_table = peer_features_df.copy()
for feature in top_features:
    # Determine appropriate formatting based on feature values
    feature_values = peer_features_df[feature].dropna().values
    if len(feature_values) > 0:
        if np.max(np.abs(feature_values)) < 10:
            formatted_table[feature] = peer_features_df[feature].map(lambda x: '{:.4f}'.format(x) if pd.notna(x) else 'N/A')
        else:
            formatted_table[feature] = peer_features_df[feature].map(lambda x: '{:.2f}'.format(x) if pd.notna(x) else 'N/A')

# save the table to CSV
formatted_table.to_csv(f"{example_firm}_top5_peers_features.csv")

# print the table
print(f"\nTable of top features for {example_firm} and its top 5 peers:")
print(formatted_table)

print(f"\nCSV outputs created:")
print(f"  {example_firm}_SHAP_values.csv - Sorted SHAP values for {example_firm}")
print(f"  {example_firm}_AXIL_weights.csv - Sorted AXIL weights for {example_firm}")
print(f"  {example_firm}_top5_peers_features.csv - Top features for {example_firm} and its top 5 peers")
print(f"  scatter_plot.pdf - Scatter plot of actual vs. predicted for all firms")

#---------------------------------------------------------------
# Create CSV of feature ranks for top peers
#---------------------------------------------------------------

print("\nCreating CSV file with feature ranks for top peers...")

# calculate ranks for all companies across all features
# first create a dataframe with just the features we care about for all companies
all_company_features = all_months_data.groupby("ticker")[top_features].mean()

# calculate ranks for each feature (lower rank = higher value)
feature_ranks = all_company_features.rank(ascending=False, method='min')

# total number of companies (for reference)
total_companies = len(all_company_features)
print(f"Ranking out of {total_companies} total companies")

# extract ranks just for our companies of interest
top_companies_ranks = feature_ranks.loc[[example_firm] + top_peers]

# add a column with the AXIL weight for easy reference
top_companies_ranks['AXIL_Weight'] = 0.0
for company in [example_firm] + top_peers:
    weight = example_firm_axil_df.loc[example_firm_axil_df['Ticker'] == company, 'AXIL_Weight'].values[0]
    top_companies_ranks.loc[company, 'AXIL_Weight'] = weight

# make the company name the first column instead of the index
top_companies_ranks.reset_index(inplace=True)
top_companies_ranks.rename(columns={'index': 'Ticker'}, inplace=True)

# add percentile information (as a percentage) for easier interpretation
for feature in top_features:
    top_companies_ranks[f"{feature}_percentile"] = 100 * (1 - (top_companies_ranks[feature] - 1) / (total_companies - 1))
    # Round to 1 decimal place
    top_companies_ranks[f"{feature}_percentile"] = top_companies_ranks[f"{feature}_percentile"].round(1)

# save to CSV
top_companies_ranks.to_csv(f"{example_firm}_top5_peer_feature_ranks.csv", index=False)
print(f"  {example_firm}_top5_peer_feature_ranks.csv - Feature ranks for {example_firm} and top 5 peers")

# code complete!