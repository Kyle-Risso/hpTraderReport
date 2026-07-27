#+----------------------------------------------------------------------------+
#|                                     high_dimensionality_analysis_models.py |
#|          Copyright 2022-2025 HP Investment Trading and Gambling Strategies |
#|                                                        https://hp-fx-g.com |
#+----------------------------------------------------------------------------+

##--- import modules
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from adjustText import adjust_text
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.cluster import KMeans
from scipy.signal import argrelextrema
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D

#+----------------------------------------------------------------------------+
#| @func: get parent directory                                                |
#| @desc: finds and returns the parent directory of the current script        |
#| @params: N/A                                                               |
#| @return: parent dir[ectory] --> parent directory of the current script     |
#+----------------------------------------------------------------------------+
def get_parent_directory():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    return parent_dir


class High_Dimensional_Analysis_Modeler():
    ##--- create the initialization method
    def __init__(self, cfd_position_records):
        self.cfd_position_records = cfd_position_records

    #+----------------------------------------------------------------------------+
    #| @func: Cluster Profitability By Symbol                                     |
    #| @desc: creates and displays a dendrogram with clustered profit by symbol   |
    #| @params: account records --> dataframe with the cleaned account records    |
    #|             num_clusters --> height to cut the tree/dendrogram at          |
    #| @return: plotted dendrogram                                                |
    #+----------------------------------------------------------------------------+
    def cluster_profitability_by_symbol(self):
        ##--- convert class variable to local variable
        cfd_position_records = self.cfd_position_records
        
        #--- encode symbols
        le = LabelEncoder()
        cfd_position_records['Symbol Encoded'] = le.fit_transform(cfd_position_records['Symbol'])

        #--- standardize gross profit
        scaler = StandardScaler()
        cfd_position_records['Gross Profit Standardized'] = scaler.fit_transform(cfd_position_records[['Gross Profit']])

        #--- hierarchical clustering
        data_for_clustering = cfd_position_records[['Gross Profit Standardized', 'Symbol Encoded']]
        Z = linkage(data_for_clustering, method='ward', metric='euclidean')

        #--- cut clusters
        num_clusters = np.floor(np.sqrt(cfd_position_records['Symbol'].nunique()))
        clusters = fcluster(Z, t=num_clusters, criterion='maxclust')
        cfd_position_records['Cluster'] = clusters

        #--- compute leaf colors by lifetime profitability per symbol
        symbol_profit = cfd_position_records.groupby('Symbol')['Gross Profit'].sum()
        symbol_colors = {}
        for symbol, profit in symbol_profit.items():
            if profit > 0:
                symbol_colors[symbol] = 'tab:blue'
            elif profit < 0:
                symbol_colors[symbol] = 'tab:orange'
            else:
                symbol_colors[symbol] = 'white'

        ##--- plot dendrogram
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 8))

        dendro = dendrogram(
            Z,
            labels=cfd_position_records['Symbol'].values,
            leaf_rotation=90,
            leaf_font_size=6,
            color_threshold=4,
            above_threshold_color='cyan',
            ax=ax
        )

        #--- color leaves by profitability
        xlbls = ax.get_xmajorticklabels()
        for lbl in xlbls:
            symbol = lbl.get_text()
            lbl.set_color(symbol_colors.get(symbol, 'grey'))
            lbl.set_fontweight('bold')

        #--- annotate cluster summaries at top-left corner
        cluster_summary = cfd_position_records.groupby('Cluster')['Gross Profit'].agg(['mean', 'count']).reset_index()

        # compute top y-position and spacing
        y_top = Z[:, 2].max() * 0.999  # slightly below top of dendrogram
        y_step = Z[:, 2].max() * 0.025  # closer spacing between labels

        for i, row in enumerate(cluster_summary.itertuples()):
            mean_profit = row.mean
            # clip tiny near-zero values to 0
            if abs(mean_profit) < 0.005:
                mean_profit = 0.0
            # format string with negative sign before dollar if needed
            if mean_profit < 0:
                profit_str = f"-${abs(mean_profit):.2f} USD"
            else:
                profit_str = f"${mean_profit:.2f} USD"
            
            ax.text(
                x=10,
                y=y_top - i * y_step,
                s=f"Cluster {int(row.Cluster)}: {int(row.count)} trades, Avg Profit {profit_str}",
                color='white',
                fontsize=6,
                ha='left',
                va='top'
            )

        plt.title('Profitability Clusters by Symbol', fontsize=20)
        plt.xlabel('Symbol', fontsize=14)
        plt.ylabel('Distance', fontsize=14)
        plt.tight_layout()
        plt.savefig(
            r'documentation/high_dimensionality_analysis/profitability_clusters_by_symbol.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()

    #+----------------------------------------------------------------------------+
    #| @func: KMeans Profitability By Symbol                                      |
    #| @desc: creates and displays a dendrogram with clustered profit by symbol   |
    #| @params: account records --> dataframe with the cleaned account records    |
    #|             num_clusters --> height to cut the tree/dendrogram at          |
    #| @return: plotted dendrogram                                                |
    #+----------------------------------------------------------------------------+
    def kmeans_profitability_by_symbol(self):
        ##--- convert class variable to local variable
        cfd_position_records = self.cfd_position_records
        
        #--- encode symbols
        le = LabelEncoder()
        cfd_position_records['Symbol Encoded'] = le.fit_transform(cfd_position_records['Symbol'])

        #--- standardize gross profit
        scaler = StandardScaler()
        cfd_position_records['Gross Profit Standardized'] = scaler.fit_transform(cfd_position_records[['Gross Profit']])

        #--- determine number of clusters
        num_clusters = int(np.sqrt(cfd_position_records['Symbol'].nunique()))

        #--- apply KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cfd_position_records['Cluster'] = kmeans.fit_predict(cfd_position_records[['Gross Profit Standardized', 'Symbol Encoded']])

        #--- define a color palette for clusters
        cluster_palette = sns.color_palette('tab10', n_colors=num_clusters)  # can handle up to 10 colors; will repeat if more
        cluster_colors = {i: cluster_palette[i % len(cluster_palette)] for i in range(num_clusters)}

        #--- plot clusters
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 8))

        for cluster in range(num_clusters):
            cluster_data = cfd_position_records[cfd_position_records['Cluster'] == cluster]
            ax.scatter(
                cluster_data['Symbol Encoded'],
                cluster_data['Gross Profit Standardized'],
                s=80,
                alpha=0.7,
                label=f'Cluster {cluster + 1}',
                color=cluster_colors[cluster]
            )

        #--- annotate symbols
        for i, row in cfd_position_records.iterrows():
            ax.text(
                row['Symbol Encoded'],
                row['Gross Profit Standardized'],
                row['Symbol'],
                fontsize=6,
                color='white',  # white keeps labels readable
                fontweight='bold',
                rotation=45,
                ha='right',
                va='bottom'
            )

        ax.set_xlabel('Symbol', fontsize=14)
        ax.set_ylabel('Profit (Standardized)', fontsize=14)
        ax.set_title('KMeans Clustering of Profitability by Symbol', fontsize=20)
        ax.legend()
        ax.grid(True, linestyle=':', color='grey', alpha=0.5)

        plt.tight_layout()
        plt.savefig(
            r'documentation/high_dimensionality_analysis/kmeans_profitability_clusters_by_symbol.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()

    #+----------------------------------------------------------------------------+
    #| @func: Bayes Classifier On Profitability                                   |
    #| @desc: runs a classification test to predict category by net profit        |
    #| @params: account records --> dataframe with the cleaned account records    |
    #| @return: classification report                                             |
    #+----------------------------------------------------------------------------+
    def bayes_classifier_on_profitability(self):
        ##--- convert class variable to local variable
        cfd_position_records = self.cfd_position_records
        
        #--- encode symbols
        le = LabelEncoder()
        cfd_position_records['Symbol Encoded'] = le.fit_transform(cfd_position_records['Symbol'])
        
        #--- remove symbols with < 2 trades
        class_counts = cfd_position_records['Symbol Encoded'].value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        filtered_data = cfd_position_records[cfd_position_records['Symbol Encoded'].isin(valid_classes)].copy()
        
        #--- encode symbols again for filtered data
        le = LabelEncoder()
        filtered_data['Symbol Encoded'] = le.fit_transform(filtered_data['Symbol'])
        
        #--- standardize gross profit
        scaler = StandardScaler()
        filtered_data['Gross Profit Standardized'] = scaler.fit_transform(filtered_data[['Gross Profit']])
        
        #--- features and target
        X = filtered_data[['Gross Profit Standardized']]
        y = filtered_data['Symbol Encoded']
        
        #--- train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        #--- train Naive Bayes
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        #--- predictions
        y_pred = model.predict(X_test)
        
        #--- confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        
        #--- plot heatmap
        plt.style.use('dark_background')
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_df,
            annot=True,
            fmt='d',
            cmap='viridis',
            linewidths=0.5,
            linecolor='gray'
        )
        plt.title("Naive Bayes Classification Heatmap (Profitability by Symbol)", fontsize=18)
        plt.xlabel("Predicted Symbol", fontsize=14)
        plt.ylabel("Actual Symbol", fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        #--- save figure
        plt.savefig(r'documentation/high_dimensionality_analysis/bayes_classification_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

    #+----------------------------------------------------------------------------+
    #| @func: Get PnL Chart                                                       |
    #| @desc: creates an area plot of the cumulative profit and loss performance  |
    #| @params: account records: dataframe with the cleaned account records       |
    #| @return: plt plot: the area plot of the cumulative PnL time-series         |
    #+----------------------------------------------------------------------------+
    def get_PnL_chart(self):
        ##--- convert class variable to local variable
        cfd_position_records = self.cfd_position_records
        
        ##--- apply plot dark theme
        plt.style.use('dark_background')

        # --- time series
        dates = mdates.date2num(cfd_position_records['Date Close'])
        pnl = cfd_position_records['Net Profit'].cumsum().to_numpy()

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.fill_between(dates, pnl, where=(pnl >= 0), interpolate=True,
                        color='lime', alpha=0.67)
        ax.fill_between(dates, pnl, where=(pnl < 0), interpolate=True,
                        color='red', alpha=0.67)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
        fig.autofmt_xdate()

        # --- identify regimes
        sign = np.sign(pnl)
        sign[sign == 0] = 1  # treat zero as positive to avoid splits

        regime_start = 0
        for i in range(1, len(pnl)):
            if sign[i] != sign[i - 1]:
                segment = slice(regime_start, i)

                if sign[i - 1] > 0:
                    idx = segment.start + np.argmax(pnl[segment])
                    color = 'dodgerblue'
                    va = 'bottom'
                else:
                    idx = segment.start + np.argmin(pnl[segment])
                    color = 'orange'
                    va = 'top'

                ax.plot(dates[idx], pnl[idx], 'o', color=color)
                ax.text(
                    dates[idx],
                    pnl[idx],
                    f"-${abs(pnl[idx]):.2f}" if pnl[idx] < 0 else f"${pnl[idx]:.2f}",
                    color=color,
                    fontsize=9,
                    ha='left',
                    va=va,
                    fontweight='bold'
                )

                regime_start = i

        # --- handle final regime
        segment = slice(regime_start, len(pnl))
        if sign[-1] > 0:
            idx = segment.start + np.argmax(pnl[segment])
            color = 'dodgerblue'
            va = 'bottom'
        else:
            idx = segment.start + np.argmin(pnl[segment])
            color = 'orange'
            va = 'top'

        ax.plot(dates[idx], pnl[idx], 'o', color=color)
        ax.text(
            dates[idx],
            pnl[idx],
            f"-${abs(pnl[idx]):.2f}" if pnl[idx] < 0 else f"${pnl[idx]:.2f}",
            color=color,
            fontsize=9,
            ha='left',
            va=va,
            fontweight='bold'
        )

        # --- aesthetics
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Cumulative PnL', fontsize=14)
        ax.set_title('Trading Performance: Regime Peaks & Drawdowns', fontsize=21)
        ax.grid(True, color='slategrey', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(
            r'documentation/high_dimensionality_analysis/PnL_time_series.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()

    #+----------------------------------------------------------------------------+
    #| @func: Get Stop Level by Time Held PCA Analysis                            |
    #| @desc: performs principle component analysis on target completion and time |
    #| @params: account records: dataframe with the cleaned account records       |
    #| @return: PCA scater plot of the 2 principle components                     |
    #+----------------------------------------------------------------------------+
    def get_stop_level_by_time_held_pca_analysis(self):
        ##--- convert class variable to local variable
        cfd_position_records = self.cfd_position_records
        
        ##--- calculate holding time in hours
        cfd_position_records['Date Open']  = pd.to_datetime(cfd_position_records['Date Open'])
        cfd_position_records['Date Close'] = pd.to_datetime(cfd_position_records['Date Close'])
        cfd_position_records['Time Held'] = (cfd_position_records['Date Close'] - cfd_position_records['Date Open']).dt.total_seconds() / 3600

        ##--- calculate stop level profit
        cfd_position_records = cfd_position_records[~((cfd_position_records['S/L'] == 0) & (cfd_position_records['T/P'] == 0))].copy()
        cfd_position_records['Points to S/L']     = (cfd_position_records['Price Open'] - cfd_position_records['S/L']) * cfd_position_records['Contract Size']
        cfd_position_records['Points to T/P']     = (cfd_position_records['Price Open'] - cfd_position_records['T/P']) * cfd_position_records['Contract Size']
        cfd_position_records['Points to Close']   = (cfd_position_records['Price Close'] - cfd_position_records['Price Open']) * cfd_position_records['Contract Size']
        cfd_position_records['Point Value']       = (cfd_position_records['Gross Profit'] / cfd_position_records['Points to Close'])
        cfd_position_records['Point Value']       = cfd_position_records['Point Value'].fillna(0)
        cfd_position_records['Max Loss']          = (cfd_position_records['Points to S/L'] * cfd_position_records['Point Value']) * -1
        cfd_position_records['Max Gain']          = (cfd_position_records['Points to T/P'] * cfd_position_records['Point Value']) * -1
        cfd_position_records['Percent Completed'] = cfd_position_records['Gross Profit'] / cfd_position_records['Max Gain']
        cfd_position_records['Percent Completed'] = cfd_position_records['Percent Completed'].fillna(0)
        
        ##--- drop intermediate calculation columns
        drop_cols = ['Points to S/L', 'Points to T/P', 'Points to Close', 'Point Value', 'Max Loss', 'Max Gain']
        cfd_position_records.drop(columns=drop_cols, inplace=True)
        cfd_position_records = cfd_position_records[~((cfd_position_records['S/L'] == 0) & (cfd_position_records['T/P'] == 0))].copy()
        
        ##--- prepare data for PCA
        X = cfd_position_records[['Percent Completed', 'Time Held']]
        X_scaled = StandardScaler().fit_transform(X)
        
        ##--- apply PCA with 2 components
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        
        ##--- add PCA results to dataframe
        cfd_position_records['PCA1'] = components[:, 0]
        cfd_position_records['PCA2'] = components[:, 1]
        
        ##--- color points by gross profit
        colors = np.where(
            cfd_position_records['Gross Profit'] > 0, 'tab:blue',
            np.where(cfd_position_records['Gross Profit'] < 0, 'tab:orange', 'white')
        )
        
        ##--- plot PCA results
        plt.style.use('dark_background')
        plt.figure(figsize=(14, 8))
        
        plt.scatter(
            cfd_position_records['PCA1'],
            cfd_position_records['PCA2'],
            c=colors,
            s=100,
            alpha=0.7,
            edgecolors='black'
        )
        
        legend_elements = [
            Line2D([0], [0], marker='o', linestyle='',
                label='Profitable Trade',
                markerfacecolor='tab:blue',
                markeredgecolor='black',
                markersize=10),

            Line2D([0], [0], marker='o', linestyle='',
                label='Losing Trade',
                markerfacecolor='tab:orange',
                markeredgecolor='black',
                markersize=10),

            Line2D([0], [0], marker='o', linestyle='',
                label='Break Even Trade',
                markerfacecolor='white',
                markeredgecolor='black',
                markersize=10)
        ]

        plt.legend(
            handles=legend_elements,
            loc='upper right',
            frameon=True,
            facecolor='black',
            edgecolor='grey',
            fontsize=10
        )
        
        plt.title('PCA Analysis: Stop Level Profitability vs Time Held', fontsize=20)
        plt.xlabel('Principal Component 1', fontsize=14)
        plt.ylabel('Principal Component 2', fontsize=14)
        plt.grid(True, linestyle=':', color='grey', alpha=0.5)
        plt.tight_layout()
        plt.savefig(r'documentation/high_dimensionality_analysis/stop_level_by_time_held_pca.png', dpi=300, bbox_inches='tight')
        plt.show()


##--- execute the main method
if __name__ == '__main__':
    ##--- change directory to parent directory
    os.chdir(get_parent_directory())

    ##--- read the position data
    cfd_position_records = pd.read_csv(r'data/complete_cfd_position_records.csv')
    
    ##--- initialize the class
    modeler = High_Dimensional_Analysis_Modeler(cfd_position_records = cfd_position_records)
    
    ##--- plot the high dimension models
    modeler.cluster_profitability_by_symbol()
    modeler.kmeans_profitability_by_symbol()
    modeler.bayes_classifier_on_profitability()
    modeler.get_PnL_chart()
    modeler.get_stop_level_by_time_held_pca_analysis()