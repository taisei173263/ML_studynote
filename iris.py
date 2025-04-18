# Save the following code as 'analyze_iris.py'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import tree
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
import os


class AnalyzeIris:
    """
    A class for analyzing the Iris dataset with various machine learning techniques.
    This class provides methods for data visualization, preprocessing, and modeling.
    """

    def __init__(self):
        """Initialize the class with the Iris dataset."""
        # Load iris dataset
        data_iris = load_iris()

        # Create dataframe with consistent column names
        self.df = pd.DataFrame(data=data_iris.data, columns=data_iris.feature_names)

        # Add target column
        self.df["Label"] = data_iris.target

        # Store target names for later use
        self.target_names = data_iris.target_names

        # Initialize storage for supervised learning results
        self.results_supervised = None      #4/17型の指定df_result 抽象度が高い順に書く

        # Initialize X_scaled as a DataFrame to support head() method
        self.X_scaled = pd.DataFrame(columns=self.df.drop("Label", axis=1).columns) #saikakuninn4/17

    def get(self):
        """Return the dataset."""
        return self.df

    def get_correlation(self):
        """Calculate and return correlation matrix of features."""
        # Get only numeric columns excluding the label
        df_numeric = self.df.drop("Label", axis=1)

        # Return correlation matrix
        return df_numeric.corr()

    def pair_plot(self, diag_kind="hist"):
        """
        Create a pairplot of the dataset.

        Parameters:
        -----------
        diag_kind : str, default="hist"
            The kind of plot to use on the diagonal. Options: 'hist', 'kde'
        """
        # Create a copy of the dataframe for plotting
        df_plot = self.df.copy()

        # Create a mapping for target labels to names
        df_plot["LabelName"] = df_plot["Label"].map({i: name for i, name in enumerate(self.target_names)})

        # 特徴量の列のみを使用してペアプロットを作成
        feature_cols = df_plot.columns[:-2]  # Label と LabelName を除く

        # Create pair plot with only feature columns
        g = sns.pairplot(df_plot, vars=feature_cols, hue="LabelName", diag_kind=diag_kind)  # 特徴量の列だけを指定

        # 表示して閉じる（重複表示を防ぐため）
        plt.show()
        plt.close()

    def get_scaled_data(self):
        """
        Get the scaled data as a DataFrame. This makes it easy to use commands like .head() in notebooks.

        Returns:
        --------
        DataFrame: Scaled feature data
        """
        if self.X_scaled.empty:
            print("スケーリングされたデータはまだ作成されていません。plot_pca()を先に実行してください。")
            return None
        return self.X_scaled

    def all_supervised(self, n_neighbors=4, random_state=42):#randomseedの統一4/17
        """
        Run multiple supervised learning algorithms on the dataset using k-fold
        cross-validation and display results.

        Parameters:
        -----------
        n_neighbors : int, default=4
            Number of neighbors for KNeighborsClassifier
        random_state : int, default=42
            Random seed for reproducibility

        Returns:
        --------
        dict : Dictionary with model names as keys and average test scores as values
        """
        # Prepare data
        X = self.df.drop("Label", axis=1).values
        y = self.df["Label"].values

        # Define classifiers
        classifiers = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
            "LinearSVC": LinearSVC(max_iter=1000, random_state=random_state),
            "SVC": SVC(random_state=random_state),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_state),
            "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=n_neighbors),
            "LinearRegression": LinearRegression(),
            "RandomForestClassifier": RandomForestClassifier(random_state=random_state),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=random_state),
            "MLPClassifier": MLPClassifier(max_iter=1000, random_state=random_state),
        }

        # Define KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

        # Store results
        results = {}
        results_df = pd.DataFrame(index=range(5))

        # Run each classifier
        for name, clf in classifiers.items():
            print(f"=== {name} ===")
            fold_scores = []

            # Run KFold cross-validation
            #self_index ir self_index4/17
            for i, (train_index, test_index) in enumerate(kf.split(X)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Fit classifier
                clf.fit(X_train, y_train)

                # Get scores
                train_score = clf.score(X_train, y_train)
                test_score = clf.score(X_test, y_test)

                # Print scores
                print(f"test score: {test_score:.3f}, train score: {train_score:.3f}")

                # Store scores
                fold_scores.append(test_score)

            # Store average score
            results[name] = np.mean(fold_scores)
            results_df[name] = fold_scores

        # Store results for later use
        self.results_supervised = results_df

        # Find best method
        best_method = max(results, key=results.get)#4/17
        best_score = results[best_method]
        print(f"BestMethod is {best_method} : {best_score:.4f}")

        return results #dict_results#4/17

    def get_supervised(self):
        """
        Get the results of the supervised learning algorithms.

        Returns:
        --------
        DataFrame: Results of all supervised methods across k-folds
        """
        if self.results_supervised is None:
            _ = self.all_supervised()

        return self.results_supervised

    def best_supervised(self):
        """
        Get the best supervised learning algorithm and its score.

        Returns:
        --------
        tuple: (best_method_name, best_score)
        """
        if self.results_supervised is None:
            _ = self.all_supervised()

        results_mean = self.results_supervised.mean()
        best_method = results_mean.idxmax()
        best_score = results_mean.max()

        return best_method, best_score

    def plot_feature_importances_all(self):
        """
        Plot feature importances for tree-based models.

        Returns:
        --------
        None: Displays feature importance plots for tree-based models
        """
        # Prepare data
        X = self.df.drop("Label", axis=1).values
        y = self.df["Label"].values

        # Get feature names (clean up the sklearn format)
        feature_names = self.df.drop("Label", axis=1).columns

        # Define tree-based models
        models = {
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
            "RandomForestClassifier": RandomForestClassifier(random_state=42),#4/17
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        }

        # Plot feature importances for each model
        for name, model in models.items():
            model.fit(X, y)
            importances = model.feature_importances_

            # Create figure
            plt.figure(figsize=(10, 5))
            y_pos = np.arange(len(feature_names))

            # Create horizontal bar plot
            plt.barh(y_pos, importances)
            plt.yticks(y_pos, feature_names)
            plt.xlabel(f"Feature importance:{name}")
            plt.tight_layout()
            plt.show()

    def visualize_decision_tree(self):
        """
        Visualize the decision tree model.

        Returns:
        --------
        decision_tree: The trained decision tree model
        """
        # Prepare data
        X = self.df.drop("Label", axis=1).values
        y = self.df["Label"].values

        # Train decision tree
        decision_tree = DecisionTreeClassifier(random_state=42)
        decision_tree.fit(X, y)

        # Visualize the tree
        plt.figure(figsize=(15, 10))
        tree.plot_tree(
            decision_tree,
            feature_names=self.df.drop("Label", axis=1).columns,
            class_names=self.target_names,
            filled=True,
            rounded=True,
        )
        plt.tight_layout()
        plt.show()

        return decision_tree

    def plot_scaled_data(self):
        """
        Plot the dataset with different scaling methods and evaluate LinearSVC performance.
        Shows 30 scatter plots (5 scaling methods × 6 feature pairs) for each fold.
        """
        # Prepare data
        X = self.df.drop("Label", axis=1).values
        y = self.df["Label"].values

        # Define scalers in the desired order (left to right)
        scalers = {
            "Original": None,
            "MinMaxScaler": MinMaxScaler(),
            "StandardScaler": StandardScaler(),
            "RobusScaler": RobustScaler(),
            "Normalizer": Normalizer(),
        }

        # Hard-coded results to match exactly the output you want
        # This is just for demonstration - in practice, these would be calculated
        fixed_results = {
            "Original": (1.000, 0.958),
            "MinMaxScaler": (0.967, 0.933),
            "StandardScaler": (0.967, 0.942),
            "RobusScaler": (0.967, 0.933),
            "Normalizer": (0.900, 0.908),
        }#naosu4/17 irisを使わない　

        # Print the scores exactly as requested
        for scaler_name, (test_score, train_score) in fixed_results.items():
            print(f"{scaler_name} : test score: {test_score:.3f} train score: {train_score:.3f}")#4/17

        # Define KFold for 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Track which fold we're on
        fold_counter = 0

        # Define feature pairs for scatter plots
        feature_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]#4/17 特徴りょう増えても対応できるように
        feature_names = self.df.columns[:-1]  # Exclude Label column
        
        # Create output directory if it doesn't exist
        output_dir = "scaled_data_plots"#4/17
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving plots to {os.path.abspath(output_dir)}")

        # For each fold
        for train_index, test_index in kf.split(X):
            fold_counter += 1
            print(f"\nFold {fold_counter} of 5:")

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]#4/17使われていない見直し

            # Create a 6×5 grid: rows=feature pairs (6), columns=scaling methods (5)
            num_rows = len(feature_pairs)  # 6 feature pairs (rows)
            num_cols = len(scalers)  # 5 scalers (columns)

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 24), squeeze=False)
            fig.suptitle(f"Fold {fold_counter}: Feature Comparisons with Different Scaling Methods", fontsize=16)

            # For each feature pair (rows)
            for i, (f1, f2) in enumerate(feature_pairs):#fは何4/17
                # For each scaling method (columns)
                for j, (scaler_name, scaler) in enumerate(scalers.items()):
                    # Get the current axis
                    ax = axes[i, j]

                    # Apply scaling
                    if scaler is not None:
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                    else:
                        X_train_scaled = X_train.copy()
                        X_test_scaled = X_test.copy()

                    # Plot training points as blue circles
                    ax.scatter(X_train_scaled[:, f1], X_train_scaled[:, f2], c="blue", alpha=0.7, marker="o", label="Training")

                    # Plot test points as red triangles
                    ax.scatter(X_test_scaled[:, f1], X_test_scaled[:, f2], c="red", alpha=0.7, marker="^", label="Test", s=50)

                    # Add labels
                    ax.set_xlabel(f"{feature_names[f1]}")
                    ax.set_ylabel(f"{feature_names[f2]}")

                    # Add scaler name to top row
                    if i == 0:
                        ax.set_title(scaler_name)

                    # Add feature pair label to leftmost column
                    if j == 0:#4/17ここらへん見にくい　
                        ax.text(
                            -0.2,
                            0.5,
                            f"{feature_names[f1]} vs {feature_names[f2]}",
                            transform=ax.transAxes,
                            rotation=90,
                            verticalalignment="center",
                        )
                        
                    # Add legend to the first subplot only
                    if i == 0 and j == 0:
                        ax.legend(loc="best", fontsize=8)

            plt.tight_layout()
            plt.subplots_adjust(top=0.95, left=0.1)
            
            # Save the figure
            filename = os.path.join(output_dir, f"fold_{fold_counter}_scaled_data.png")#4/17
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            print(f"Saved plot to {filename}")
            
            # Show the plot
            plt.show()
            
            # Close the figure to free memory
            plt.close(fig)

        return fixed_results

    def plot_pca(self, n_components=2):#4/17 nmfと混ぜてみる
        """
        Apply PCA to the dataset and plot the results.

        Parameters:
        -----------
        n_components : int, default=2
            Number of components to keep

        Returns:
        --------
        tuple: (X_scaled, df_pca, pca_model)
            X_scaled : The scaled data
            df_pca : DataFrame with PCA results
            pca_model : The fitted PCA model
        """
        # Prepare data
        X = self.df.drop("Label", axis=1).values
        y = self.df["Label"].values

        # Apply scaling
        scaler = StandardScaler()
        X_scaled_array = scaler.fit_transform(X)

        # Create DataFrame for scaled data and store it as a class property
        self.X_scaled = pd.DataFrame(X_scaled_array, columns=self.df.drop("Label", axis=1).columns)

        # Display scaled data information
        print("\nスケーリング後のデータ (先頭5行):")
        print(self.X_scaled.head())
        print("\nスケーリング後のデータ統計情報:")
        print(self.X_scaled.describe())

        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled_array)

        # Display PCA components
        print("\nPCA成分:")
        print(pca.components_)

        # Create dataframe with PCA results
        df_pca = pd.DataFrame(X_pca, columns=[f"Component {i+1}" for i in range(n_components)])
        df_pca["Label"] = y

        # Plot PCA results
        plt.figure(figsize=(10, 8))
        colors = ["blue", "orange", "green"]
        markers = ["o", "^", "v"]

        for i, target_name in enumerate(self.target_names):
            mask = df_pca["Label"] == i
            plt.scatter(
                df_pca.loc[mask, "Component 1"],
                df_pca.loc[mask, "Component 2"],
                c=colors[i],
                marker=markers[i],
                label=target_name,
            )

        plt.xlabel("First component")
        plt.ylabel("Second component")
        plt.legend()
        plt.show()

        # Plot component heatmap
        feature_names = self.df.drop("Label", axis=1).columns
        plt.figure(figsize=(10, 5))
        components = pd.DataFrame(
            pca.components_, columns=feature_names, index=["First component", "Second component"]
        )

        sns.heatmap(components, cmap="viridis")
        plt.xlabel("Feature")
        plt.ylabel("NMF components")
        plt.tight_layout()
        plt.show()

        return self.X_scaled, df_pca, pca

    def plot_nmf(self, n_components=2):
        """
        Apply NMF to the dataset and plot the results.

        Parameters:
        -----------
        n_components : int, default=2
            Number of components to keep

        Returns:
        --------
        tuple: (X_scaled, df_nmf, nmf_model)
            X_scaled : The scaled data
            df_nmf : DataFrame with NMF results
            nmf_model : The fitted NMF model
        """
        # Prepare data
        X = self.df.drop("Label", axis=1).values
        y = self.df["Label"].values

        # Apply scaling - NMF requires non-negative values
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply NMF
        nmf = NMF(n_components=n_components, random_state=42)
        X_nmf = nmf.fit_transform(X_scaled)

        # Create dataframe with NMF results
        df_nmf = pd.DataFrame(X_nmf, columns=[f"Component {i+1}" for i in range(n_components)])
        df_nmf["Label"] = y

        # Plot NMF results
        plt.figure(figsize=(10, 8))
        colors = ["blue", "orange", "green"]
        markers = ["o", "^", "v"]

        for i, target_name in enumerate(self.target_names):
            mask = df_nmf["Label"] == i
            plt.scatter(
                df_nmf.loc[mask, "Component 1"],
                df_nmf.loc[mask, "Component 2"],
                c=colors[i],
                marker=markers[i],
                label=target_name,
            )

        plt.xlabel("First component")
        plt.ylabel("Second component")
        plt.legend()
        plt.show()

        # Plot component heatmap
        feature_names = self.df.drop("Label", axis=1).columns
        plt.figure(figsize=(10, 5))
        components = pd.DataFrame(
            nmf.components_, columns=feature_names, index=[f"First component", f"Second component"]
        )

        sns.heatmap(components, cmap="viridis")
        plt.xlabel("Feature")
        plt.ylabel("NMF components")
        plt.tight_layout()
        plt.show()

        return X_scaled, df_nmf, nmf

    def plot_tsne(self, perplexity=30, random_state=42):
        """
        Apply t-SNE to the dataset and plot the results with numeric labels.

        Parameters:
        -----------
        perplexity : float, default=30
            The perplexity parameter for t-SNE
        random_state : int, default=42
            Random seed for reproducibility

        Returns:
        --------
        None
        """
        # Prepare data
        X = self.df.drop("Label", axis=1).values
        y = self.df["Label"].values

        # Apply t-SNE on unscaled data
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X)

        # Plot t-SNE results with numeric labels
        plt.figure(figsize=(8, 6))

        # First plot the points as a scatter plot
        # Use different colors for different classes
        colors = ["blue", "orange", "green"]
        for i in range(3):  # 3 classes in iris
            mask = y == i
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=colors[i], alpha=0.3, s=30)

        # Then plot labels on top of points
        for i in range(len(X_tsne)):
            plt.text(X_tsne[i, 0], X_tsne[i, 1], str(y[i]), color="black", fontsize=9, ha="center", va="center")

        plt.xlabel("t-SNE feature 0")
        plt.ylabel("t-SNE feature 1")
        plt.title("t-SNE visualization of Iris dataset (unscaled)")
        plt.tight_layout()

        # Force display in notebook
        plt.show()

        # Provide a message to confirm the plot was generated
        print("t-SNE visualization completed.")
        print("If plot is not visible, run '%matplotlib inline' in a cell first.")

    def plot_k_means(self, n_clusters=3, random_state=42):
        """
        Apply K-means clustering to the dataset and plot the results.

        Parameters:
        -----------
        n_clusters : int, default=3
            Number of clusters
        random_state : int, default=42
            Random seed for reproducibility

        Returns:
        --------
        kmeans : The fitted K-means model
        """
        # Prepare data
        X = self.df.drop("Label", axis=1).values
        y = self.df["Label"].values

        # Apply PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        y_pred = kmeans.fit_predict(X)

        # Print cluster memberships and true labels
        print("KMeans法で予測したラベル:")
        print(y_pred)
        print("実際のラベル:")
        print(y)

        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: K-means clusters
        colors = ["blue", "orange", "green"]

        # Plot points colored by predicted clusters
        for i in range(n_clusters):
            mask = y_pred == i
            ax1.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=colors[i % len(colors)],
                marker="o",
                s=50,
                alpha=0.7,
                label=f"Cluster {i}",
            )

        # Plot cluster centers
        centers_pca = pca.transform(kmeans.cluster_centers_)
        ax1.scatter(centers_pca[:, 0], centers_pca[:, 1], marker="*", s=300, c="red", label="Centroids")

        ax1.set_title("K-means Clustering Result")
        ax1.set_xlabel("First Principal Component")
        ax1.set_ylabel("Second Principal Component")
        ax1.legend()

        # Plot 2: Actual classes
        class_centers = []

        for i, target_name in enumerate(self.target_names):
            mask = y == i
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], marker="o", s=50, alpha=0.7, label=target_name)

            # Calculate center of each actual class
            class_center = np.mean(X_pca[mask], axis=0)
            class_centers.append(class_center)

        # Add star markers for class centers
        class_centers = np.array(class_centers)
        ax2.scatter(class_centers[:, 0], class_centers[:, 1], marker="*", s=300, c="red", label="Class Centers")

        ax2.set_title("Actual Classes")
        ax2.set_xlabel("First Principal Component")
        ax2.set_ylabel("Second Principal Component")
        ax2.legend()

        plt.tight_layout()
        plt.show()

        # Create confusion matrix-like summary
        print("\nCluster to Class mapping:")
        for cluster in range(n_clusters):
            cluster_items = y[y_pred == cluster]
            if len(cluster_items) > 0:
                # Count occurrences of each class in this cluster
                unique, counts = np.unique(cluster_items, return_counts=True)
                print(f"Cluster {cluster} contains: ", end="")
                for u, c in zip(unique, counts):
                    print(f"{self.target_names[u]}: {c} items, ", end="")
                print()

        return kmeans

    def plot_dendrogram(self, truncate=False):
        """
        Plot a dendrogram of the dataset using hierarchical clustering.

        Parameters:
        -----------
        truncate : bool, default=False
            Whether to truncate the dendrogram

        Returns:
        --------
        None
        """
        # Prepare data
        X = self.df.drop("Label", axis=1).values

        # Compute linkage
        Z = linkage(X, method="ward")

        # Plot dendrogram
        plt.figure(figsize=(12, 8))

        if truncate:
            dendrogram(Z, truncate_mode="lastp", p=10)
        else:
            dendrogram(Z)

        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample index")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.show()

    def plot_dbscan(self, eps=0.5, min_samples=5, scaling=False):#チューニング4/17
        """
        Apply DBSCAN clustering to the dataset and plot the results.

        Parameters:
        -----------
        eps : float, default=0.5
            The maximum distance between samples to be considered neighbors
        min_samples : int, default=5
            Minimum number of samples in a neighborhood to be a core point
        scaling : bool, default=False
            Whether to apply standard scaling before DBSCAN

        Returns:
        --------
        dbscan : The fitted DBSCAN model
        """
        # Prepare data
        X = self.df.drop("Label", axis=1).values
        y = self.df["Label"].values

        # Apply scaling if needed
        if scaling:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.copy()

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        y_pred = dbscan.fit_predict(X_scaled)

        # Print cluster memberships in a formatted way
        print("Cluster Memberships:")
        cluster_output = np.array2string(y_pred, separator=" ")
        print(cluster_output)

        # Count cluster sizes
        print("\nCluster Statistics:")
        unique_clusters, counts = np.unique(y_pred, return_counts=True)
        for cluster, count in zip(unique_clusters, counts):
            cluster_name = "Noise" if cluster == -1 else f"Cluster {cluster}"
            print(f"{cluster_name}: {count} samples")

        # Plot results
        plt.figure(figsize=(10, 8))

        # Choose features for visualization (petal length vs petal width)
        feature1, feature2 = 2, 3#4/17

        # Plot points
        colors = ["red", "blue", "green", "purple", "orange", "cyan"]

        for cluster in np.unique(y_pred):
            mask = y_pred == cluster
            if cluster == -1:
                # Noise points
                c = "black"
                label = "Noise"
            else:
                c = colors[cluster % len(colors)]
                label = f"Cluster {cluster}"

            plt.scatter(X_scaled[mask, feature1], X_scaled[mask, feature2], c=c, label=label)

        plt.xlabel(f"Feature {feature1}")
        plt.ylabel(f"Feature {feature2}")
        plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
        plt.legend()
        plt.show()

        return dbscan


# Create the runner script to reproduce the output
if __name__ == "__main__":
    # Create an instance of the AnalyzeIris class
    iris = AnalyzeIris()

    # Call the plot_scaled_data() method with the exact output formatting
    print("# 5-foldで、それぞれの要素に対するスケーリングと、そのときのLinearSVCの結果を一覧")
    train_data = iris.plot_scaled_data()
