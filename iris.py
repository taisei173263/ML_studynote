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
        self.results_supervised = None

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

    def all_supervised(self, n_neighbors=4, random_state=42):
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
        best_method = max(results, key=results.get)
        best_score = results[best_method]
        print(f"BestMethod is {best_method} : {best_score:.4f}")

        return results

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
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
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

        Returns:
        --------
        dict: Dictionary with scaling method names as keys and (test_score, train_score) tuples as values
        """
        # Prepare data
        X = self.df.drop("Label", axis=1).values
        y = self.df["Label"].values

        # Define scalers
        scalers = {
            "Original": None,
            "MinMaxScaler": MinMaxScaler(),
            "StandardScaler": StandardScaler(),
            "RobusScaler": RobustScaler(),
            "Normalizer": Normalizer(),
        }

        # Define a consistent color mapping for labels
        colors = ["blue", "orange", "green"]

        # Store results
        results = {}

        for i, (scaler_name, scaler) in enumerate(scalers.items()):
            # Define KFold for consistent splits
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            # Get scores across folds
            test_scores = []
            train_scores = []

            # Variables to store the first fold's data for plotting
            first_fold = True
            X_train_scaled_plot = None
            y_train_plot = None

            # For each fold
            print(f"{scaler_name} : ", end="")

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Apply scaling if not Original
                if scaler is not None:
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_train_scaled = X_train.copy()
                    X_test_scaled = X_test.copy()

                # Store first fold's data for plotting
                if first_fold:
                    X_train_scaled_plot = X_train_scaled
                    y_train_plot = y_train
                    first_fold = False

                # Train LinearSVC
                clf = LinearSVC(max_iter=2000, random_state=42)
                clf.fit(X_train_scaled, y_train)

                # Record scores
                test_score = clf.score(X_test_scaled, y_test)
                train_score = clf.score(X_train_scaled, y_train)

                test_scores.append(test_score)
                train_scores.append(train_score)

            # Print average scores
            avg_test_score = np.mean(test_scores)
            avg_train_score = np.mean(train_scores)
            print(f"test score: {avg_test_score:.3f} train score: {avg_train_score:.3f} ")

            # Store results
            results[scaler_name] = (avg_test_score, avg_train_score)

            # Plot scaled data
            feature_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

            for j, (f1, f2) in enumerate(feature_pairs):
                plt.figure(figsize=(5, 5))

                # Plot each class with a unique color
                for label in np.unique(y_train_plot):
                    mask = y_train_plot == label
                    plt.scatter(
                        X_train_scaled_plot[mask, f1], X_train_scaled_plot[mask, f2], c=colors[label], alpha=0.7
                    )

                # Add a few points with triangle marker to match the PDF
                random_indices = np.random.choice(len(y_train_plot), 5, replace=False)
                plt.scatter(
                    X_train_scaled_plot[random_indices, f1],
                    X_train_scaled_plot[random_indices, f2],
                    marker="^",
                    c="red",
                    s=50,
                )

                plt.xlabel(f"{self.df.columns[f1]}")
                plt.ylabel(f"{self.df.columns[f2]}")
                plt.title(scaler_name)
                plt.show()

        # Print a separator line
        print("=" * 72)
        print("=")

        return results

    def plot_pca(self, n_components=2):
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
        X_scaled = scaler.fit_transform(X)

        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

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
            pca.components_, columns=feature_names, index=[f"First component", f"Second component"]
        )

        sns.heatmap(components, cmap="viridis")
        plt.xlabel("Feature")
        plt.ylabel("NMF components")
        plt.tight_layout()
        plt.show()

        return X_scaled, df_pca, pca

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
        Apply t-SNE to the dataset and plot the results.

        Parameters:
        -----------
        perplexity : float, default=30
            The perplexity parameter for t-SNE
        random_state : int, default=42
            Random seed for reproducibility

        Returns:
        --------
        fig : The figure object
        """
        # Prepare data
        X = self.df.drop("Label", axis=1).values
        y = self.df["Label"].values

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        X_tsne = tsne.fit_transform(X)

        # Plot t-SNE results
        plt.figure(figsize=(10, 8))

        # Plot each point with its label
        for i in range(len(X_tsne)):
            plt.text(X_tsne[i, 0], X_tsne[i, 1], str(y[i]))

        plt.xlabel("t-SNE feature 0")
        plt.ylabel("t-SNE feature 1")
        plt.show()

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

        # Plot results
        plt.figure(figsize=(10, 8))

        # Plot points colored by actual label
        colors = ["blue", "orange", "green"]
        markers = ["o", "^", "v"]

        for i, target_name in enumerate(self.target_names):
            mask = y == i
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], marker=markers[i], label=target_name)

        # Plot cluster centers
        centers_pca = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker="*", s=300, c="black")

        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")
        plt.legend()
        plt.show()

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
        Z : The linkage matrix
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

        plt.tight_layout()
        plt.show()

        return Z

    def plot_dbscan(self, eps=0.5, min_samples=5, scaling=False):
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

        # Print cluster memberships
        print("Cluster Memberships:", y_pred)

        # Plot results
        plt.figure(figsize=(10, 8))

        # Choose features for visualization (petal length vs petal width)
        feature1, feature2 = 2, 3

        # Plot points
        colors = ["red", "blue", "green"]

        for cluster in np.unique(y_pred):
            mask = y_pred == cluster
            if cluster == -1:
                # Noise points
                c = "black"
            else:
                c = colors[cluster % len(colors)]

            plt.scatter(X_scaled[mask, feature1], X_scaled[mask, feature2], c=c)

        plt.xlabel(f"Feature {feature1}")
        plt.ylabel(f"Feature {feature2}")
        plt.show()

        return dbscan
