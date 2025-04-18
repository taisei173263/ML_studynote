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
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import Bunch


class DataAnalyzer:
    """
    A class for analyzing tabular datasets with various machine learning techniques.
    This class provides methods for data visualization, preprocessing, and modeling.
    """

    def __init__(self):
        """Initialize the class with the dataset."""
        # Load dataset with explicit type handling
        iris_data = load_iris()

        # Convert to dictionary if necessary to avoid typing issues
        iris_dict = {
            "data": iris_data.data if hasattr(iris_data, "data") else iris_data["data"],
            "target": iris_data.target if hasattr(iris_data, "target") else iris_data["target"],
            "feature_names": (
                iris_data.feature_names if hasattr(iris_data, "feature_names") else iris_data["feature_names"]
            ),
            "target_names": (
                iris_data.target_names if hasattr(iris_data, "target_names") else iris_data["target_names"]
            ),
        }

        # Create dataframe with consistent column names
        self.df_data = pd.DataFrame(data=iris_dict["data"], columns=iris_dict["feature_names"])

        # Add target column
        self.df_data["Label"] = iris_dict["target"]

        # Store target names for later use
        self.target_names = iris_dict["target_names"]

        # Initialize storage for supervised learning results
        self.df_results_supervised = None

        # Initialize X_scaled as a DataFrame to support head() method
        self.df_scaled_features = pd.DataFrame(columns=self.df_data.drop("Label", axis=1).columns)

    def get(self):
        """Return the dataset."""
        return self.df_data

    def get_correlation(self):
        """Calculate and return correlation matrix of features."""
        # Get only numeric columns excluding the label
        df_numeric = self.df_data.drop("Label", axis=1)

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
        # Validate diag_kind parameter
        valid_diag_kinds = ["auto", "hist", "kde", None]
        if diag_kind not in valid_diag_kinds:
            print(f"Warning: diag_kind '{diag_kind}' not in {valid_diag_kinds}. Using 'hist' instead.")
            diag_kind = "hist"

        # Create a copy of the dataframe for plotting
        df_plot = self.df_data.copy()

        # Create a mapping for target labels to names
        df_plot["LabelName"] = df_plot["Label"].map({i: name for i, name in enumerate(self.target_names)})

        # Use only feature columns for pairplot
        feature_cols = df_plot.columns[:-2]  # Exclude Label and LabelName

        # Create pair plot with only feature columns
        # Use a fixed literal value for diag_kind to satisfy type checking
        if diag_kind == "hist":
            _ = sns.pairplot(df_plot, vars=feature_cols, hue="LabelName", diag_kind="hist")
        elif diag_kind == "kde":
            _ = sns.pairplot(df_plot, vars=feature_cols, hue="LabelName", diag_kind="kde")
        elif diag_kind == "auto":
            _ = sns.pairplot(df_plot, vars=feature_cols, hue="LabelName", diag_kind="auto")
        else:
            _ = sns.pairplot(df_plot, vars=feature_cols, hue="LabelName", diag_kind=None)

        # Display and close (to prevent duplicate displays)
        plt.show()
        plt.close()

    def get_scaled_data(self):
        """
        Get the scaled data as a DataFrame. This makes it easy to use commands like .head() in notebooks.

        Returns:
        --------
        DataFrame: Scaled feature data
        """
        if self.df_scaled_features.empty:
            print("スケーリングされたデータはまだ作成されていません。plot_pca()を先に実行してください。")
            return None
        return self.df_scaled_features

    def all_supervised(self, n_neighbors=4, random_seed=42):
        """
        Run multiple supervised learning algorithms on the dataset using k-fold
        cross-validation and display results.

        Parameters:
        -----------
        n_neighbors : int, default=4
            Number of neighbors for KNeighborsClassifier
        random_seed : int, default=42
            Random seed for reproducibility

        Returns:
        --------
        dict : Dictionary with model names as keys and average test scores as values
        """
        # Prepare data
        X_features = self.df_data.drop("Label", axis=1).values
        y_labels = self.df_data["Label"].values

        # Define classifiers
        classifiers = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_seed),
            "LinearSVC": LinearSVC(max_iter=1000, random_state=random_seed),
            "SVC": SVC(random_state=random_seed),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_seed),
            "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=n_neighbors),
            "LinearRegression": LinearRegression(),
            "RandomForestClassifier": RandomForestClassifier(random_state=random_seed),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=random_seed),
            "MLPClassifier": MLPClassifier(max_iter=1000, random_state=random_seed),
        }

        # Define KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

        # Store results
        dict_results = {}
        df_fold_results = pd.DataFrame(index=range(5))

        # Run each classifier
        for model_name, classifier in classifiers.items():
            print(f"=== {model_name} ===")
            fold_scores = []

            # Run KFold cross-validation
            for fold_index, (train_index, test_index) in enumerate(kf.split(X_features)):
                X_train, X_test = X_features[train_index], X_features[test_index]
                y_train, y_test = y_labels[train_index], y_labels[test_index]

                # Fit classifier
                classifier.fit(X_train, y_train)

                # Get scores
                train_score = classifier.score(X_train, y_train)
                test_score = classifier.score(X_test, y_test)

                # Print scores
                print(f"test score: {test_score:.3f}, train score: {train_score:.3f}")

                # Store scores
                fold_scores.append(test_score)

            # Store average score
            dict_results[model_name] = np.mean(fold_scores)
            df_fold_results[model_name] = fold_scores

        # Store results for later use
        self.df_results_supervised = df_fold_results

        # Find best method - use more explicit approach to avoid type errors
        best_method = None
        best_score = -float("inf")
        for method, score in dict_results.items():
            if score > best_score:
                best_score = score
                best_method = method

        print(f"BestMethod is {best_method} : {best_score:.4f}")

        return dict_results

    def get_supervised(self):
        """
        Get the results of the supervised learning algorithms.

        Returns:
        --------
        DataFrame: Results of all supervised methods across k-folds
        """
        if self.df_results_supervised is None:
            _ = self.all_supervised()

        return self.df_results_supervised

    def best_supervised(self):
        """
        Get the best supervised learning algorithm and its score.

        Returns:
        --------
        tuple: (best_method_name, best_score)
        """
        if self.df_results_supervised is None:
            _ = self.all_supervised()

        # Ensure df_results_supervised is a DataFrame
        if self.df_results_supervised is not None and not self.df_results_supervised.empty:
            results_mean = self.df_results_supervised.mean()
            best_method = results_mean.idxmax()
            best_score = results_mean.max()
            return best_method, best_score
        else:
            return None, 0.0

    def plot_feature_importances_all(self):
        """
        Plot feature importances for tree-based models.

        Returns:
        --------
        None: Displays feature importance plots for tree-based models
        """
        # Prepare data
        X_features = self.df_data.drop("Label", axis=1).values
        y_labels = self.df_data["Label"].values

        # Get feature names
        feature_names = list(self.df_data.drop("Label", axis=1).columns)

        # Define tree-based models with consistent random seed
        random_seed = 42
        tree_models = {
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_seed),
            "RandomForestClassifier": RandomForestClassifier(random_state=random_seed),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=random_seed),
        }

        # Plot feature importances for each model
        for model_name, model in tree_models.items():
            model.fit(X_features, y_labels)
            importances = model.feature_importances_

            # Create figure
            plt.figure(figsize=(10, 5))
            y_pos = np.arange(len(feature_names))

            # Create horizontal bar plot
            plt.barh(y_pos, importances)
            plt.yticks(y_pos, feature_names)
            plt.xlabel(f"Feature importance: {model_name}")
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
        X_features = self.df_data.drop("Label", axis=1).values
        y_labels = self.df_data["Label"].values.astype(int)  # Ensure y is int type

        # Train decision tree
        random_seed = 42
        decision_tree = DecisionTreeClassifier(random_state=random_seed)
        decision_tree.fit(X_features, y_labels)

        # Visualize the tree
        plt.figure(figsize=(15, 10))
        tree.plot_tree(
            decision_tree,
            feature_names=list(self.df_data.drop("Label", axis=1).columns),
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
        Shows scatter plots for all feature pairs with different scaling methods for each fold.

        Returns:
        --------
        dict_scaler_results : Dictionary with scaler names as keys and tuples of (test_score, train_score) as values
        """
        # Prepare data
        X_features = self.df_data.drop("Label", axis=1).values
        y_labels = self.df_data["Label"].values

        # Get the number of features
        num_features = X_features.shape[1]

        # Define scalers in the desired order (left to right)
        scalers = {
            "Original": None,
            "MinMaxScaler": MinMaxScaler(),
            "StandardScaler": StandardScaler(),
            "RobusScaler": RobustScaler(),
            "Normalizer": Normalizer(),
        }

        # Define KFold for 5-fold cross-validation
        random_seed = 42
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

        # Dictionary to store results for each scaler
        dict_scaler_results = {}

        # Dictionary to store fold-by-fold results
        fold_results = {scaler_name: [] for scaler_name in scalers.keys()}

        # Calculate actual results for each scaler
        for scaler_name, scaler in scalers.items():
            fold_test_scores = []
            fold_train_scores = []

            # Print header for this scaler
            print(f"\n=== {scaler_name} ===")

            # Run k-fold cross-validation
            for fold_idx, (train_index, test_index) in enumerate(kf.split(X_features)):
                X_train, X_test = X_features[train_index], X_features[test_index]
                y_train, y_test = y_labels[train_index], y_labels[test_index]

                # Apply scaling if needed
                if scaler is not None:
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_train_scaled = X_train.copy()
                    X_test_scaled = X_test.copy()

                # Train and evaluate LinearSVC
                classifier = LinearSVC(max_iter=1000, random_state=random_seed)
                classifier.fit(X_train_scaled, y_train)

                # Record scores
                train_score = classifier.score(X_train_scaled, y_train)
                test_score = classifier.score(X_test_scaled, y_test)

                # Print fold-specific scores
                print(f"  Fold {fold_idx+1}: test score: {test_score:.3f}, train score: {train_score:.3f}")

                fold_test_scores.append(test_score)
                fold_train_scores.append(train_score)

                # Store results for this fold
                fold_results[scaler_name].append((test_score, train_score))

            # Calculate average scores
            avg_test_score = sum(fold_test_scores) / len(fold_test_scores)
            avg_train_score = sum(fold_train_scores) / len(fold_train_scores)

            # Store results
            dict_scaler_results[scaler_name] = (avg_test_score, avg_train_score)

            # Print average scores
            print(f"  Average: test score: {avg_test_score:.3f} train score: {avg_train_score:.3f}")

        # Display summary table of all scalers and their fold-by-fold performance
        print("\n=== Fold-by-Fold Results ===")
        for fold_idx in range(5):
            print(f"\nFold {fold_idx+1} Results:")
            print("-" * 80)
            print(f"{'Scaler':<15} {'Test Score':<12} {'Train Score':<12}")
            print("-" * 80)
            for scaler_name in scalers.keys():
                test_score, train_score = fold_results[scaler_name][fold_idx]
                print(f"{scaler_name:<15} {test_score:.4f}       {train_score:.4f}")

        # Print final average results
        print("\n=== Final Average Results ===")
        print("-" * 80)
        print(f"{'Scaler':<15} {'Test Score':<12} {'Train Score':<12}")
        print("-" * 80)
        for scaler_name, (avg_test_score, avg_train_score) in dict_scaler_results.items():
            print(f"{scaler_name:<15} {avg_test_score:.4f}       {avg_train_score:.4f}")

        # Track which fold we're on
        fold_counter = 0

        # Dynamically generate all possible feature pairs based on the number of features
        import itertools

        feature_pairs = list(itertools.combinations(range(num_features), 2))

        # Print information about the feature pairs
        print(f"\nGenerated {len(feature_pairs)} feature pairs for {num_features} features")

        # Get feature names
        feature_names = self.df_data.columns[:-1]  # Exclude Label column
        print("Feature pairs to be plotted:")
        for idx, (i, j) in enumerate(feature_pairs):
            print(f"Pair {idx+1}: {feature_names[i]} vs {feature_names[j]}")

        # For each fold
        for train_index, test_index in kf.split(X_features):
            fold_counter += 1
            print(f"\nFold {fold_counter} of 5:")

            X_train, X_test = X_features[train_index], X_features[test_index]

            # Create a grid: rows=feature pairs, columns=scaling methods
            num_rows = len(feature_pairs)
            num_cols = len(scalers)

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows), squeeze=False)
            fig.suptitle(f"Fold {fold_counter}: Feature Comparisons with Different Scaling Methods", fontsize=16)

            # For each feature pair (rows)
            for row_idx, (feature_idx1, feature_idx2) in enumerate(feature_pairs):
                # For each scaling method (columns)
                for col_idx, (scaler_name, scaler) in enumerate(scalers.items()):
                    # Get the current axis
                    ax = axes[row_idx, col_idx]

                    # Apply scaling
                    if scaler is not None:
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                    else:
                        X_train_scaled = X_train.copy()
                        X_test_scaled = X_test.copy()

                    # Plot training points as blue circles
                    ax.scatter(
                        X_train_scaled[:, feature_idx1],
                        X_train_scaled[:, feature_idx2],
                        c="blue",
                        alpha=0.7,
                        marker="o",
                        label="Training",
                    )

                    # Plot test points as red triangles
                    ax.scatter(
                        X_test_scaled[:, feature_idx1],
                        X_test_scaled[:, feature_idx2],
                        c="red",
                        alpha=0.7,
                        marker="^",
                        label="Test",
                        s=50,
                    )

                    # Add labels
                    ax.set_xlabel(f"{feature_names[feature_idx1]}")
                    ax.set_ylabel(f"{feature_names[feature_idx2]}")

                    # Add scaler name to top row
                    if row_idx == 0:
                        ax.set_title(scaler_name)

                    # Add feature pair label to leftmost column
                    if col_idx == 0:
                        ax.text(
                            -0.2,
                            0.5,
                            f"{feature_names[feature_idx1]} vs {feature_names[feature_idx2]}",
                            transform=ax.transAxes,
                            rotation=90,
                            verticalalignment="center",
                        )

                    # Add legend to the first subplot only
                    if row_idx == 0 and col_idx == 0:
                        ax.legend(loc="best", fontsize=8)

            plt.tight_layout()
            plt.subplots_adjust(top=0.95, left=0.1)

            # Show the plot
            plt.show()

            # Close the figure to free memory
            plt.close(fig)

        return dict_scaler_results

    def plot_pca(self, n_components=2):
        """
        Apply PCA to the dataset and plot the results.

        Parameters:
        -----------
        n_components : int, default=2
            Number of components to keep

        Returns:
        --------
        tuple: (df_scaled_features, df_pca, pca_model)
            df_scaled_features : The scaled data
            df_pca : DataFrame with PCA results
            pca_model : The fitted PCA model
        """
        # Prepare data
        X_features = self.df_data.drop("Label", axis=1).values
        y_labels = self.df_data["Label"].values

        # Apply scaling
        scaler = StandardScaler()
        X_scaled_array = scaler.fit_transform(X_features)

        # Create DataFrame for scaled data and store it as a class property
        self.df_scaled_features = pd.DataFrame(X_scaled_array, columns=self.df_data.drop("Label", axis=1).columns)

        # Display scaled data information
        print("\nスケーリング後のデータ (先頭5行):")
        print(self.df_scaled_features.head())
        print("\nスケーリング後のデータ統計情報:")
        print(self.df_scaled_features.describe())

        # Apply PCA with consistent random seed
        random_seed = 42
        pca = PCA(n_components=n_components, random_state=random_seed)
        X_pca = pca.fit_transform(X_scaled_array)

        # Display only PCA components directly
        print("components_")
        print(pca.components_)

        # Create dataframe with PCA results
        df_pca = pd.DataFrame(X_pca, columns=[f"Component {i+1}" for i in range(n_components)])
        df_pca["Label"] = y_labels

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
        feature_names = self.df_data.drop("Label", axis=1).columns
        plt.figure(figsize=(10, 5))
        components = pd.DataFrame(
            pca.components_, columns=feature_names, index=["First component", "Second component"]
        )

        sns.heatmap(components, cmap="viridis")
        plt.xlabel("Feature")
        plt.ylabel("PCA components")
        plt.tight_layout()
        plt.show()

        return self.df_scaled_features, df_pca, pca

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
        X_features = self.df_data.drop("Label", axis=1).values
        y_labels = self.df_data["Label"].values

        # Apply scaling - NMF requires non-negative values
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_features)

        # Apply NMF
        random_seed = 42
        nmf = NMF(n_components=n_components, random_state=random_seed)
        X_nmf = nmf.fit_transform(X_scaled)

        # Create dataframe with NMF results
        df_nmf = pd.DataFrame(X_nmf, columns=[f"Component {i+1}" for i in range(n_components)])
        df_nmf["Label"] = y_labels

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
        feature_names = self.df_data.drop("Label", axis=1).columns
        plt.figure(figsize=(10, 5))
        components = pd.DataFrame(
            nmf.components_, columns=feature_names, index=["First component", "Second component"]
        )

        sns.heatmap(components, cmap="viridis")
        plt.xlabel("Feature")
        plt.ylabel("NMF components")
        plt.tight_layout()
        plt.show()

        return X_scaled, df_nmf, nmf

    def plot_tsne(self, perplexity=30, random_seed=42):
        """
        Apply t-SNE to the dataset and plot the results with numeric labels.

        Parameters:
        -----------
        perplexity : float, default=30
            The perplexity parameter for t-SNE
        random_seed : int, default=42
            Random seed for reproducibility

        Returns:
        --------
        None
        """
        # Prepare data
        X_features = self.df_data.drop("Label", axis=1).values
        y_labels = self.df_data["Label"].values

        # Apply t-SNE on unscaled data
        tsne = TSNE(n_components=2, random_state=random_seed, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X_features)

        # Plot t-SNE results with numeric labels
        plt.figure(figsize=(8, 6))

        # Get all unique class labels in the dataset
        y_labels_array = np.asarray(y_labels)  # Convert to numpy array if it's not already
        unique_classes = np.unique(y_labels_array)
        num_classes = len(unique_classes)

        # Create a colormap that can handle any number of classes
        # We'll use a built-in colormap with distinct colors
        cmap = plt.cm.get_cmap("tab10", num_classes)  # 'tab10' has 10 distinct colors
        colors = [cmap(i) for i in range(num_classes)]

        # First plot the points as a scatter plot for each class
        for i, class_label in enumerate(unique_classes):
            # Create a boolean array where True indicates points belonging to the current class
            class_points_selector = y_labels == class_label

            # Plot points for this class with the corresponding color
            plt.scatter(
                X_tsne[class_points_selector, 0],  # x-coordinates of points in this class
                X_tsne[class_points_selector, 1],  # y-coordinates of points in this class
                c=[colors[i]],  # Color for this class
                alpha=0.3,  # Transparency
                s=30,  # Point size
                label=f"Class {class_label}",
            )

        # Then plot labels on top of points
        for i in range(len(X_tsne)):
            plt.text(X_tsne[i, 0], X_tsne[i, 1], str(y_labels[i]), color="black", fontsize=9, ha="center", va="center")

        plt.xlabel("t-SNE feature 0")
        plt.ylabel("t-SNE feature 1")
        plt.title("t-SNE visualization of dataset (unscaled)")
        plt.legend()
        plt.tight_layout()

        # Force display in notebook
        plt.show()

        # Provide a message to confirm the plot was generated
        print("t-SNE visualization completed.")
        print("If plot is not visible, run '%matplotlib inline' in a cell first.")

    def plot_k_means(self, n_clusters=3, random_seed=42):
        """
        Apply K-means clustering to the dataset and plot the results.

        Parameters:
        -----------
        n_clusters : int, default=3
            Number of clusters
        random_seed : int, default=42
            Random seed for reproducibility

        Returns:
        --------
        kmeans : The fitted K-means model
        """
        # Prepare data
        X_features = self.df_data.drop("Label", axis=1).values
        y_labels = self.df_data["Label"].values

        # Apply PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_features)

        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
        y_pred = kmeans.fit_predict(X_features)

        # Print cluster memberships and true labels
        print("KMeans法で予測したラベル:")
        print(y_pred)
        print("実際のラベル:")
        print(y_labels)

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
            mask = y_labels == i
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
            cluster_items = y_labels[y_pred == cluster]
            if len(cluster_items) > 0:
                # Count occurrences of each class in this cluster
                cluster_items_array = np.asarray(cluster_items)
                unique, counts = np.unique(cluster_items_array, return_counts=True)
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
        X_features = self.df_data.drop("Label", axis=1).values

        # Compute linkage
        Z = linkage(X_features, method="ward")

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
        X_features = self.df_data.drop("Label", axis=1).values

        # Apply scaling if needed
        if scaling:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_features)
        else:
            X_scaled = X_features.copy()

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        y_pred = dbscan.fit_predict(X_scaled)

        # Print cluster memberships in a formatted way
        print("Cluster Memberships:")
        cluster_output = np.array2string(y_pred, separator=" ")
        print(cluster_output)

        # Count cluster sizes
        print("\nCluster Statistics:")
        y_pred_array = np.asarray(y_pred)
        unique_clusters, counts = np.unique(y_pred_array, return_counts=True)
        for cluster, count in zip(unique_clusters, counts):
            cluster_name = "Noise" if cluster == -1 else f"Cluster {cluster}"
            print(f"{cluster_name}: {count} samples")

        # Plot results
        plt.figure(figsize=(10, 8))

        # Choose features for visualization (petal length vs petal width)
        feature_index1, feature_index2 = 2, 3

        # Plot points
        colors = ["red", "blue", "green", "purple", "orange", "cyan"]

        for cluster in np.unique(y_pred_array):
            mask = y_pred == cluster
            if cluster == -1:
                # Noise points
                c = "black"
                label = "Noise"
            else:
                c = colors[cluster % len(colors)]
                label = f"Cluster {cluster}"

            plt.scatter(X_scaled[mask, feature_index1], X_scaled[mask, feature_index2], c=c, label=label)

        plt.xlabel(f"Feature {feature_index1}")
        plt.ylabel(f"Feature {feature_index2}")
        plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
        plt.legend()
        plt.show()

        return dbscan


# Create the runner script to reproduce the output
if __name__ == "__main__":
    # Create an instance of the DataAnalyzer class
    analyzer = DataAnalyzer()

    # Call the plot_scaled_data() method with the exact output formatting
    print("# 5-foldで、それぞれの要素に対するスケーリングと、そのときのLinearSVCの結果を一覧")
    training_results = analyzer.plot_scaled_data()

# Create an alias for DataAnalyzer as AnalyzeIris for backward compatibility
AnalyzeIris = DataAnalyzer
