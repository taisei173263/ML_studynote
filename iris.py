# Save the following code as 'analyze_iris.py'

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# すべてのprint出力を省略せずに表示
from IPython.core.interactiveshell import InteractiveShell
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


class DataAnalyzer:
    """
    A class for analyzing tabular datasets with various machine learning techniques.
    This class provides methods for data visualization, preprocessing, and modeling.
    """

    # Class-level random seed for reproducibility
    RANDOM_SEED = 42

    def __init__(self):
        """Initialize the class with the dataset."""
        iris = load_iris()
        self.df_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        self.df_data["Label"] = iris.target  # （アヤメの種類を示す 0, 1, 2 のラベル）
        self.target_names = iris.target_names
        self.df_results_supervised = None  # 後で教師あり学習モデルの結果を格納するため
        self.df_scaled_features = pd.DataFrame(columns=self.df_data.drop("Label", axis=1).columns)
        # スケーリングされたデータを格納するためのデータフレームを作成

    def get(self):
        """Return the dataset."""
        return self.df_data

    def get_correlation(self):
        """Calculate and return correlation matrix of features."""
        return self.df_data.drop("Label", axis=1).corr()  #

    def pair_plot(self, diag_kind="hist"):
        """
        Create a pairplot of the dataset.

        Parameters:
        -----------
        diag_kind : str, default="hist"
            The kind of plot to use on the diagonal. Options: 'hist', 'kde'
        """
        valid_diag_kinds = ["auto", "hist", "kde", None]
        if diag_kind not in valid_diag_kinds:
            print(f"Warning: diag_kind '{diag_kind}' not in {valid_diag_kinds}. Using 'hist' instead.")
            diag_kind = "hist"

        df_plot = self.df_data.copy()
        df_plot["LabelName"] = df_plot["Label"].map({i: name for i, name in enumerate(self.target_names)})
        feature_cols = df_plot.columns[:-2]  # (アヤメの種類を示す 0, 1, 2 のラベルを除く)

        _ = sns.pairplot(df_plot, vars=feature_cols, hue="LabelName", diag_kind=diag_kind)
        plt.show()
        plt.close()

    def all_supervised(self, n_neighbors=4):  # 分類時に考慮する「近傍点の数」を指定
        """
        Run multiple supervised learning algorithms on the dataset using k-fold
        cross-validation and display results.

        Parameters:
        -----------
        n_neighbors : int, default=4
            Number of neighbors for KNeighborsClassifier

        Returns:
        --------
        dict : Dictionary with model names as keys and average test scores as values
        """
        X_features = self.df_data.drop("Label", axis=1).values
        y_labels = self.df_data["Label"].values

        classifiers = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=self.RANDOM_SEED),
            "LinearSVC": LinearSVC(max_iter=1000, random_state=self.RANDOM_SEED),
            "SVC": SVC(random_state=self.RANDOM_SEED),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=self.RANDOM_SEED),
            "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=n_neighbors),
            "LinearRegression": LinearRegression(),
            "RandomForestClassifier": RandomForestClassifier(random_state=self.RANDOM_SEED),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=self.RANDOM_SEED),
            "MLPClassifier": MLPClassifier(max_iter=1000, random_state=self.RANDOM_SEED),
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=self.RANDOM_SEED)
        dict_results = {}
        df_fold_results = pd.DataFrame(index=range(5))

        for model_name, classifier in classifiers.items():
            print(f"=== {model_name} ===")
            fold_scores = []

            for fold_index, (index_train, index_test) in enumerate(kf.split(X_features)):
                X_train, X_test = X_features[index_train], X_features[index_test]
                y_train, y_test = y_labels[index_train], y_labels[index_test]

                classifier.fit(X_train, y_train)
                train_score = classifier.score(X_train, y_train)
                test_score = classifier.score(X_test, y_test)

                print(f"test score: {test_score:.3f}, train score: {train_score:.3f}")
                fold_scores.append(test_score)

            dict_results[model_name] = np.mean(fold_scores)
            df_fold_results[model_name] = fold_scores

        self.df_results_supervised = df_fold_results

        # Calculate best method and score directly here to avoid calling best_supervised
        # which would cause recursion
        results_mean = df_fold_results.mean()
        best_method = results_mean.idxmax()
        best_score = results_mean.max()
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

        # Calculate best method from results
        if self.df_results_supervised is not None and not self.df_results_supervised.empty:
            results_mean = self.df_results_supervised.mean()
            best_method = results_mean.idxmax()
            best_score = results_mean.max()
            return best_method, best_score
        else:
            return None, 0.0  # 最良の手法が見つからなかった（エラーケース）,スコアも同様にエラーケースとして0.0を返す

    def plot_feature_importances_all(self):  # 決定木モデルの特徴量重要性をプロット
        """
        Plot feature importances for tree-based models.

        Returns:
        --------
        None: Displays feature importance plots for tree-based models
        """
        X_features = self.df_data.drop("Label", axis=1).values
        y_labels = self.df_data["Label"].values
        feature_names = list(self.df_data.drop("Label", axis=1).columns)  # 特徴量の名前を取得

        tree_models = {
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=self.RANDOM_SEED),
            "RandomForestClassifier": RandomForestClassifier(random_state=self.RANDOM_SEED),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=self.RANDOM_SEED),
        }

        for model_name, model in tree_models.items():
            model.fit(X_features, y_labels)
            plt.figure(figsize=(8, 3))
            plt.barh(np.arange(len(feature_names)), model.feature_importances_)
            plt.yticks(np.arange(len(feature_names)), feature_names)
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
        X_features = self.df_data.drop("Label", axis=1).values  # 特徴量の値を取得
        y_labels = self.df_data["Label"].values.astype(int)  # ラベルの値を取得

        decision_tree = DecisionTreeClassifier(random_state=self.RANDOM_SEED)
        decision_tree.fit(X_features, y_labels)

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
        # 基本データの準備
        X_features = self.df_data.drop("Label", axis=1).values
        y_labels = self.df_data["Label"].values
        num_features = X_features.shape[1]
        feature_names = self.df_data.columns[:-1]

        # スケーラーの定義
        scalers = {
            "Original": None,
            "MinMaxScaler": MinMaxScaler(),
            "StandardScaler": StandardScaler(),
            "RobustScaler": RobustScaler(),
            "Normalizer": Normalizer(),
        }

        # 5分割交差検証の設定
        kf = KFold(n_splits=5, shuffle=True, random_state=self.RANDOM_SEED)
        dict_scaler_results = {}
        fold_results = {scaler_name: [] for scaler_name in scalers.keys()}

        # 各スケーラーでの評価
        for scaler_name, scaler in scalers.items():
            fold_test_scores = []
            fold_train_scores = []
            print(f"\n=== {scaler_name} ===")

            # 各フォールドでの評価
            for fold_idx, (index_train, index_test) in enumerate(kf.split(X_features)):
                # データ分割
                X_train, X_test = X_features[index_train], X_features[index_test]
                y_train, y_test = y_labels[index_train], y_labels[index_test]

                # スケーリング
                X_train_scaled = scaler.fit_transform(X_train) if scaler else X_train.copy()
                X_test_scaled = scaler.transform(X_test) if scaler else X_test.copy()

                # 分類器の学習と評価
                classifier = LinearSVC(max_iter=1000, random_state=self.RANDOM_SEED)
                classifier.fit(X_train_scaled, y_train)
                train_score = classifier.score(X_train_scaled, y_train)
                test_score = classifier.score(X_test_scaled, y_test)

                # 結果の記録
                print(f"  Fold {fold_idx+1}: test score: {test_score:.3f}, train score: {train_score:.3f}")
                fold_test_scores.append(test_score)
                fold_train_scores.append(train_score)
                fold_results[scaler_name].append((test_score, train_score))

            # 平均スコアの計算と表示
            avg_test_score = np.mean(fold_test_scores)
            avg_train_score = np.mean(fold_train_scores)
            dict_scaler_results[scaler_name] = (avg_test_score, avg_train_score)
            print(f"  Average: test score: {avg_test_score:.3f} train score: {avg_train_score:.3f}")

        # フォールド別結果の表示
        self._print_fold_results(fold_results, scalers)

        # 特徴量ペアの生成
        feature_pairs = list(itertools.combinations(range(num_features), 2))
        print(f"\nGenerated {len(feature_pairs)} feature pairs for {num_features} features")

        # 特徴量ペアのリスト表示
        print("Feature pairs to be plotted:")
        for idx, (i, j) in enumerate(feature_pairs):
            print(f"Pair {idx+1}: {feature_names[i]} vs {feature_names[j]}")

        # 各フォールドでの可視化
        self._visualize_folds(kf, X_features, feature_pairs, scalers, feature_names)

        return dict_scaler_results

    def _print_fold_results(self, fold_results, scalers):  # フォールド別結果と最終平均結果を表示するヘルパーメソッド
        """フォールド別結果と最終平均結果を表示するヘルパーメソッド"""
        # フォールド別結果
        print("\n=== Fold-by-Fold Results ===")
        for fold_idx in range(5):
            print(f"\nFold {fold_idx+1} Results:")
            print("-" * 80)
            print(f"{'Scaler':<15} {'Test Score':<12} {'Train Score':<12}")
            print("-" * 80)
            for scaler_name in scalers.keys():
                test_score, train_score = fold_results[scaler_name][fold_idx]
                print(f"{scaler_name:<15} {test_score:.4f}       {train_score:.4f}")

        # 最終平均結果
        print("\n=== Final Average Results ===")
        print("-" * 80)
        print(f"{'Scaler':<15} {'Test Score':<12} {'Train Score':<12}")
        print("-" * 80)

        # 平均スコアの計算と表示
        for scaler_name in scalers.keys():
            scores = fold_results[scaler_name]
            avg_test = np.mean([s[0] for s in scores])
            avg_train = np.mean([s[1] for s in scores])
            print(f"{scaler_name:<15} {avg_test:.4f}       {avg_train:.4f}")

    def _visualize_folds(self, kf, X_features, feature_pairs, scalers, feature_names):
        """各フォールドでの特徴量ペアの可視化を行うヘルパーメソッド"""
        # 各フォールドの特徴量ペア散布図
        for fold_idx, (index_train, index_test) in enumerate(kf.split(X_features)):
            print(f"\nFold {fold_idx+1} of 5:")
            X_train, X_test = X_features[index_train], X_features[index_test]

            # グリッド作成
            num_rows = len(feature_pairs)
            num_cols = len(scalers)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows), squeeze=False)
            fig.suptitle(f"Fold {fold_idx+1}: Feature Comparisons with Different Scaling Methods", fontsize=16)

            # 特徴量ペアごとの散布図
            for row_idx, (feature_idx1, feature_idx2) in enumerate(feature_pairs):
                for col_idx, (scaler_name, scaler) in enumerate(scalers.items()):
                    ax = axes[row_idx, col_idx]

                    # スケーリング
                    X_train_scaled = scaler.fit_transform(X_train) if scaler else X_train.copy()
                    X_test_scaled = scaler.transform(X_test) if scaler else X_test.copy()

                    # 散布図作成
                    ax.scatter(
                        X_train_scaled[:, feature_idx1],
                        X_train_scaled[:, feature_idx2],
                        c="blue",
                        alpha=0.7,
                        marker="o",
                        label="Training",
                    )
                    ax.scatter(
                        X_test_scaled[:, feature_idx1],
                        X_test_scaled[:, feature_idx2],
                        c="red",
                        alpha=0.7,
                        marker="^",
                        label="Test",
                        s=50,
                    )

                    # ラベルとタイトル設定
                    ax.set_xlabel(f"{feature_names[feature_idx1]}")
                    ax.set_ylabel(f"{feature_names[feature_idx2]}")

                    if row_idx == 0:
                        ax.set_title(scaler_name)
                    if col_idx == 0:
                        ax.text(
                            -0.2,
                            0.5,
                            f"{feature_names[feature_idx1]} vs {feature_names[feature_idx2]}",
                            transform=ax.transAxes,
                            rotation=90,
                            verticalalignment="center",
                        )
                    if row_idx == 0 and col_idx == 0:
                        ax.legend(loc="best", fontsize=8)

            plt.tight_layout()
            plt.subplots_adjust(top=0.95, left=0.1)
            plt.show()
            plt.close(fig)

    def _plot_dimension_reduction(
        self, model, n_components, scaler, method_name
    ):  # 次元削減を適用して結果をプロットするヘルパーメソッド
        """
        Apply dimension reduction to the dataset and plot the results.

        Parameters:
        -----------
        model : object
            The dimension reduction model (PCA or NMF)
        n_components : int
            Number of components to keep
        scaler : object
            The scaler to use (StandardScaler or MinMaxScaler)
        method_name : str
            The name of the method (PCA or NMF)

        Returns:
        --------
        tuple: (X_scaled, df_result, model)
            X_scaled : The scaled data
            df_result : DataFrame with dimension reduction results
            model : The fitted model
        """
        X_features = self.df_data.drop("Label", axis=1).values
        y_labels = self.df_data["Label"].values
        X_scaled = scaler.fit_transform(X_features)

        # random_stateの設定（共通処理）
        if hasattr(model, "random_state"):
            model.random_state = self.RANDOM_SEED

        # モデルの適用（共通処理）
        X_result = model.fit_transform(X_scaled)

        # PCA特有の処理
        if method_name == "PCA":
            self.df_scaled_features = pd.DataFrame(X_scaled, columns=self.df_data.drop("Label", axis=1).columns)
            print("\nスケーリング後のデータ (先頭5行):")
            print(self.df_scaled_features.head())
            print("\nスケーリング後のデータ統計情報:")
            print(self.df_scaled_features.describe())
            print("components_")
            print(model.components_)

        # 以下、共通の可視化処理
        df_result = pd.DataFrame(X_result, columns=[f"Component {i+1}" for i in range(n_components)])
        df_result["Label"] = y_labels

        plt.figure(figsize=(10, 8))
        colors = ["blue", "orange", "green"]
        markers = ["o", "^", "v"]

        for i, target_name in enumerate(self.target_names):
            mask = df_result["Label"] == i
            plt.scatter(
                df_result.loc[mask, "Component 1"],
                df_result.loc[mask, "Component 2"],
                c=colors[i],
                marker=markers[i],
                label=target_name,
            )

        plt.xlabel("First component")
        plt.ylabel("Second component")
        plt.legend()
        plt.show()

        feature_names = self.df_data.drop("Label", axis=1).columns
        plt.figure(figsize=(10, 5))
        components = pd.DataFrame(
            model.components_, columns=feature_names, index=["First component", "Second component"]
        )

        sns.heatmap(components, cmap="viridis")
        plt.xlabel("Feature")
        plt.ylabel(f"{method_name} components")
        plt.tight_layout()
        plt.show()

        return (self.df_scaled_features, df_result, model) if method_name == "PCA" else (X_scaled, df_result, model)

    def plot_pca(self, n_components=2):  # PCAを適用して結果をプロットするメソッド
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
        return self._plot_dimension_reduction(
            PCA(n_components=n_components), n_components, StandardScaler(), "PCA"
        )  # PCAを適用して結果をプロットするメソッド

    def plot_nmf(self, n_components=2):  # NMFを適用して結果をプロットするメソッド
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
        return self._plot_dimension_reduction(NMF(n_components=n_components), n_components, MinMaxScaler(), "NMF")

    def plot_tsne(self, perplexity=30):  # t-SNEを適用して結果をプロットするメソッド
        """
        Apply t-SNE to the dataset and plot the results with numeric labels.

        Parameters:
        -----------
        perplexity : float, default=30
            The perplexity parameter for t-SNE

        Returns:
        --------
        None
        """
        X_features = self.df_data.drop("Label", axis=1).values
        y_labels = self.df_data["Label"].values
        X_tsne = TSNE(n_components=2, random_state=self.RANDOM_SEED, perplexity=perplexity).fit_transform(X_features)

        plt.figure(figsize=(8, 6))
        unique_classes = np.unique(y_labels)
        cmap = plt.cm.get_cmap("tab10", len(unique_classes))
        colors = [cmap(i) for i in range(len(unique_classes))]

        for i, class_label in enumerate(unique_classes):
            plt.scatter(
                X_tsne[y_labels == class_label, 0],
                X_tsne[y_labels == class_label, 1],
                c=[colors[i]],
                alpha=0.3,
                s=30,
                label=f"Class {class_label}",
            )

        for i in range(len(X_tsne)):
            plt.text(X_tsne[i, 0], X_tsne[i, 1], str(y_labels[i]), color="black", fontsize=9, ha="center", va="center")

        plt.xlabel("t-SNE feature 0")
        plt.ylabel("t-SNE feature 1")
        plt.title("t-SNE visualization of dataset (unscaled)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        print("t-SNE visualization completed.")
        print("If plot is not visible, run '%matplotlib inline' in a cell first.")

    def plot_k_means(self, n_clusters=3):  # K-meansクラスタリングを適用して結果をプロットするメソッド
        """
        Apply K-means clustering to the dataset and plot the results.

        Parameters:
        -----------
        n_clusters : int, default=3
            Number of clusters

        Returns:
        --------
        kmeans : The fitted K-means model
        """
        X_features = self.df_data.drop("Label", axis=1).values
        y_labels = self.df_data["Label"].values
        X_pca = PCA(n_components=2).fit_transform(X_features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.RANDOM_SEED)
        y_pred = kmeans.fit_predict(X_features)

        print("KMeans法で予測したラベル:")
        print(y_pred)
        print("実際のラベル:")
        print(y_labels)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        colors = ["blue", "orange", "green"]

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

        centers_pca = PCA(n_components=2).fit(X_features).transform(kmeans.cluster_centers_)
        ax1.scatter(centers_pca[:, 0], centers_pca[:, 1], marker="*", s=300, c="red", label="Centroids")
        ax1.set_title("K-means Clustering Result")
        ax1.set_xlabel("First Principal Component")
        ax1.set_ylabel("Second Principal Component")
        ax1.legend()

        class_centers = []
        for i, target_name in enumerate(self.target_names):
            mask = y_labels == i
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], marker="o", s=50, alpha=0.7, label=target_name)
            class_centers.append(np.mean(X_pca[mask], axis=0))

        class_centers = np.array(class_centers)
        ax2.scatter(class_centers[:, 0], class_centers[:, 1], marker="*", s=300, c="red", label="Class Centers")
        ax2.set_title("Actual Classes")
        ax2.set_xlabel("First Principal Component")
        ax2.set_ylabel("Second Principal Component")
        ax2.legend()

        plt.tight_layout()
        plt.show()

        print("\nCluster to Class mapping:")
        for cluster in range(n_clusters):
            cluster_items = y_labels[y_pred == cluster]
            if len(cluster_items) > 0:
                unique, counts = np.unique(cluster_items, return_counts=True)
                print(f"Cluster {cluster} contains: ", end="")
                for u, c in zip(unique, counts):
                    print(f"{self.target_names[u]}: {c} items, ", end="")
                print()

        return kmeans

    def plot_dendrogram(self, truncate=False):  # 階層型クラスタリングを適用して結果をプロットするメソッド
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
        X_features = self.df_data.drop("Label", axis=1).values
        Z = linkage(X_features, method="ward")

        plt.figure(figsize=(12, 8))
        dendrogram(Z, truncate_mode="lastp", p=10) if truncate else dendrogram(Z)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample index")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.show()

    def plot_dbscan(self, eps=0.5, min_samples=5, scaling=False):  # DBSCANを適用して結果をプロットするメソッド
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
        X_features = self.df_data.drop("Label", axis=1).values
        X_scaled = StandardScaler().fit_transform(X_features) if scaling else X_features.copy()

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        y_pred = dbscan.fit_predict(X_scaled)

        print("Cluster Memberships:")
        print(np.array2string(y_pred, separator=" "))

        print("\nCluster Statistics:")
        unique_clusters, counts = np.unique(y_pred, return_counts=True)
        for cluster, count in zip(unique_clusters, counts):
            cluster_name = "Noise" if cluster == -1 else f"Cluster {cluster}"  # -1はノイズクラスターを表す
            print(f"{cluster_name}: {count} samples")

        plt.figure(figsize=(10, 8))
        colors = ["red", "blue", "green", "purple", "orange", "cyan"]
        feature_index1, feature_index2 = 2, 3

        for cluster in np.unique(y_pred):  # ユニークなクラスターを取得
            mask = y_pred == cluster
            c = "black" if cluster == -1 else colors[cluster % len(colors)]
            label = "Noise" if cluster == -1 else f"Cluster {cluster}"
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


# Create an alias for DataAnalyzer as AnalyzeIris for backward compatibility
AnalyzeIris = DataAnalyzer
