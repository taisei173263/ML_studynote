import matplotlib

# バックエンドをAggに設定して表示ウィンドウを出さないようにする
matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


class AnalyzeIris:
    def __init__(self, output_dir="output"):
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None
        self.model_scores = {}

        # 出力ディレクトリの作成
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _check_data_loaded(self):
        """データが読み込まれているか確認する"""
        if self.data is None or self.X is None or self.y is None:
            print("データを読み込みます...")
            self.get()

        # 念のため再チェック
        if self.data is None or self.X is None or self.y is None:
            print("データの読み込みに失敗しました。")
            return False
        return True

    def get(self):
        """irisデータセットを取得し、必要な形式に変換する"""
        try:
            iris = load_iris()
            self.feature_names = iris.feature_names
            self.target_names = iris.target_names

            # データフレームに変換
            self.data = pd.DataFrame(data=iris.data, columns=self.feature_names)
            self.data["target"] = iris.target
            self.data["species"] = self.data["target"].map(
                {0: self.target_names[0], 1: self.target_names[1], 2: self.target_names[2]}
            )

            # X, yに分割
            self.X = self.data[self.feature_names]
            self.y = self.data["target"]
            return self.data

        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return None

    def get_correlation(self):
        """変数間の相関係数を計算して表示する"""
        if not self._check_data_loaded():
            return None

        try:
            corr = self.X.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            plt.title("Feature Correlation Matrix")

            # ファイルに保存
            filepath = os.path.join(self.output_dir, "correlation_matrix.png")
            plt.savefig(filepath)
            plt.close()
            print(f"相関行列を保存しました: {filepath}")

            return corr
        except Exception as e:
            print(f"相関行列の計算中にエラーが発生しました: {e}")
            return None

    def pair_plot(self, diag_kind=None):
        """seabornを使ってpair_plotを表示する"""
        if not self._check_data_loaded():
            return None

        try:
            g = sns.pairplot(self.data, hue="species", diag_kind=diag_kind)
            plt.suptitle("Iris Pairplot", y=1.02)

            # ファイルに保存
            filepath = os.path.join(self.output_dir, "pairplot.png")
            plt.savefig(filepath)
            plt.close()
            print(f"ペアプロットを保存しました: {filepath}")

            return g
        except Exception as e:
            print(f"ペアプロットの作成中にエラーが発生しました: {e}")
            return None

    def all_supervised(self, n_neighbors=4):
        """複数の教師あり学習モデルを実行して評価する"""
        if not self._check_data_loaded():
            return None

        try:
            # 評価対象のモデル
            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "LinearSVC": LinearSVC(max_iter=1000, dual=False),
                "SVC": SVC(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=n_neighbors),
                "LinearRegression": LinearRegression(),
                "RandomForestClassifier": RandomForestClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "MLPClassifier": MLPClassifier(max_iter=1000),
            }

            # K分割交差検証
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            # 結果格納用
            self.model_scores = {}

            # 各モデルに対して評価
            for name, model in models.items():
                print(f"=== {name} ===")
                test_scores = []
                train_scores = []

                for train_idx, test_idx in kf.split(self.X):
                    X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
                    y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

                    model.fit(X_train, y_train)

                    # トレーニングスコア計算
                    if name == "LinearRegression":
                        train_score = model.score(X_train, y_train)
                        test_score = model.score(X_test, y_test)
                    else:
                        train_score = accuracy_score(y_train, model.predict(X_train))
                        test_score = accuracy_score(y_test, model.predict(X_test))

                    train_scores.append(train_score)
                    test_scores.append(test_score)

                    print(f"test score: {test_score:.3f}, train score: {train_score:.3f}")

                # スコアを保存
                self.model_scores[name] = {
                    "test_scores": test_scores,
                    "train_scores": train_scores,
                    "mean_test_score": np.mean(test_scores),
                    "mean_train_score": np.mean(train_scores),
                }

                print()

            return self.model_scores
        except Exception as e:
            print(f"モデル評価中にエラーが発生しました: {e}")
            return None

    def get_supervised(self):
        """学習結果をDataFrameで返す"""
        if not self.model_scores:
            print("モデルの評価が行われていません。先にall_supervised()を実行してください。")
            return None

        try:
            results = []
            for model_name, scores in self.model_scores.items():
                for i, (test, train) in enumerate(zip(scores["test_scores"], scores["train_scores"])):
                    results.append({"model": model_name, "fold": i + 1, "test_score": test, "train_score": train})

            df_scores = pd.DataFrame(results)
            return df_scores
        except Exception as e:
            print(f"結果の変換中にエラーが発生しました: {e}")
            return None

    def best_supervised(self):
        """最良のモデルを返す"""
        if not self.model_scores:
            print("モデルの評価が行われていません。先にall_supervised()を実行してください。")
            return None, 0

        try:
            best_model = None
            best_score = 0

            for model_name, scores in self.model_scores.items():
                mean_score = scores["mean_test_score"]
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model_name

            return best_model, best_score
        except Exception as e:
            print(f"最良モデルの選択中にエラーが発生しました: {e}")
            return None, 0

    def plot_feature_importances_all(self):
        """特徴量の重要度をプロットする"""
        if not self._check_data_loaded():
            return None

        try:
            models_with_importances = [
                "DecisionTreeClassifier",
                "RandomForestClassifier",
                "GradientBoostingClassifier",
            ]

            if self.feature_names is None or len(self.feature_names) == 0:
                print("特徴量名が設定されていません。")
                return None

            plt.figure(figsize=(15, 10))
            plot_count = 0

            for i, model_name in enumerate(models_with_importances):
                # モデルを再学習
                if model_name == "DecisionTreeClassifier":
                    model = DecisionTreeClassifier()
                elif model_name == "RandomForestClassifier":
                    model = RandomForestClassifier()
                elif model_name == "GradientBoostingClassifier":
                    model = GradientBoostingClassifier()
                else:
                    continue

                model.fit(self.X, self.y)
                plot_count += 1

                # 重要度をプロット
                plt.subplot(1, 3, plot_count)
                importance = model.feature_importances_
                indices = np.argsort(importance)[::-1]

                feature_count = len(self.feature_names)
                plt.bar(range(feature_count), importance[indices])
                plt.xticks(range(feature_count), [self.feature_names[j] for j in indices], rotation=90)
                plt.title(f"{model_name} Feature Importance")

            plt.tight_layout()
            plt.savefig()
            plt.close()
        except Exception as e:
            print(f"特徴量重要度のプロット中にエラーが発生しました: {e}")
            return None

    def visualize_decision_tree(self):
        """決定木を可視化する"""
        if not self._check_data_loaded():
            return None

        try:
            tree = DecisionTreeClassifier()
            tree.fit(self.X, self.y)

            plt.figure(figsize=(15, 10))
            plot_tree(tree, filled=True, feature_names=self.feature_names, class_names=self.target_names)
            plt.title("Decision Tree Visualization")
            plt.savefig()
            plt.close()

            return tree
        except Exception as e:
            print(f"決定木の可視化中にエラーが発生しました: {e}")
            return None

    def plot_scaled_data(self):
        """異なるスケーリング手法でデータを変換し、LinearSVCの結果を評価する"""
        if not self._check_data_loaded():
            return None

        try:
            scalers = {
                "Original": None,
                "MinMaxScaler": MinMaxScaler(),
                "StandardScaler": StandardScaler(),
                "RobusScaler": RobustScaler(),
                "Normalizer": Normalizer(),
            }

            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for train_idx, test_idx in kf.split(self.X):
                X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

                print("=" * 73)

                for name, scaler in scalers.items():
                    if scaler:
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                    else:
                        X_train_scaled = X_train
                        X_test_scaled = X_test

                    model = LinearSVC(max_iter=1000, dual=False)
                    model.fit(X_train_scaled, y_train)

                    train_score = accuracy_score(y_train, model.predict(X_train_scaled))
                    test_score = accuracy_score(y_test, model.predict(X_test_scaled))

                    print(f"{name:<15}: test score: {test_score:.3f}      train score: {train_score:.3f}     ")

            print("=" * 73)

            # 最初のスケーリングデータを返す (標準スケーリング)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.X)
            return pd.DataFrame(X_scaled, columns=self.feature_names)
        except Exception as e:
            print(f"スケーリングデータのプロット中にエラーが発生しました: {e}")
            return None

    def plot_pca(self, n_components=2):
        """PCA分析を行い結果をプロットする"""
        if not self._check_data_loaded():
            return None

        try:
            # データのスケーリング
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.X)

            # PCA実行
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)

            # 結果をデータフレーム化
            df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
            df_pca["species"] = self.data["species"]

            # プロット
            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="species", palette="viridis", s=100)
            plt.title("PCA of Iris Dataset")

            # 主成分の寄与率
            explained_variance = pca.explained_variance_ratio_
            plt.xlabel(f"PC1 ({explained_variance[0]:.2f})")
            plt.ylabel(f"PC2 ({explained_variance[1]:.2f})")

            # ファイルに保存
            filepath = os.path.join(self.output_dir, "pca_plot.png")
            plt.savefig(filepath)
            plt.close()
            print(f"PCA分析結果を保存しました: {filepath}")

            return pd.DataFrame(X_scaled, columns=self.feature_names), df_pca, pca
        except Exception as e:
            print(f"PCA分析中にエラーが発生しました: {e}")
            return None, None, None

    def plot_nmf(self, n_components=2):
        """NMF分析を行い結果をプロットする"""
        if not self._check_data_loaded():
            return None

        try:
            # データのスケーリング (負の値は使えないのでMinMaxScalerを使用)
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(self.X)

            # NMF実行
            nmf = NMF(n_components=n_components, random_state=42)
            X_nmf = nmf.fit_transform(X_scaled)

            # 結果をデータフレーム化
            df_nmf = pd.DataFrame(X_nmf, columns=[f"NMF{i+1}" for i in range(n_components)])
            df_nmf["species"] = self.data["species"]

            # プロット
            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=df_nmf, x="NMF1", y="NMF2", hue="species", palette="viridis", s=100)
            plt.title("NMF of Iris Dataset")
            plt.savefig()
            plt.close()

            return pd.DataFrame(X_scaled, columns=self.feature_names), df_nmf, nmf
        except Exception as e:
            print(f"NMF分析中にエラーが発生しました: {e}")
            return None, None, None

    def plot_tsne(self):
        """t-SNE分析を行い結果をプロットする"""
        if not self._check_data_loaded():
            return None

        try:
            # t-SNE実行 (スケールしていない元データを使用)
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(self.X)

            # 結果をデータフレーム化
            df_tsne = pd.DataFrame(X_tsne, columns=["t-SNE1", "t-SNE2"])
            df_tsne["species"] = self.data["species"]

            # プロット
            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=df_tsne, x="t-SNE1", y="t-SNE2", hue="species", palette="viridis", s=100)
            plt.title("t-SNE of Iris Dataset")
            plt.savefig()
            plt.close()

            return df_tsne
        except Exception as e:
            print(f"t-SNE分析中にエラーが発生しました: {e}")
            return None

    def plot_k_means(self):
        """K-means分析を行い結果をプロットする"""
        if not self._check_data_loaded():
            return None

        try:
            # K-means実行
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(self.X)

            # 結果をデータフレーム化
            df_kmeans = self.X.copy()
            df_kmeans["cluster"] = clusters
            df_kmeans["actual"] = self.y

            # プロット
            plt.figure(figsize=(15, 5))

            # K-means結果
            plt.subplot(1, 2, 1)
            sns.scatterplot(x=self.X.iloc[:, 0], y=self.X.iloc[:, 1], hue=clusters, palette="viridis", s=100)
            plt.title("K-means Clustering")
            plt.xlabel(self.feature_names[0])
            plt.ylabel(self.feature_names[1])

            # 実際のラベル
            plt.subplot(1, 2, 2)
            sns.scatterplot(x=self.X.iloc[:, 0], y=self.X.iloc[:, 1], hue=self.y, palette="viridis", s=100)
            plt.title("Actual Classes")
            plt.xlabel(self.feature_names[0])
            plt.ylabel(self.feature_names[1])

            plt.tight_layout()

            # ファイルに保存
            filepath = os.path.join(self.output_dir, "kmeans_plot.png")
            plt.savefig(filepath)
            plt.close()
            print(f"K-means分析結果を保存しました: {filepath}")

            print("KMeans法で予測したラベル:")
            print(clusters)
            print("\n実際のラベル:")
            print(self.y.values)

            return df_kmeans
        except Exception as e:
            print(f"K-means分析中にエラーが発生しました: {e}")
            return None

    def plot_dendrogram(self, truncate=False):
        """階層的クラスタリングのデンドログラムをプロットする"""
        if not self._check_data_loaded() or self.y is None:
            return None

        try:
            # リンケージ行列を計算
            linked = linkage(self.X, "ward")

            # デンドログラムをプロット
            plt.figure(figsize=(12, 8))
            dendrogram(
                linked,
                truncate_mode="lastp" if truncate else None,
                p=10 if truncate else None,
                leaf_font_size=10.0,
                orientation="top",
                labels=self.y.values if len(self.y) <= 30 else None,
            )
            plt.title("Hierarchical Clustering Dendrogram")
            plt.xlabel("Sample index")
            plt.ylabel("Distance")
            plt.savefig()
            plt.close()

            return linked
        except Exception as e:
            print(f"デンドログラムの作成中にエラーが発生しました: {e}")
            return None

    def plot_dbscan(self, scaling=False, eps=0.5, min_samples=5):
        """DBSCAN分析を行い結果をプロットする"""
        if not self._check_data_loaded():
            return None

        try:
            # データのスケーリング (オプション)
            if scaling:
                scaler = StandardScaler()
                X_dbscan = scaler.fit_transform(self.X)
            else:
                X_dbscan = self.X.values

            # DBSCAN実行
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X_dbscan)

            # 結果をデータフレーム化
            df_dbscan = self.X.copy()
            df_dbscan["cluster"] = clusters

            # プロット
            plt.figure(figsize=(12, 10))

            # プロットのためのカラーマップ (-1はノイズ点で黒にする)
            cmap = plt.cm.viridis
            cmaplist = [cmap(i) for i in range(cmap.N)]
            cmaplist[0] = (0, 0, 0, 1.0)  # ノイズ点を黒に
            cmap_custom = plt.matplotlib.colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)

            # 散布図行列
            sns.pairplot(
                df_dbscan, hue="cluster", palette=cmap_custom, plot_kws={"alpha": 0.8, "s": 80}, diag_kind="kde"
            )
            plt.suptitle("DBSCAN Clustering", y=1.02, fontsize=16)
            plt.tight_layout()
            plt.savefig()
            plt.close()

            print("Cluster Memberships:", clusters)

            return df_dbscan
        except Exception as e:
            print(f"DBSCAN分析中にエラーが発生しました: {e}")
            return None


# サンプル使用例
if __name__ == "__main__":
    # インスタンスを作成
    analyzer = AnalyzeIris(output_dir="iris_analysis_outputs")

    # データセットを読み込み
    data = analyzer.get()
    print("データを読み込みました。\n")

    # 基本的な分析の例
    print("相関行列を分析します:")
    analyzer.get_correlation()

    print("\nペアプロットを作成します:")
    analyzer.pair_plot(diag_kind="kde")

    print("\nPCA分析を実行します:")
    analyzer.plot_pca()

    print("\nクラスタリング分析を実行します:")
    analyzer.plot_k_means()

    print("\n完了しました。すべての分析結果は以下のディレクトリに保存されています:")
    print(os.path.abspath(analyzer.output_dir))
