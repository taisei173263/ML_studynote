import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # 非インタラクティブなバックエンドを使用

import matplotlib.pyplot as plt

plt.close("all")  # 全ての図を明示的に閉じる
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

# グラフ出力用のディレクトリを作成
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class AnalyzeIris:
    def __init__(self, output_dir="output"):
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None
        self.model_scores = {}
        self.output_dir = output_dir

        # 出力ディレクトリの作成
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
            # データフレームの作成
            df = pd.DataFrame(data=self.X)
            df.columns = self.feature_names
            df["species"] = [self.target_names[i] for i in self.y]

            if diag_kind == "hist" or diag_kind == "kde":
                # 指定された対角成分のみ表示
                g = sns.pairplot(df, hue="species", diag_kind=diag_kind)
                g.fig.suptitle(f"Pair Plot with {diag_kind.upper()} on Diagonal", y=1.02)
                filepath = os.path.join(self.output_dir, f"pairplot_{diag_kind}.png")
                plt.savefig(filepath)
                plt.close()
                print(f"ペアプロット({diag_kind})を保存しました: {filepath}")
            else:
                # ヒストグラムバージョン
                g1 = sns.pairplot(df, hue="species", diag_kind="hist")
                g1.fig.suptitle("Pair Plot with Histograms on Diagonal", y=1.02)
                filepath1 = os.path.join(self.output_dir, "pairplot_hist.png")
                plt.savefig(filepath1)
                plt.close()
                print(f"ペアプロット(hist)を保存しました: {filepath1}")

                # KDEバージョン
                g2 = sns.pairplot(df, hue="species", diag_kind="kde")
                g2.fig.suptitle("Pair Plot with KDE on Diagonal", y=1.02)
                filepath2 = os.path.join(self.output_dir, "pairplot_kde.png")
                plt.savefig(filepath2)
                plt.close()
                print(f"ペアプロット(kde)を保存しました: {filepath2}")

            return True
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
                "LinearSVC": LinearSVC(max_iter=1000, dual="auto"),
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
            return None, None

        try:
            # DataFrameの表示オプションを設定
            pd.set_option("display.max_columns", None)  # 列の省略をなくす
            pd.set_option("display.width", None)  # 表示幅の制限をなくす
            pd.set_option("display.max_colwidth", None)  # 列の幅の制限をなくす
            pd.set_option("display.expand_frame_repr", False)  # 折り返しをなくす
            pd.set_option("display.float_format", lambda x: "%.3f" % x)  # 小数点以下3桁に固定
            pd.set_option("display.max_rows", None)  # 行の省略をなくす
            pd.set_option("display.precision", 3)  # 全体の精度を3桁に設定

            # 各モデルのテストスコアを抽出
            test_scores = {}
            for model_name, scores in self.model_scores.items():
                test_scores[model_name] = scores["test_scores"]

            # DataFrameに変換
            df_scores = pd.DataFrame(test_scores)

            # 記述統計を計算
            stats = df_scores.describe()

            # 列名を短く省略
            rename_dict = {
                "LogisticRegression": "LogReg",
                "LinearSVC": "LSVC",
                "DecisionTreeClassifier": "DTree",
                "KNeighborsClassifier": "KNN",
                "LinearRegression": "LinReg",
                "RandomForestClassifier": "RF",
                "GradientBoostingClassifier": "GBoost",
                "MLPClassifier": "MLP",
            }
            df_scores = df_scores.rename(columns=rename_dict)
            stats = stats.rename(columns=rename_dict)

            return df_scores, stats
        except Exception as e:
            print(f"結果の変換中にエラーが発生しました: {e}")
            return None, None

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
        """特徴量の重要度を横棒グラフで表示する"""
        if not self._check_data_loaded():
            return None

        try:
            models_with_importances = [
                "DecisionTreeClassifier",
                "RandomForestClassifier",
                "GradientBoostingClassifier",
            ]

            # 図を作成
            fig = plt.figure(figsize=(15, 10))

            # 表示するモデルの数をカウント
            available_models = 0

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
                available_models += 1

                # 重要度を取得
                importance = model.feature_importances_

                # インデックスでソート（降順）
                sorted_indices = np.argsort(importance)[::-1]

                # サブプロット作成
                ax = fig.add_subplot(1, len(models_with_importances), i + 1)

                # 横棒グラフの作成
                y_pos = np.arange(len(self.feature_names))
                ax.barh(y_pos, importance[sorted_indices], align="center")
                ax.set_yticks(y_pos)
                ax.set_yticklabels([self.feature_names[idx] for idx in sorted_indices])
                ax.invert_yaxis()  # 上から下へ値が大きい順に表示
                ax.set_xlabel("Feature Importance (0-1.0)")
                ax.set_xlim(0, 1.0)  # X軸の範囲を0～1に設定
                ax.set_title(f"{model_name}")

            if available_models == 0:
                print("特徴量の重要度を持つモデルが見つかりません。")
                return None

            # 全モデルをまとめて1つの図として保存
            plt.tight_layout()  # レイアウトを調整
            filepath = os.path.join(self.output_dir, "feature_importance_all_models.png")
            plt.savefig(filepath)
            plt.close()
            print(f"特徴量重要度を保存しました: {filepath}")

            return True
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
            plt.tight_layout()

            # ファイルに保存
            filepath = os.path.join(self.output_dir, "decision_tree.png")
            plt.savefig(filepath)
            plt.close()
            print(f"決定木を保存しました: {filepath}")

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
                "RobustScaler": RobustScaler(),
                "Normalizer": Normalizer(),
            }

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            results = []

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.X)):
                X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

                print("=" * 73)

                fold_results = []  # 現在のフォールドの結果を保存

                for name, scaler in scalers.items():
                    if scaler:
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                    else:
                        X_train_scaled = X_train
                        X_test_scaled = X_test

                    model = LinearSVC(max_iter=1000, dual="auto")
                    model.fit(X_train_scaled, y_train)

                    train_score = accuracy_score(y_train, model.predict(X_train_scaled))
                    test_score = accuracy_score(y_test, model.predict(X_test_scaled))

                    # 結果を保存
                    result = {
                        "fold": fold_idx + 1,
                        "scaler": name,
                        "train_score": train_score,
                        "test_score": test_score,
                    }
                    results.append(result)
                    fold_results.append(result)

                    # 結果を表示
                    print(f"{name:<14}: test score: {test_score:.3f}      train score: {train_score:.3f}")

                # 散布図の作成と保存
                plt.figure(figsize=(8, 6))

                # 各スケーラーの結果をプロット
                for result in fold_results:
                    plt.scatter(result["train_score"], result["test_score"], label=result["scaler"], s=100)

                # 理想的な性能を示す対角線を追加
                plt.plot([0.8, 1.0], [0.8, 1.0], "k--", alpha=0.3)

                plt.xlabel("Training Score")
                plt.ylabel("Test Score")
                plt.title(f"Scaling Methods Comparison (Fold {fold_idx + 1})")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.xlim(0.8, 1.0)
                plt.ylim(0.8, 1.0)

                # 散布図をファイルに保存
                filepath = os.path.join(self.output_dir, f"scaling_comparison_fold_{fold_idx + 1}.png")
                plt.savefig(filepath, bbox_inches="tight")
                plt.close()
                print(f"散布図を保存しました: {filepath}")

            # 全フォールドの結果をまとめた散布図も作成
            plt.figure(figsize=(10, 8))
            df_results = pd.DataFrame(results)

            for scaler in scalers.keys():
                scaler_data = df_results[df_results["scaler"] == scaler]
                plt.scatter(scaler_data["train_score"], scaler_data["test_score"], label=f"{scaler}", alpha=0.6, s=100)

            plt.plot([0.8, 1.0], [0.8, 1.0], "k--", alpha=0.3)
            plt.xlabel("Training Score")
            plt.ylabel("Test Score")
            plt.title("Scaling Methods Comparison (All Folds)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(0.8, 1.0)
            plt.ylim(0.8, 1.0)

            # 全体の散布図を保存
            filepath = os.path.join(self.output_dir, "scaling_comparison_all_folds.png")
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()
            print("\n全フォールドの散布図を保存しました:", filepath)

            return pd.DataFrame(results)

        except Exception as e:
            print(f"スケーリングデータの評価中にエラーが発生しました: {e}")
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

            filepath = os.path.join(self.output_dir, "pca_analysis.png")
            plt.savefig(filepath)
            plt.close()
            print(f"PCA分析結果を保存しました: {filepath}")

            # PCAの成分も可視化
            components_df = pd.DataFrame(
                pca.components_, columns=self.feature_names, index=[f"PC{i+1}" for i in range(n_components)]
            )

            plt.figure(figsize=(10, 6))
            sns.heatmap(components_df, cmap="coolwarm", annot=True)
            plt.title("PCA Components")

            # ファイルに保存
            filepath_components = os.path.join(self.output_dir, "pca_components.png")
            plt.savefig(filepath_components)
            plt.close()
            print(f"PCA主成分を保存しました: {filepath_components}")

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

            # NMF実行 (反復回数を増やして警告を減らす)
            nmf = NMF(n_components=n_components, random_state=42, max_iter=400)
            X_nmf = nmf.fit_transform(X_scaled)

            # 結果をデータフレーム化
            df_nmf = pd.DataFrame(X_nmf, columns=[f"NMF{i+1}" for i in range(n_components)])
            df_nmf["species"] = self.data["species"]

            # プロット
            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=df_nmf, x="NMF1", y="NMF2", hue="species", palette="viridis", s=100)
            plt.title("NMF of Iris Dataset")

            filepath = os.path.join(self.output_dir, "nmf_analysis.png")
            plt.savefig(filepath)
            plt.close()
            print(f"NMF分析結果を保存しました: {filepath}")

            # NMFの成分も可視化
            components_df = pd.DataFrame(
                nmf.components_, columns=self.feature_names, index=[f"NMF{i+1}" for i in range(n_components)]
            )

            plt.figure(figsize=(10, 6))
            sns.heatmap(components_df, cmap="coolwarm", annot=True)
            plt.title("NMF Components")

            # ファイルに保存
            filepath_components = os.path.join(self.output_dir, "nmf_components.png")
            plt.savefig(filepath_components)
            plt.close()
            print(f"NMF主成分を保存しました: {filepath_components}")

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

            filepath = os.path.join(self.output_dir, "tsne_analysis.png")
            plt.savefig(filepath)
            plt.close()
            print(f"t-SNE分析結果を保存しました: {filepath}")

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

            filepath = os.path.join(self.output_dir, "kmeans_analysis.png")
            plt.savefig(filepath)
            plt.close()
            print(f"K-means分析結果を保存しました: {filepath}")

            # 3D散布図も作成
            fig = plt.figure(figsize=(15, 10))

            # K-means結果
            ax1 = fig.add_subplot(121, projection="3d")
            ax1.scatter(
                self.X.iloc[:, 0], self.X.iloc[:, 1], self.X.iloc[:, 2], c=clusters, cmap="viridis", s=50, alpha=0.8
            )
            ax1.set_title("K-means Clustering (3D)")
            ax1.set_xlabel(self.feature_names[0])
            ax1.set_ylabel(self.feature_names[1])
            ax1.set_zlabel(self.feature_names[2])

            # 実際のラベル
            ax2 = fig.add_subplot(122, projection="3d")
            ax2.scatter(
                self.X.iloc[:, 0], self.X.iloc[:, 1], self.X.iloc[:, 2], c=self.y, cmap="viridis", s=50, alpha=0.8
            )
            ax2.set_title("Actual Classes (3D)")
            ax2.set_xlabel(self.feature_names[0])
            ax2.set_ylabel(self.feature_names[1])
            ax2.set_zlabel(self.feature_names[2])

            plt.tight_layout()

            # ファイルに保存
            filepath_3d = os.path.join(self.output_dir, "kmeans_analysis_3d.png")
            plt.savefig(filepath_3d)
            plt.close()
            print(f"K-means 3D分析結果を保存しました: {filepath_3d}")

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
        if not self._check_data_loaded():
            return None

        try:
            # リンケージ行列を計算 (DataFrameをnumpy arrayに変換)
            X_array = self.X.values
            linked = linkage(X_array, "ward")

            # デンドログラムをプロット
            plt.figure(figsize=(12, 8))
            dendrogram(
                linked,
                truncate_mode="lastp" if truncate else None,
                p=10 if truncate else None,
                leaf_font_size=10.0,
                orientation="top",
            )
            plt.title("Hierarchical Clustering Dendrogram" + (" (Truncated)" if truncate else ""))
            plt.xlabel("Sample index")
            plt.ylabel("Distance")

            # ファイル名に truncated を追加するかどうか
            truncate_str = "_truncated" if truncate else ""
            filepath = os.path.join(self.output_dir, f"dendrogram{truncate_str}.png")
            plt.savefig(filepath)
            plt.close()
            print(f"デンドログラムを保存しました: {filepath}")

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

            # 特徴量の組み合わせをプロット
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))

            # プロットのためのカラーマップ (-1はノイズ点で黒にする)
            cmap = plt.cm.get_cmap("viridis")
            import matplotlib.colors as mcolors

            cmaplist = [cmap(i) for i in range(cmap.N)]
            cmaplist[0] = (0, 0, 0, 1.0)  # ノイズ点を黒に
            cmap_custom = mcolors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)

            # 特徴量ペアのプロット
            feature_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

            for i, (f1, f2) in enumerate(feature_pairs):
                row, col = i // 3, i % 3
                axes[row, col].scatter(X_dbscan[:, f1], X_dbscan[:, f2], c=clusters, cmap=cmap_custom, s=50)
                axes[row, col].set_xlabel(self.feature_names[f1])
                axes[row, col].set_ylabel(self.feature_names[f2])
                axes[row, col].set_title(f"{self.feature_names[f1]} vs {self.feature_names[f2]}")

            plt.tight_layout()
            plt.suptitle("DBSCAN Clustering" + (" (Scaled)" if scaling else ""), y=1.02, fontsize=16)

            # スケーリングの有無をファイル名に反映
            scaled_str = "_scaled" if scaling else ""
            filepath = os.path.join(self.output_dir, f"dbscan{scaled_str}.png")
            plt.savefig(filepath)
            plt.close()
            print(f"DBSCAN分析結果を保存しました: {filepath}")

            # 3D散布図も作成
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

            # ノイズポイントとクラスターポイントを分ける
            noise_points = X_dbscan[clusters == -1]
            cluster_points = X_dbscan[clusters != -1]
            cluster_labels = clusters[clusters != -1]

            # ノイズポイントを黒で、クラスターポイントを色付きでプロット
            if len(noise_points) > 0:
                ax.scatter(
                    noise_points[:, 0],
                    noise_points[:, 1],
                    noise_points[:, 2],
                    c="black",
                    s=30,
                    alpha=0.5,
                    label="Noise",
                )

            if len(cluster_points) > 0:
                scatter = ax.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    cluster_points[:, 2],
                    c=cluster_labels,
                    cmap="viridis",
                    s=50,
                    alpha=0.8,
                    label="Clusters",
                )

                # クラスターのラベルの数を取得して凡例を追加
                legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend1)

            ax.set_title("DBSCAN Clustering 3D" + (" (Scaled)" if scaling else ""))
            ax.set_xlabel(self.feature_names[0])
            ax.set_ylabel(self.feature_names[1])
            ax.set_zlabel(self.feature_names[2])

            # ファイルに保存
            filepath_3d = os.path.join(self.output_dir, f"dbscan{scaled_str}_3d.png")
            plt.savefig(filepath_3d)
            plt.close()
            print(f"DBSCAN 3D分析結果を保存しました: {filepath_3d}")

            print("Cluster Memberships:", clusters)

            return df_dbscan
        except Exception as e:
            print(f"DBSCAN分析中にエラーが発生しました: {e}")
            return None


# サンプルコード
if __name__ == "__main__":
    print("Irisデータセット分析プログラムを実行します")

    # インスタンス作成
    iris = AnalyzeIris(output_dir="output")

    # データ読み込み
    iris.get()

    while True:
        print("\n=== Irisデータセット分析メニュー ===")
        print("1. 変数間の相関係数を確認する (get_correlation)")
        print("2. seabornを使ってpair_plotする (pair_plot)")
        print("3. ペアプロット対角成分をカーネル密度推定してプロットする (pair_plot(diag_kind='kde'))")
        print("4. 複数の教師あり学習モデルを評価する (all_supervised)")
        print("5. 学習結果の詳細を表示する (get_supervised)")
        print("6. 最良のモデルを表示する (best_supervised)")
        print("7. 特徴量の重要度を可視化する (plot_feature_importances_all)")
        print("8. 決定木を可視化する (visualize_decision_tree)")
        print("9. データスケーリング方法を比較する (plot_scaled_data)")
        print("10. PCA分析を実行する (plot_pca)")
        print("11. NMF分析を実行する (plot_nmf)")
        print("12. t-SNE分析を実行する (plot_tsne)")
        print("13. K-means分析を実行する (plot_k_means)")
        print("14. 階層的クラスタリングを実行する (plot_dendrogram)")
        print("15. DBSCANクラスタリングを実行する (plot_dbscan)")
        print("0. 終了")

        choice = input("\n実行する分析の番号を入力してください (0-15): ")

        if choice == "0":
            print("プログラムを終了します。")
            break

        try:
            choice = int(choice)

            if choice == 1:
                # 変数間の相関係数を確認する
                iris.get_correlation()

            elif choice == 2:
                # seabornを使ってpair_plotする
                iris.pair_plot()

            elif choice == 3:
                # ペアプロット対角成分をカーネル密度推定してプロットする
                iris.pair_plot(diag_kind="kde")

            elif choice == 4:
                # 複数の教師あり学習モデルを評価する
                n_neighbors = 4
                try:
                    n_input = input("KNeighborsClassifierのn_neighborsを指定してください (デフォルト: 4): ")
                    if n_input.strip():
                        n_neighbors = int(n_input)
                except:
                    print("入力が無効です。デフォルト値(4)を使用します。")

                iris.all_supervised(n_neighbors=n_neighbors)

            elif choice == 5:
                # 学習結果の詳細を表示する
                df_scores, stats = iris.get_supervised()
                if df_scores is not None:
                    print("\n各モデルの詳細なスコア:")
                    print(df_scores)
                    print("\n各モデルの統計情報:")
                    print(stats)

            elif choice == 6:
                # 最良のモデルを表示する
                best_method, best_score = iris.best_supervised()
                if best_method is not None:
                    print(f"\nベストなモデル: {best_method}, スコア: {best_score:.4f}")

            elif choice == 7:
                # 特徴量の重要度を可視化する
                iris.plot_feature_importances_all()

            elif choice == 8:
                # 決定木を可視化する
                iris.visualize_decision_tree()

            elif choice == 9:
                # データスケーリング方法を比較する
                iris.plot_scaled_data()

            elif choice == 10:
                # PCA分析を実行する
                X_scaled, df_pca, pca = iris.plot_pca(n_components=2)
                if X_scaled is not None:
                    print("\nスケーリング後のデータ (先頭5行):")
                    print(X_scaled.head())
                    print("\nスケーリング後のデータ統計情報:")
                    print(X_scaled.describe())
                    print("\nPCA成分:")
                    print(pca.components_)

            elif choice == 11:
                # NMF分析を実行する
                X_scaled, df_nmf, nmf = iris.plot_nmf(n_components=2)

            elif choice == 12:
                # t-SNE分析を実行する
                iris.plot_tsne()

            elif choice == 13:
                # K-means分析を実行する
                iris.plot_k_means()

            elif choice == 14:
                # 階層的クラスタリングを実行する
                truncate = input("デンドログラムを切り詰めますか？ (y/n, デフォルト: n): ").lower() == "y"
                iris.plot_dendrogram(truncate=truncate)

            elif choice == 15:
                # DBSCANクラスタリングを実行する
                scaling = input("データをスケーリングしますか？ (y/n, デフォルト: y): ").lower() != "n"

                try:
                    eps_input = input("eps値を指定してください (デフォルト: 0.5): ")
                    eps = float(eps_input) if eps_input.strip() else 0.5

                    min_samples_input = input("min_samples値を指定してください (デフォルト: 5): ")
                    min_samples = int(min_samples_input) if min_samples_input.strip() else 5
                except:
                    print("入力が無効です。デフォルト値を使用します。")
                    eps = 0.5
                    min_samples = 5

                iris.plot_dbscan(scaling=scaling, eps=eps, min_samples=min_samples)

            else:
                print("無効な選択です。0から15の数字を入力してください。")

        except ValueError:
            print("無効な入力です。数字を入力してください。")
        except Exception as e:
            print(f"エラーが発生しました: {e}")

        input("\nEnterキーを押して続行...")

    print("\n分析結果は以下のディレクトリに保存されています:")
    print(os.path.abspath(iris.output_dir))
