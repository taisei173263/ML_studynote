import os
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

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

# 設定（環境変数などから読み込むように改善可能）
CONFIG = {
    "output_dir": "output",
    "default_dpi": 100,
    "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    "markers": ["o", "^", "s", "v", "D"],
    "plt_params": {"figure_size": (10, 8), "scatter_size": 100, "alpha": 0.8},
}


# デコレータ関数：エラーハンドリング
def error_handler(func):
    """関数のエラーを捕捉して処理するデコレータ"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"{func.__name__}の実行中にエラーが発生しました: {e}")
            import traceback

            traceback.print_exc()
            return None

    return wrapper


# ファイル操作クラス
class FileHandler:
    """ファイル操作を担当するクラス"""

    def __init__(self, output_dir: str):
        """
        Parameters
        ----------
        output_dir : str
            出力ディレクトリのパス
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_figure(self, filename: str, fig: plt.Figure = None, dpi: int = None) -> str:
        """
        図をファイルに保存する

        Parameters
        ----------
        filename : str
            ファイル名（拡張子含む）
        fig : plt.Figure, optional
            保存する図。Noneの場合は現在の図を使用
        dpi : int, optional
            解像度。Noneの場合はデフォルト値を使用

        Returns
        -------
        str
            保存されたファイルのパス
        """
        filepath = os.path.join(self.output_dir, filename)

        if fig:
            fig.savefig(filepath, dpi=dpi or CONFIG["default_dpi"], bbox_inches="tight")
        else:
            plt.savefig(filepath, dpi=dpi or CONFIG["default_dpi"], bbox_inches="tight")

        plt.close()
        print(f"図を保存しました: {filepath}")
        return filepath


# データ読み込みと管理クラス
class DataLoader:
    """データの読み込みと管理を担当するクラス"""

    def __init__(self):
        """データローダーの初期化"""
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = []
        self.target_names = []

    @error_handler
    def load_data(self, dataset_name: str, **kwargs) -> pd.DataFrame:
        """
        指定されたデータセットを読み込む

        Parameters
        ----------
        dataset_name : str
            データセット名（'iris'など）
        **kwargs : dict
            データセット読み込みに関する追加パラメータ

        Returns
        -------
        pd.DataFrame
            読み込まれたデータフレーム
        """
        if dataset_name.lower() == "iris":
            return self._load_iris()
        elif dataset_name.lower() == "custom":
            # カスタムデータセット読み込みの例
            file_path = kwargs.get("file_path")
            if not file_path:
                raise ValueError("custom データセットには file_path パラメータが必要です")
            return self._load_custom_csv(file_path, **kwargs)
        else:
            raise ValueError(f"未対応のデータセット: {dataset_name}")

    @error_handler
    def _load_iris(self) -> pd.DataFrame:
        """
        irisデータセットを読み込む

        Returns
        -------
        pd.DataFrame
            Irisデータセット
        """
        iris = load_iris()
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names

        # データフレームに変換
        self.data = pd.DataFrame(data=iris.data, columns=self.feature_names)
        self.data["target"] = iris.target
        self.data["species"] = self.data["target"].map({i: name for i, name in enumerate(self.target_names)})

        # X, yに分割
        self.X = self.data[self.feature_names]
        self.y = self.data["target"]

        return self.data

    @error_handler
    def _load_custom_csv(self, file_path: str, target_column: str = None, **kwargs) -> pd.DataFrame:
        """
        カスタムCSVファイルを読み込む

        Parameters
        ----------
        file_path : str
            CSVファイルのパス
        target_column : str, optional
            ターゲット列の名前
        **kwargs : dict
            pandas.read_csvに渡す追加パラメータ

        Returns
        -------
        pd.DataFrame
            読み込まれたデータフレーム
        """
        self.data = pd.read_csv(file_path, **kwargs)

        if target_column and target_column in self.data.columns:
            self.y = self.data[target_column]
            feature_cols = [col for col in self.data.columns if col != target_column]
            self.X = self.data[feature_cols]
            self.feature_names = feature_cols
        else:
            # ターゲット列が指定されていない場合は最後の列をターゲットとみなす
            self.feature_names = list(self.data.columns[:-1])
            self.X = self.data[self.feature_names]
            self.y = self.data.iloc[:, -1]

        return self.data

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        データを取得する

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series]
            完全なデータセット、特徴量、ターゲット
        """
        return self.data, self.X, self.y


# 可視化クラス
class Visualizer:
    """可視化機能を担当するクラス"""

    def __init__(self, file_handler: FileHandler):
        """
        Parameters
        ----------
        file_handler : FileHandler
            ファイル操作のためのハンドラ
        """
        self.file_handler = file_handler

    @error_handler
    def plot_correlation(self, X: pd.DataFrame, filename: str = "correlation_matrix.png") -> None:
        """
        相関行列を可視化する

        Parameters
        ----------
        X : pd.DataFrame
            特徴量データフレーム
        filename : str, optional
            保存するファイル名
        """
        plt.figure(figsize=CONFIG["plt_params"]["figure_size"])
        corr = X.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Feature Correlation Matrix")
        self.file_handler.save_figure(filename)

    @error_handler
    def plot_pair(self, data: pd.DataFrame, hue: str, diag_kind: str = None, filename_suffix: str = "") -> None:
        """
        ペアプロットを作成する

        Parameters
        ----------
        data : pd.DataFrame
            完全なデータセット
        hue : str
            色分けに使用する列名
        diag_kind : str, optional
            対角線に表示するプロットの種類 ('hist', 'kde')
        filename_suffix : str, optional
            ファイル名に追加する接尾辞
        """
        # 対角線に表示するプロットの種類を設定
        if diag_kind in ["hist", "kde"]:
            pair_plot = sns.pairplot(data, hue=hue, diag_kind=diag_kind)
            pair_plot.fig.suptitle(f"Pair Plot with {diag_kind.upper()} on Diagonal", y=1.02)
            self.file_handler.save_figure(f"pairplot_{diag_kind}{filename_suffix}.png")
        else:
            # ヒストグラムバージョン
            hist_plot = sns.pairplot(data, hue=hue, diag_kind="hist")
            hist_plot.fig.suptitle("Pair Plot with Histograms on Diagonal", y=1.02)
            self.file_handler.save_figure(f"pairplot_hist{filename_suffix}.png")

            # KDEバージョン
            kde_plot = sns.pairplot(data, hue=hue, diag_kind="kde")
            kde_plot.fig.suptitle("Pair Plot with KDE on Diagonal", y=1.02)
            self.file_handler.save_figure(f"pairplot_kde{filename_suffix}.png")

    @error_handler
    def plot_feature_importances(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> None:
        """
        特徴量の重要度を可視化する

        Parameters
        ----------
        models : Dict[str, Any]
            重要度を持つモデルの辞書 {モデル名: モデルクラス}
        X : pd.DataFrame
            特徴量データフレーム
        y : pd.Series
            ターゲット変数
        """
        if not models:
            print("特徴量重要度を持つモデルが指定されていません")
            return

        # 図を作成
        fig = plt.figure(figsize=(15, 10))
        feature_names = X.columns

        for i, (model_name, model) in enumerate(models.items()):
            # モデルを学習
            model.fit(X, y)

            # 重要度を取得
            importance = model.feature_importances_

            # インデックスでソート（降順）
            sorted_indices = np.argsort(importance)[::-1]

            # サブプロット作成
            ax = fig.add_subplot(1, len(models), i + 1)

            # 横棒グラフの作成
            y_pos = np.arange(len(feature_names))
            ax.barh(y_pos, importance[sorted_indices], align="center")
            ax.set_yticks(y_pos)
            ax.set_yticklabels([feature_names[idx] for idx in sorted_indices])
            ax.invert_yaxis()  # 上から下へ値が大きい順に表示
            ax.set_xlabel("Feature Importance (0-1.0)")
            ax.set_xlim(0, 1.0)  # X軸の範囲を0～1に設定
            ax.set_title(f"{model_name}")

        plt.tight_layout()
        self.file_handler.save_figure("feature_importance_all_models.png")

    @error_handler
    def plot_decision_tree(
        self, X: pd.DataFrame, y: pd.Series, feature_names: List[str], class_names: List[str] = None
    ) -> None:
        """
        決定木モデルを可視化する

        Parameters
        ----------
        X : pd.DataFrame
            特徴量データフレーム
        y : pd.Series
            ターゲット変数
        feature_names : List[str]
            特徴量の名前リスト
        class_names : List[str], optional
            クラスの名前リスト
        """
        tree = DecisionTreeClassifier()
        tree.fit(X, y)

        plt.figure(figsize=(15, 10))
        plot_tree(tree, filled=True, feature_names=feature_names, class_names=class_names)
        plt.title("Decision Tree Visualization")
        plt.tight_layout()

        self.file_handler.save_figure("decision_tree.png")

    @error_handler
    def plot_scaling_comparison(self, results: List[Dict[str, Any]], fold_idx: int = None) -> None:
        """
        スケーリング手法の比較散布図を作成する

        Parameters
        ----------
        results : List[Dict[str, Any]]
            各スケーリング手法の結果リスト
        fold_idx : int, optional
            分割インデックス（Noneの場合は全体の結果とみなす）
        """
        plt.figure(figsize=CONFIG["plt_params"]["figure_size"])

        # 辞書をデータフレームに変換
        df_results = pd.DataFrame(results)

        if fold_idx is not None:
            # 特定のフォールドの結果のみを表示
            fold_results = df_results[df_results["fold"] == fold_idx + 1]

            for _, row in fold_results.iterrows():
                plt.scatter(
                    row["train_score"], row["test_score"], label=row["scaler"], s=CONFIG["plt_params"]["scatter_size"]
                )

            title = f"Scaling Methods Comparison (Fold {fold_idx + 1})"
            filename = f"scaling_comparison_fold_{fold_idx + 1}.png"
        else:
            # 全フォールドの結果をスケーラーごとに色分け
            for scaler in df_results["scaler"].unique():
                scaler_data = df_results[df_results["scaler"] == scaler]
                plt.scatter(
                    scaler_data["train_score"],
                    scaler_data["test_score"],
                    label=scaler,
                    alpha=0.6,
                    s=CONFIG["plt_params"]["scatter_size"],
                )

            title = "Scaling Methods Comparison (All Folds)"
            filename = "scaling_comparison_all_folds.png"

        # 理想的な性能を示す対角線を追加
        plt.plot([0.8, 1.0], [0.8, 1.0], "k--", alpha=0.3)
        plt.xlabel("Training Score")
        plt.ylabel("Test Score")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0.8, 1.0)
        plt.ylim(0.8, 1.0)

        self.file_handler.save_figure(filename)

    @error_handler
    def plot_dimension_reduction(
        self,
        technique: str,
        X_transformed: np.ndarray,
        labels: np.ndarray,
        label_names: List[str] = None,
        explained_variance: List[float] = None,
        components: np.ndarray = None,
        feature_names: List[str] = None,
    ) -> None:
        """
        次元削減結果を可視化する

        Parameters
        ----------
        technique : str
            次元削減手法の名前（'PCA', 'NMF', 't-SNE'）
        X_transformed : np.ndarray
            変換後のデータ
        labels : np.ndarray
            データ点のラベル
        label_names : List[str], optional
            ラベルの名前リスト
        explained_variance : List[float], optional
            各成分の寄与率（PCAのみ）
        components : np.ndarray, optional
            成分行列（PCA, NMFのみ）
        feature_names : List[str], optional
            特徴量の名前リスト（成分可視化に使用）
        """
        # データフレームに変換
        df = pd.DataFrame(X_transformed, columns=[f"{technique}1", f"{technique}2"])
        df["label"] = labels

        if label_names is not None and len(label_names) > 0:
            df["label_name"] = [label_names[i] for i in labels]
            hue = "label_name"
        else:
            hue = "label"

        # 散布図プロット
        plt.figure(figsize=CONFIG["plt_params"]["figure_size"])
        scatter = sns.scatterplot(
            data=df,
            x=f"{technique}1",
            y=f"{technique}2",
            hue=hue,
            palette="viridis",
            s=CONFIG["plt_params"]["scatter_size"],
        )

        if technique == "PCA" and explained_variance is not None:
            plt.xlabel(f"PC1 ({explained_variance[0]:.2f})")
            plt.ylabel(f"PC2 ({explained_variance[1]:.2f})")
        else:
            plt.xlabel(f"{technique} Component 1")
            plt.ylabel(f"{technique} Component 2")

        plt.title(f"{technique} of Dataset")
        self.file_handler.save_figure(f"{technique.lower()}_analysis.png")

        # 成分を可視化（PCAとNMFのみ）
        if components is not None and feature_names and technique in ["PCA", "NMF"]:
            components_df = pd.DataFrame(components, columns=feature_names, index=[f"{technique}1", f"{technique}2"])

            plt.figure(figsize=(10, 6))
            sns.heatmap(components_df, cmap="coolwarm", annot=True)
            plt.title(f"{technique} Components")

            self.file_handler.save_figure(f"{technique.lower()}_components.png")

    @error_handler
    def plot_clusters(
        self,
        method: str,
        X: pd.DataFrame,
        clusters: np.ndarray,
        actual_labels: np.ndarray = None,
        feature_names: List[str] = None,
        plot_3d: bool = True,
        is_scaled: bool = False,
    ) -> None:
        """
        クラスタリング結果を可視化する

        Parameters
        ----------
        method : str
            クラスタリング手法の名前（'KMeans', 'DBSCAN'）
        X : pd.DataFrame
            特徴量データフレーム
        clusters : np.ndarray
            クラスタリング結果のラベル
        actual_labels : np.ndarray, optional
            実際のクラスラベル
        feature_names : List[str], optional
            特徴量の名前リスト
        plot_3d : bool, optional
            3D散布図も作成するかどうか
        is_scaled : bool, optional
            データがスケーリングされているかどうか
        """
        X_array = X.values  # numpy配列に変換

        if method == "DBSCAN":
            # DBSCANの場合、特徴量の組み合わせを全て可視化
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
                axes[row, col].scatter(X_array[:, f1], X_array[:, f2], c=clusters, cmap=cmap_custom, s=50)

                if feature_names:
                    axes[row, col].set_xlabel(feature_names[f1])
                    axes[row, col].set_ylabel(feature_names[f2])
                    axes[row, col].set_title(f"{feature_names[f1]} vs {feature_names[f2]}")
                else:
                    axes[row, col].set_xlabel(f"Feature {f1}")
                    axes[row, col].set_ylabel(f"Feature {f2}")
                    axes[row, col].set_title(f"Feature {f1} vs Feature {f2}")

            plt.tight_layout()
            plt.suptitle(f"DBSCAN Clustering{' (Scaled)' if is_scaled else ''}", y=1.02, fontsize=16)

            scaled_str = "_scaled" if is_scaled else ""
            self.file_handler.save_figure(f"dbscan{scaled_str}.png")

        else:
            # K-meansなど、2つのプロットでクラスターと実際のラベルを比較
            plt.figure(figsize=(15, 5))

            # クラスタリング結果
            plt.subplot(1, 2, 1)
            sns.scatterplot(x=X_array[:, 0], y=X_array[:, 1], hue=clusters, palette="viridis", s=100)

            if feature_names:
                plt.xlabel(feature_names[0])
                plt.ylabel(feature_names[1])
            else:
                plt.xlabel("Feature 0")
                plt.ylabel("Feature 1")

            plt.title(f"{method} Clustering")

            # 実際のラベル
            if actual_labels is not None:
                plt.subplot(1, 2, 2)
                sns.scatterplot(x=X_array[:, 0], y=X_array[:, 1], hue=actual_labels, palette="viridis", s=100)

                if feature_names:
                    plt.xlabel(feature_names[0])
                    plt.ylabel(feature_names[1])
                else:
                    plt.xlabel("Feature 0")
                    plt.ylabel("Feature 1")

                plt.title("Actual Classes")

            plt.tight_layout()
            self.file_handler.save_figure(f"{method.lower()}_analysis.png")

        # 3D散布図の作成（オプション）
        if plot_3d and X_array.shape[1] >= 3:
            if method == "DBSCAN":
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection="3d")

                # ノイズポイントとクラスターポイントを分ける
                noise_points = X_array[clusters == -1]
                cluster_points = X_array[clusters != -1]
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

                if feature_names:
                    ax.set_xlabel(feature_names[0])
                    ax.set_ylabel(feature_names[1])
                    ax.set_zlabel(feature_names[2])
                else:
                    ax.set_xlabel("Feature 0")
                    ax.set_ylabel("Feature 1")
                    ax.set_zlabel("Feature 2")

                title_suffix = " (Scaled)" if is_scaled else ""
                ax.set_title(f"{method} Clustering 3D{title_suffix}")

                plt.tight_layout()
                scaled_str = "_scaled" if is_scaled else ""
                self.file_handler.save_figure(f"{method.lower()}{scaled_str}_3d.png")
            else:
                fig = plt.figure(figsize=(15, 10))

                # クラスタリング結果
                ax1 = fig.add_subplot(121, projection="3d")
                ax1.scatter(X_array[:, 0], X_array[:, 1], X_array[:, 2], c=clusters, cmap="viridis", s=50, alpha=0.8)

                if feature_names:
                    ax1.set_xlabel(feature_names[0])
                    ax1.set_ylabel(feature_names[1])
                    ax1.set_zlabel(feature_names[2])
                else:
                    ax1.set_xlabel("Feature 0")
                    ax1.set_ylabel("Feature 1")
                    ax1.set_zlabel("Feature 2")

                ax1.set_title(f"{method} Clustering (3D)")

                # 実際のラベル
                if actual_labels is not None:
                    ax2 = fig.add_subplot(122, projection="3d")
                    ax2.scatter(
                        X_array[:, 0], X_array[:, 1], X_array[:, 2], c=actual_labels, cmap="viridis", s=50, alpha=0.8
                    )

                    if feature_names:
                        ax2.set_xlabel(feature_names[0])
                        ax2.set_ylabel(feature_names[1])
                        ax2.set_zlabel(feature_names[2])
                    else:
                        ax2.set_xlabel("Feature 0")
                        ax2.set_ylabel("Feature 1")
                        ax2.set_zlabel("Feature 2")

                    ax2.set_title("Actual Classes (3D)")

                plt.tight_layout()
                self.file_handler.save_figure(f"{method.lower()}_analysis_3d.png")

    @error_handler
    def plot_dendrogram(self, linkage_matrix: np.ndarray, truncate: bool = False, labels: List[str] = None) -> None:
        """
        階層的クラスタリングのデンドログラムを作成する

        Parameters
        ----------
        linkage_matrix : np.ndarray
            階層的クラスタリングのリンケージ行列
        truncate : bool, optional
            デンドログラムを切り詰めるかどうか
        labels : List[str], optional
            ノードラベル
        """
        plt.figure(figsize=(12, 8))

        # truncateがTrueの場合とFalseの場合で異なるパラメータを渡す
        if truncate:
            dendrogram(
                linkage_matrix,
                truncate_mode="lastp",
                p=10,
                leaf_font_size=10.0,
                orientation="top",
                labels=labels,
            )
        else:
            dendrogram(
                linkage_matrix,
                leaf_font_size=10.0,
                orientation="top",
                labels=labels,
            )

        plt.title("Hierarchical Clustering Dendrogram" + (" (Truncated)" if truncate else ""))
        plt.xlabel("Sample index")
        plt.ylabel("Distance")

        truncate_str = "_truncated" if truncate else ""
        self.file_handler.save_figure(f"dendrogram{truncate_str}.png")

    def plot_scaling_fold_performance(self, fold_results: List[Dict[str, Any]], fold_idx: int) -> None:
        """
        フォールドごとのスケーリングパフォーマンスを可視化

        Parameters
        ----------
        fold_results : List[Dict[str, Any]]
            フォールドの結果リスト
        fold_idx : int
            フォールドのインデックス
        """
        plt.figure(figsize=CONFIG["plt_params"]["figure_size"])
        for result in fold_results:
            plt.scatter(
                result["train_score"],
                result["test_score"],
                label=result["scaler"],
                s=CONFIG["plt_params"]["scatter_size"],
            )

        self._add_performance_plot_elements(f"Scaling Methods Comparison (Fold {fold_idx + 1})")
        self.file_handler.save_figure(f"scaling_comparison_fold_{fold_idx + 1}.png")
        plt.close()

    def plot_scaling_feature_scatter(
        self, scaled_data: Dict[str, Dict[str, Any]], feature_pairs: List[Tuple[str, str]], y: pd.Series, fold_idx: int
    ) -> None:
        """
        スケーリングされた特徴量の散布図を作成

        Parameters
        ----------
        scaled_data : Dict[str, Dict[str, Any]]
            スケーリングされたデータ
        feature_pairs : List[Tuple[str, str]]
            特徴量のペアリスト
        y : pd.Series
            ターゲット変数
        fold_idx : int
            フォールドのインデックス
        """
        unique_classes = np.unique(y)
        colors = CONFIG["colors"][: len(unique_classes)]
        markers = CONFIG["markers"][: len(unique_classes)]
        class_colors = {cls: color for cls, color in zip(unique_classes, colors)}

        for name, data in scaled_data.items():
            n_pairs = len(feature_pairs)
            n_cols = 5
            n_rows = (n_pairs + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
            axes = axes.flatten()

            X = data["X_train_scaled"]
            y_train = data["y_train"]

            for i, (feature1, feature2) in enumerate(feature_pairs):
                if i < len(axes):
                    ax = axes[i]
                    for cls_idx, cls in enumerate(unique_classes):
                        mask = y_train == cls
                        ax.scatter(
                            X.loc[mask, feature1],
                            X.loc[mask, feature2],
                            c=class_colors[cls],
                            marker=markers[cls_idx],
                            alpha=CONFIG["plt_params"]["alpha"],
                            s=30,
                            label=f"Class {cls}",
                        )

                    ax.set_xlabel(feature1)
                    ax.set_ylabel(feature2)
                    ax.set_title(f"{feature1} vs {feature2}")
                    ax.grid(True, alpha=0.3)

                    if i == 0:
                        ax.legend()

            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            self.file_handler.save_figure(f"feature_scatter_{name}_fold_{fold_idx + 1}.png", fig=fig, dpi=150)
            plt.close(fig)

    def plot_scaling_all_folds(self, results: List[Dict[str, Any]]) -> None:
        """
        全フォールドのスケーリング結果をまとめて可視化

        Parameters
        ----------
        results : List[Dict[str, Any]]
            全フォールドの結果リスト
        """
        plt.figure(figsize=CONFIG["plt_params"]["figure_size"])
        df_results = pd.DataFrame(results)

        for scaler in df_results["scaler"].unique():
            scaler_data = df_results[df_results["scaler"] == scaler]
            plt.scatter(
                scaler_data["train_score"],
                scaler_data["test_score"],
                label=scaler,
                alpha=0.6,
                s=CONFIG["plt_params"]["scatter_size"],
            )

        self._add_performance_plot_elements("Scaling Methods Comparison (All Folds)")
        self.file_handler.save_figure("scaling_comparison_all_folds.png")

    def _add_performance_plot_elements(self, title: str) -> None:
        """パフォーマンスプロットの共通要素を追加"""
        plt.plot([0.8, 1.0], [0.8, 1.0], "k--", alpha=0.3)
        plt.xlabel("Training Score")
        plt.ylabel("Test Score")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0.8, 1.0)
        plt.ylim(0.8, 1.0)


class ScalingReporter:
    """スケーリング分析結果のレポート生成を担当するクラス"""

    def __init__(self, file_handler: FileHandler):
        """
        Parameters
        ----------
        file_handler : FileHandler
            ファイル操作のためのハンドラ
        """
        self.file_handler = file_handler

    def create_performance_summary(self, results: List[Dict[str, Any]]) -> None:
        """パフォーマンスサマリーを作成して保存"""
        df_results = pd.DataFrame(results)
        summary_df = df_results.groupby("scaler").agg({"train_score": ["mean", "std"], "test_score": ["mean", "std"]})
        summary_df.columns = ["train_mean", "train_std", "test_mean", "test_std"]
        summary_df = summary_df.reset_index()
        summary_df = summary_df.sort_values(by="test_mean", ascending=False)

        self._print_summary(summary_df)
        self._save_summary(summary_df)

    def _print_summary(self, summary_df: pd.DataFrame) -> None:
        """サマリーを表示"""
        print("\n=== スケーリング手法の平均パフォーマンス ===")
        print(summary_df.to_string(float_format=lambda x: f"{x:.3f}"))

    def _save_summary(self, summary_df: pd.DataFrame) -> None:
        """サマリーをファイルに保存"""
        summary_path = os.path.join(self.file_handler.output_dir, "scaling_performance_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n平均パフォーマンス結果を保存しました: {summary_path}")


# モデル管理クラス
class ModelHandler:
    """機械学習モデルの管理と評価を担当するクラス"""

    def __init__(self):
        """モデルハンドラの初期化"""
        self.model_registry = {}
        self.model_scores = {}
        self._register_default_models()

    def _register_default_models(self) -> None:
        """デフォルトモデルを登録する"""
        self.register_model("LogisticRegression", LogisticRegression(max_iter=1000))
        self.register_model("LinearSVC", LinearSVC(max_iter=1000, dual="auto"))
        self.register_model("SVC", SVC())
        self.register_model("DecisionTreeClassifier", DecisionTreeClassifier())
        self.register_model("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=4))
        self.register_model("LinearRegression", LinearRegression())
        self.register_model("RandomForestClassifier", RandomForestClassifier())
        self.register_model("GradientBoostingClassifier", GradientBoostingClassifier())
        self.register_model("MLPClassifier", MLPClassifier(max_iter=1000))

    def register_model(self, name: str, model: Any) -> None:
        """
        新しいモデルを登録する

        Parameters
        ----------
        name : str
            モデルの名前
        model : Any
            scikit-learnモデルのインスタンス
        """
        self.model_registry[name] = model

    def get_models_with_feature_importance(self) -> Dict[str, Any]:
        """
        特徴量重要度を持つモデルを取得する

        Returns
        -------
        Dict[str, Any]
            特徴量重要度を持つモデルの辞書
        """
        importance_models = {}
        for name, model in self.model_registry.items():
            if hasattr(model, "feature_importances_") or name in [
                "DecisionTreeClassifier",
                "RandomForestClassifier",
                "GradientBoostingClassifier",
            ]:
                importance_models[name] = model
        return importance_models

    @error_handler
    def evaluate_models(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, model_names: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        モデルを評価する

        Parameters
        ----------
        X : pd.DataFrame
            特徴量データフレーム
        y : pd.Series
            ターゲット変数
        n_splits : int, optional
            交差検証の分割数
        model_names : List[str], optional
            評価するモデルの名前リスト

        Returns
        -------
        Dict[str, Dict[str, Any]]
            モデルごとの評価結果
        """
        # K分割交差検証
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # 評価するモデルを選択
        if model_names:
            models_to_evaluate = {
                name: self.model_registry[name] for name in model_names if name in self.model_registry
            }
        else:
            models_to_evaluate = self.model_registry

        # 結果格納用
        self.model_scores = {}

        # 各モデルに対して評価
        for name, model in models_to_evaluate.items():
            print(f"=== {name} ===")
            test_scores = []
            train_scores = []

            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

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

    @error_handler
    def get_model_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        モデル評価結果を取得する

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            スコアのデータフレームと統計情報
        """
        if not self.model_scores:
            print("モデルの評価が行われていません。先にevaluate_models()を実行してください。")
            return None, None

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

    @error_handler
    def find_best_model(self) -> Tuple[str, float]:
        """
        最良のモデルを特定する

        Returns
        -------
        Tuple[str, float]
            最良のモデル名とそのスコア
        """
        if not self.model_scores:
            print("モデルの評価が行われていません。先にevaluate_models()を実行してください。")
            return None, 0

        best_model = None
        best_score = 0

        for model_name, scores in self.model_scores.items():
            mean_score = scores["mean_test_score"]
            if mean_score > best_score:
                best_score = mean_score
                best_model = model_name

        return best_model, best_score


class DataAnalyzer:
    """各コンポーネントを統合した分析クラス"""

    def __init__(self, output_dir: str = CONFIG["output_dir"]):
        """
        Parameters
        ----------
        output_dir : str, optional
            出力ディレクトリのパス
        """
        self.file_handler = FileHandler(output_dir)
        self.data_loader = DataLoader()
        self.visualizer = Visualizer(self.file_handler)
        self.model_handler = ModelHandler()

        # 分析用変数
        self.data = None
        self.X = None
        self.y = None

        self.scalers = {
            "No Scaling": None,
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "Normalizer": Normalizer(),
        }

    def load_dataset(self, dataset_name: str, **kwargs) -> pd.DataFrame:
        """
        データセットを読み込む

        Parameters
        ----------
        dataset_name : str
            データセット名
        **kwargs : dict
            データセット読み込みに関する追加パラメータ

        Returns
        -------
        pd.DataFrame
            読み込まれたデータフレーム
        """
        self.data = self.data_loader.load_data(dataset_name, **kwargs)
        self.X = self.data_loader.X
        self.y = self.data_loader.y

        return self.data

    def _check_data_loaded(self) -> bool:
        """
        データが読み込まれているか確認する

        Returns
        -------
        bool
            データが読み込まれているかどうか
        """
        if self.data is None or self.X is None or self.y is None:
            print("データをまだ読み込んでいません")
            return False
        return True

    def get_correlation(self) -> pd.DataFrame:
        """
        変数間の相関係数を計算し可視化する

        Returns
        -------
        pd.DataFrame
            相関行列
        """
        if not self._check_data_loaded():
            return None

        # 相関行列の計算
        corr = self.X.corr()

        # 可視化
        self.visualizer.plot_correlation(self.X)

        return corr

    def pair_plot(self, diag_kind: str = None) -> bool:
        """
        ペアプロットを作成する

        Parameters
        ----------
        diag_kind : str, optional
            対角線に表示するプロットの種類 ('hist', 'kde')

        Returns
        -------
        bool
            成功したかどうか
        """
        if not self._check_data_loaded():
            return False

        self.visualizer.plot_pair(self.data, "species", diag_kind)
        return True

    def all_supervised(self, n_neighbors: int = 4, model_names: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        複数の教師あり学習モデルを評価する

        Parameters
        ----------
        n_neighbors : int, optional
            KNeighborsClassifierのn_neighbors値
        model_names : List[str], optional
            評価するモデルの名前リスト

        Returns
        -------
        Dict[str, Dict[str, Any]]
            モデルごとの評価結果
        """
        if not self._check_data_loaded():
            return None

        # KNeighborsClassifierのパラメータを更新
        knn = self.model_handler.model_registry.get("KNeighborsClassifier")
        if knn and n_neighbors != 4:
            self.model_handler.register_model("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=n_neighbors))

        return self.model_handler.evaluate_models(self.X, self.y, model_names=model_names)

    def get_supervised(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        学習結果をDataFrameで返す

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            スコアのデータフレームと統計情報
        """
        return self.model_handler.get_model_results()

    def best_supervised(self) -> Tuple[str, float]:
        """
        最良のモデルを返す

        Returns
        -------
        Tuple[str, float]
            最良のモデル名とそのスコア
        """
        return self.model_handler.find_best_model()

    def plot_feature_importances_all(self) -> bool:
        """
        特徴量の重要度を可視化する

        Returns
        -------
        bool
            成功したかどうか
        """
        if not self._check_data_loaded():
            return False

        importance_models = self.model_handler.get_models_with_feature_importance()
        self.visualizer.plot_feature_importances(importance_models, self.X, self.y)
        return True

    def visualize_decision_tree(self) -> bool:
        """
        決定木を可視化する

        Returns
        -------
        bool
            成功したかどうか
        """
        if not self._check_data_loaded():
            return False

        self.visualizer.plot_decision_tree(
            self.X, self.y, self.data_loader.feature_names, self.data_loader.target_names
        )
        return True

    def plot_scaled_data(self) -> pd.DataFrame:
        """
        異なるスケーリング手法でデータを変換し、LinearSVCの結果と特徴量散布図を評価する

        Returns
        -------
        pd.DataFrame
            スケーリング結果のデータフレーム
        """
        if not self._check_data_loaded():
            return None

        try:
            # 特徴量ペアの生成
            feature_pairs = self._generate_feature_pairs()

            # スケーリング分析の実行
            results = []
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.X)):
                X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

                print("=" * 80)
                print(f"Fold {fold_idx + 1}/5")
                print("=" * 80)

                fold_results = []
                scaled_data = {}

                # 各スケーラーでの評価
                for name, scaler in self.scalers.items():
                    # データのスケーリング
                    if scaler:
                        X_train_scaled = pd.DataFrame(
                            scaler.fit_transform(X_train), columns=self.X.columns, index=X_train.index
                        )
                        X_test_scaled = pd.DataFrame(
                            scaler.transform(X_test), columns=self.X.columns, index=X_test.index
                        )
                    else:
                        X_train_scaled = X_train.copy()
                        X_test_scaled = X_test.copy()

                    # モデル評価
                    model = LinearSVC(max_iter=1000, dual="auto")
                    model.fit(X_train_scaled, y_train)

                    train_score = accuracy_score(y_train, model.predict(X_train_scaled))
                    test_score = accuracy_score(y_test, model.predict(X_test_scaled))

                    result = {
                        "fold": fold_idx + 1,
                        "scaler": name,
                        "train_score": train_score,
                        "test_score": test_score,
                    }
                    results.append(result)
                    fold_results.append(result)

                    # スケーリングされたデータを保存
                    scaled_data[name] = {
                        "X_train_scaled": X_train_scaled,
                        "y_train": y_train,
                    }

                    print(f"{name:<14}: test score: {test_score:.3f}      train score: {train_score:.3f}")

                # フォールドごとのパフォーマンス散布図
                self.visualizer.plot_scaling_fold_performance(fold_results, fold_idx)

                # 特徴量散布図
                self.visualizer.plot_scaling_feature_scatter(scaled_data, feature_pairs, y_train, fold_idx)

            # 全体の結果を可視化
            self.visualizer.plot_scaling_comparison(results)

            # パフォーマンスサマリーの作成
            scaling_reporter = ScalingReporter(self.file_handler)
            scaling_reporter.create_performance_summary(results)

            return pd.DataFrame(results)

        except Exception as e:
            print(f"スケーリングデータの評価中にエラーが発生しました: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _generate_feature_pairs(self) -> List[Tuple[str, str]]:
        """特徴量ペアの組み合わせを生成"""
        feature_names = self.X.columns
        return [
            (feature_names[i], feature_names[j])
            for i in range(len(feature_names))
            for j in range(i + 1, len(feature_names))
        ]

    def plot_pca(self, n_components: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
        """
        PCA分析を行い結果をプロットする

        Parameters
        ----------
        n_components : int, optional
            主成分の数

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, Any]
            スケーリングデータ、PCA結果、PCAモデル
        """
        if not self._check_data_loaded():
            return None, None, None

        # データのスケーリング
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        # PCA実行
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # 可視化
        self.visualizer.plot_dimension_reduction(
            "PCA",
            X_pca,
            self.y,
            self.data_loader.target_names,
            pca.explained_variance_ratio_,
            pca.components_,
            self.data_loader.feature_names,
        )

        return pd.DataFrame(X_scaled, columns=self.data_loader.feature_names), pd.DataFrame(X_pca), pca

    def plot_nmf(self, n_components: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
        """
        NMF分析を行い結果をプロットする

        Parameters
        ----------
        n_components : int, optional
            成分の数

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, Any]
            スケーリングデータ、NMF結果、NMFモデル
        """
        if not self._check_data_loaded():
            return None, None, None

        # データのスケーリング (負の値は使えないのでMinMaxScalerを使用)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(self.X)

        # NMF実行
        nmf = NMF(n_components=n_components, random_state=42, max_iter=400)
        X_nmf = nmf.fit_transform(X_scaled)

        # 可視化
        self.visualizer.plot_dimension_reduction(
            "NMF", X_nmf, self.y, self.data_loader.target_names, None, nmf.components_, self.data_loader.feature_names
        )

        return pd.DataFrame(X_scaled, columns=self.data_loader.feature_names), pd.DataFrame(X_nmf), nmf

    def plot_tsne(self) -> pd.DataFrame:
        """
        t-SNE分析を行い結果をプロットする

        Returns
        -------
        pd.DataFrame
            t-SNE結果のデータフレーム
        """
        if not self._check_data_loaded():
            return None

        # t-SNE実行
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(self.X)

        # 可視化
        self.visualizer.plot_dimension_reduction("t-SNE", X_tsne, self.y, self.data_loader.target_names)

        return pd.DataFrame(X_tsne)

    def plot_k_means(self) -> pd.DataFrame:
        """
        K-means分析を行い結果をプロットする

        Returns
        -------
        pd.DataFrame
            クラスタリング結果を含むデータフレーム
        """
        if not self._check_data_loaded():
            return None

        # K-means実行
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(self.X)

        # 可視化
        self.visualizer.plot_clusters("KMeans", self.X, clusters, self.y, self.data_loader.feature_names)

        print("KMeans法で予測したラベル:")
        print(clusters)
        print("\n実際のラベル:")
        print(self.y.values)

        # 結果をデータフレームに追加
        df_kmeans = self.X.copy()
        df_kmeans["cluster"] = clusters
        df_kmeans["actual"] = self.y

        return df_kmeans

    def plot_dendrogram(self, truncate: bool = False) -> np.ndarray:
        """
        階層的クラスタリングのデンドログラムをプロットする

        Parameters
        ----------
        truncate : bool, optional
            デンドログラムを切り詰めるかどうか

        Returns
        -------
        np.ndarray
            リンケージ行列
        """
        if not self._check_data_loaded():
            return None

        # リンケージ行列を計算
        linked = linkage(self.X.values, "ward")

        # 可視化
        self.visualizer.plot_dendrogram(linked, truncate)

        return linked

    def plot_dbscan(self, scaling: bool = False, eps: float = 0.5, min_samples: int = 5) -> pd.DataFrame:
        """
        DBSCAN分析を行い結果をプロットする

        Parameters
        ----------
        scaling : bool, optional
            データをスケーリングするかどうか
        eps : float, optional
            DBSCANのepsパラメータ
        min_samples : int, optional
            DBSCANのmin_samplesパラメータ

        Returns
        -------
        pd.DataFrame
            クラスタリング結果を含むデータフレーム
        """
        if not self._check_data_loaded():
            return None

        # データのスケーリング (オプション)
        if scaling:
            scaler = StandardScaler()
            X_dbscan = scaler.fit_transform(self.X)
        else:
            X_dbscan = self.X.values

        # DBSCAN実行
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_dbscan)

        # 可視化
        self.visualizer.plot_clusters(
            "DBSCAN",
            self.X if not scaling else pd.DataFrame(X_dbscan, columns=self.data_loader.feature_names),
            clusters,
            self.y,
            self.data_loader.feature_names,
            True,
            scaling,
        )

        print("Cluster Memberships:")
        print(clusters)

        # 結果をデータフレームに追加
        df_dbscan = self.X.copy()
        df_dbscan["cluster"] = clusters

        return df_dbscan


# Irisデータセット専用の分析クラス (必要に応じてカスタマイズ可能)
class IrisAnalyzer(DataAnalyzer):
    """Irisデータセット専用の分析クラス"""

    def load_dataset(self) -> pd.DataFrame:
        """
        Irisデータセットを読み込む

        Returns
        -------
        pd.DataFrame
            Irisデータセット
        """
        return super().load_dataset("iris")


# メイン処理
def main():
    """Irisデータセット分析プログラムのメイン処理"""
    print("Irisデータセット分析プログラムを実行します")

    # データ分析オブジェクト作成
    analyzer = IrisAnalyzer(output_dir="output")

    # データセット読み込み
    data = analyzer.load_dataset()
    print("データ読み込み完了")
    print("\n" + str(data.head(20)))  # 最初の20行を表示

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
                analyzer.get_correlation()

            elif choice == 2:
                # seabornを使ってpair_plotする
                analyzer.pair_plot()

            elif choice == 3:
                # ペアプロット対角成分をカーネル密度推定してプロットする
                analyzer.pair_plot(diag_kind="kde")

            elif choice == 4:
                # 複数の教師あり学習モデルを評価する
                n_neighbors = 4
                try:
                    n_input = input("KNeighborsClassifierのn_neighborsを指定してください (デフォルト: 4): ")
                    if n_input.strip():
                        n_neighbors = int(n_input)
                except:
                    print("入力が無効です。デフォルト値(4)を使用します。")

                analyzer.all_supervised(n_neighbors=n_neighbors)

            elif choice == 5:
                # 学習結果の詳細を表示する
                df_scores, stats = analyzer.get_supervised()
                if df_scores is not None:
                    print("\n各モデルの詳細なスコア:")
                    print(df_scores)
                    print("\n各モデルの統計情報:")
                    print(stats)

            elif choice == 6:
                # 最良のモデルを表示する
                best_method, best_score = analyzer.best_supervised()
                if best_method is not None:
                    print(f"\nベストなモデル: {best_method}, スコア: {best_score:.4f}")

            elif choice == 7:
                # 特徴量の重要度を可視化する
                analyzer.plot_feature_importances_all()

            elif choice == 8:
                # 決定木を可視化する
                analyzer.visualize_decision_tree()

            elif choice == 9:
                # データスケーリング方法を比較する
                analyzer.plot_scaled_data()

            elif choice == 10:
                # PCA分析を実行する
                X_scaled, df_pca, pca = analyzer.plot_pca(n_components=2)
                if X_scaled is not None:
                    print("\nスケーリング後のデータ (先頭5行):")
                    print(X_scaled.head())
                    print("\nスケーリング後のデータ統計情報:")
                    print(X_scaled.describe())
                    print("\nPCA成分:")
                    print(pca.components_)

            elif choice == 11:
                # NMF分析を実行する
                X_scaled, df_nmf, nmf = analyzer.plot_nmf(n_components=2)

            elif choice == 12:
                # t-SNE分析を実行する
                analyzer.plot_tsne()

            elif choice == 13:
                # K-means分析を実行する
                analyzer.plot_k_means()

            elif choice == 14:
                # 階層的クラスタリングを実行する
                truncate = input("デンドログラムを切り詰めますか？ (y/n, デフォルト: n): ").lower() == "y"
                analyzer.plot_dendrogram(truncate=truncate)

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

                analyzer.plot_dbscan(scaling=scaling, eps=eps, min_samples=min_samples)

            else:
                print("無効な選択です。0から15の数字を入力してください。")

        except ValueError:
            print("無効な入力です。数字を入力してください。")
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            import traceback

            traceback.print_exc()

        input("\nEnterキーを押して続行...")

    print("\n分析結果は以下のディレクトリに保存されています:")
    print(os.path.abspath(analyzer.file_handler.output_dir))


if __name__ == "__main__":
    main()