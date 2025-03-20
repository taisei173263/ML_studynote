import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler


class ClusteringAnalysis:
    def __init__(self):
        # データの読み込みと前処理
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.feature_names = self.iris.feature_names
        self.target_names = self.iris.target_names

        # データの標準化
        self.X_scaled = StandardScaler().fit_transform(self.X)

        # PCAによる2次元への削減
        self.pca = PCA(n_components=2)
        self.X_pca = self.pca.fit_transform(self.X_scaled)

    def visualize_original_data(self):
        """元のデータの可視化"""
        plt.figure(figsize=(12, 4))

        # PCAで2次元に削減したデータのプロット
        plt.subplot(121)
        for i, target_name in enumerate(self.target_names):
            mask = self.y == i
            plt.scatter(self.X_pca[mask, 0], self.X_pca[mask, 1], label=target_name)
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.title("PCA of Iris Dataset")
        plt.legend()

        # 分散説明率
        plt.subplot(122)
        explained_variance = self.pca.explained_variance_ratio_
        plt.bar(["PC1", "PC2"], explained_variance)
        plt.title("Explained Variance Ratio")
        plt.ylabel("Ratio")

        plt.tight_layout()
        plt.savefig("output/original_data_visualization.png")
        plt.close()

    def perform_kmeans(self):
        """K-meansクラスタリングの実行と評価"""
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(self.X_scaled)

        # 結果の可視化
        plt.figure(figsize=(8, 6))
        for i in range(3):
            mask = clusters == i
            plt.scatter(self.X_pca[mask, 0], self.X_pca[mask, 1], label=f"Cluster {i}")
        plt.title("K-means Clustering")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.legend()
        plt.savefig("output/kmeans_clustering.png")
        plt.close()

        # 精度の計算
        accuracy = adjusted_rand_score(self.y, clusters)
        return clusters, accuracy

    def perform_hierarchical(self):
        """階層的クラスタリングの実行と可視化"""
        # リンケージ行列の計算
        linkage_matrix = linkage(self.X_scaled, method="ward")

        # デンドログラムの描画
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        plt.savefig("output/hierarchical_clustering.png")
        plt.close()

        return linkage_matrix

    def perform_dbscan(self):
        """DBSCANの実行と評価"""
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(self.X_scaled)

        # 結果の可視化
        plt.figure(figsize=(8, 6))
        # ノイズポイント（-1）は黒で表示
        noise_mask = clusters == -1
        plt.scatter(self.X_pca[noise_mask, 0], self.X_pca[noise_mask, 1], c="black", label="Noise", alpha=0.5)

        # クラスタの表示
        for i in set(clusters) - {-1}:
            mask = clusters == i
            plt.scatter(self.X_pca[mask, 0], self.X_pca[mask, 1], label=f"Cluster {i}")

        plt.title("DBSCAN Clustering")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.legend()
        plt.savefig("output/dbscan_clustering.png")
        plt.close()

        # ノイズを除いた精度の計算
        non_noise_mask = clusters != -1
        accuracy = adjusted_rand_score(self.y[non_noise_mask], clusters[non_noise_mask])
        return clusters, accuracy

    def compare_results(self):
        """各手法の結果を比較・分析"""
        # 各手法の実行
        kmeans_clusters, kmeans_accuracy = self.perform_kmeans()
        hierarchical_matrix = self.perform_hierarchical()
        dbscan_clusters, dbscan_accuracy = self.perform_dbscan()

        # 結果のサマリー
        results = {
            "K-means": {"accuracy": kmeans_accuracy, "n_clusters": len(np.unique(kmeans_clusters))},
            "DBSCAN": {
                "accuracy": dbscan_accuracy,
                "n_clusters": len(np.unique(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0),
                "n_noise": np.sum(dbscan_clusters == -1),
            },
        }

        return pd.DataFrame(results).round(3)


def main():
    # 出力ディレクトリの作成
    import os

    os.makedirs("output", exist_ok=True)

    # 分析の実行
    analysis = ClusteringAnalysis()

    # 元データの可視化
    analysis.visualize_original_data()

    # 各手法の実行と結果の比較
    results = analysis.compare_results()

    # 結果の出力
    print("\n=== クラスタリング手法の比較結果 ===")
    print(results)

    # 結果をファイルに保存
    with open("output/clustering_results.txt", "w", encoding="utf-8") as f:
        f.write("=== Irisデータセットのクラスタリング分析結果 ===\n\n")
        f.write(str(results))
        f.write("\n\n各手法の特徴:\n")
        f.write("1. K-means: 球形クラスタに適しており、計算効率が良い\n")
        f.write("2. 階層的クラスタリング: クラスタの階層構造を視覚化できる\n")
        f.write("3. DBSCAN: 密度ベースで任意形状のクラスタを検出可能\n")


if __name__ == "__main__":
    main()
