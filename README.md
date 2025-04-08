# iris.py コード解説

このREADMEでは、`iris.py`の各部分を初心者向けに詳しく解説します。このプログラムはアヤメの花（Iris）のデータセットを分析する機械学習プログラムです。

## 目次
1. [モジュールのインポート](#モジュールのインポート)
2. [設定と定数](#設定と定数)
3. [エラーハンドリングデコレータ](#エラーハンドリングデコレータ)
4. [ファイル操作クラス](#ファイル操作クラス)
5. [データ読み込みと管理クラス](#データ読み込みと管理クラス)
6. [可視化クラス](#可視化クラス)
7. [メインプログラム](#メインプログラム)
8. [使い方](#使い方)
9. [実務での機械学習手法の活用](#実務での機械学習手法の活用)

## モジュールのインポート

```python
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
```

各行の説明:
- `import os`: ファイルやディレクトリ操作のためのモジュール
- `from functools import wraps`: 関数のデコレータを作成するために使用
- `from typing import...`: Pythonの型ヒントを使用するためのモジュール
- `import matplotlib`: グラフを描画するためのライブラリ
- `import numpy as np`: 数値計算のためのライブラリ（npは略称）
- `import pandas as pd`: データフレーム操作のためのライブラリ（pdは略称）
- `matplotlib.use("Agg")`: 画面表示なしでグラフを生成するための設定
- `import matplotlib.pyplot as plt`: グラフ描画用のサブモジュール（pltは略称）
- `plt.close("all")`: すべてのグラフを閉じる（メモリリーク防止）
- `import seaborn as sns`: 統計データ可視化のためのライブラリ（snsは略称）
- `from scipy.cluster.hierarchy import...`: 階層的クラスタリングのための関数
- `from sklearn...`: 機械学習アルゴリズムを提供するscikit-learnライブラリの各種モジュール

## 設定と定数

```python
CONFIG = {
    "output_dir": "output",
    "default_dpi": 100,
    "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    "markers": ["o", "^", "s", "v", "D"],
    "plt_params": {"figure_size": (10, 8), "scatter_size": 100, "alpha": 0.8},
}
```

この辞書では、プログラム全体で使用する設定を定義しています:
- `output_dir`: 画像ファイルの保存先ディレクトリ
- `default_dpi`: 画像の解像度（dots per inch）
- `colors`: グラフで使用する色のリスト（16進数カラーコード）
- `markers`: 散布図で使用するマーカー（点の形）のリスト
- `plt_params`: プロット関連のパラメータ（図のサイズ、点の大きさ、透明度）

## エラーハンドリングデコレータ

```python
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
```

この関数は、デコレータと呼ばれる特殊な関数です:
- デコレータとは、他の関数を包む（デコレートする）ことで、その振る舞いを拡張する関数
- `error_handler`デコレータは、関数で例外が発生した場合にエラーメッセージを表示し、プログラムがクラッシュするのを防ぐ
- `@wraps(func)`は元の関数のメタデータ（関数名、ドキュメント文字列など）を保持するためのデコレータ
- `wrapper`関数は元の関数を呼び出し、例外が発生した場合はそれをキャッチして処理する

## ファイル操作クラス

```python
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
```

このクラスはファイル操作を担当し、主にMatplotlibで作成された図を保存します:
- `__init__`メソッド: クラスのインスタンスを初期化し、出力ディレクトリを作成する
- `save_figure`メソッド: 図をファイルに保存する
  - `filename`: 保存するファイル名
  - `fig`: 保存する図のオブジェクト（省略可能）
  - `dpi`: 解像度（省略可能）
  - 図を保存した後にメモリリークを防ぐために`plt.close()`を呼び出す

## データ読み込みと管理クラス

```python
class DataLoader:
    """データの読み込みと管理を担当するクラス"""

    def __init__(self):
        """データローダーの初期化"""
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = []
        self.target_names = []
```

このクラスの初期化部分では、以下の変数を準備しています:
- `self.data`: 読み込んだデータセット全体
- `self.X`: 特徴量（入力変数）のみのデータ
- `self.y`: ターゲット（目標変数、正解ラベル）
- `self.feature_names`: 特徴量の名前リスト
- `self.target_names`: ターゲットのクラス名リスト

```python
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
```

このメソッドはデータセットを読み込みます:
- `@error_handler`: 上で定義したエラーハンドリングデコレータを適用
- `dataset_name`: 読み込むデータセットの名前
- `**kwargs`: 追加パラメータ（キーワード引数）
- データセット名に応じて異なる読み込みメソッドを呼び出す
- 対応していないデータセット名の場合はエラーを発生させる

```python
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
```

このメソッドは具体的にIrisデータセットを読み込みます:
- `load_iris()`でscikit-learnに組み込まれているIrisデータセットを読み込む
- `feature_names`と`target_names`を取得
- データをpandasのDataFrameに変換し、特徴量、ターゲット、種名の列を含める
- データを特徴量`X`とターゲット`y`に分割

## 可視化クラス

```python
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
```

このクラスはデータの可視化を担当します:
- `file_handler`: 図の保存に使用するFileHandlerインスタンス

```python
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
```

このメソッドは特徴量間の相関行列を可視化します:
- `X`: 相関を計算する特徴量データ
- `filename`: 保存するファイル名
- `X.corr()`で特徴量間の相関係数を計算
- seabornの`heatmap`関数でヒートマップを描画
- 図を指定したファイル名で保存

## メインプログラム

プログラムの最後には、アプリケーションのメイン部分があります:

```python
def main():
    """Irisデータセット分析プログラムのメイン処理"""
    print("Irisデータセット分析プログラムを実行します")

    # データ分析オブジェクト作成
    analyzer = IrisAnalyzer(output_dir="output")

    # データセット読み込み
    data = analyzer.load_dataset()
    print("データ読み込み完了")
    print("\n" + str(data.head(20)))  # 最初の20行を表示

    # メニュー表示と選択肢の処理
    # ...（詳細は省略）

if __name__ == "__main__":
    main()
```

メイン部分の説明:
- `main()`関数: プログラムの主要な処理を行う
- データ分析オブジェクトを作成し、データセットを読み込む
- インタラクティブなメニューを表示して、ユーザーの選択に応じて様々な分析を実行
- `if __name__ == "__main__":`: このファイルが直接実行された場合のみ`main()`関数を呼び出す

## 使い方

このプログラムを実行すると、コンソールにメニューが表示され、以下のような分析が可能です:

1. 変数間の相関係数確認
2. ペアプロット（散布図行列）表示
3. KDE（カーネル密度推定）を使ったペアプロット
4. 複数の教師あり学習モデルを評価（分類器の比較）
5. 学習結果の詳細表示
6. 最良のモデル表示
7. 特徴量の重要度の可視化
8. 決定木の可視化
9. データスケーリング方法の比較
10. PCA（主成分分析）の実行
11. NMF（非負値行列因子分解）の実行
12. t-SNE（t-分布確率的近傍埋め込み）の実行
13. K-meansクラスタリングの実行
14. 階層的クラスタリングの実行
15. DBSCANクラスタリングの実行

各分析の結果は、`output`ディレクトリに画像ファイルとして保存されます。

## 実務での機械学習手法の活用

このセクションでは、`iris.py`で実装されている各機械学習手法について、コード上の実装方法と実務での活用法を併せて解説します。

### 1. 教師あり学習モデル

#### Random Forest（ランダムフォレスト）

**コード実装:**

```python
# iris.pyでのRandomForestの実装例
from sklearn.ensemble import RandomForestClassifier

# モデル定義
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# モデル学習
rf_model.fit(X_train, y_train)

# 予測
y_pred = rf_model.predict(X_test)

# 特徴量重要度の取得
feature_importances = rf_model.feature_importances_
```

**実務活用:**
- **金融分野**: 与信スコアリングや不正検知に利用。大量の取引データから怪しい取引パターンを検出
- **医療分野**: 患者の検査結果からの疾病予測。特徴量重要度を使って重要な検査項目を特定
- **マーケティング**: 顧客離反予測。どの顧客が解約しそうかを予測し、効果的なリテンション戦略を実施

**実装のポイント**:
- `n_estimators`（木の数）は多いほど安定するが計算コストが上がる
- `max_depth`（木の最大深さ）を制限して過学習を防ぐ
- `min_samples_leaf`（葉ノードのサンプル数最小値）を適切に設定して一般化能力を向上
- `feature_importances_`属性を活用して意思決定の根拠を説明

#### Gradient Boosting（勾配ブースティング）

**コード実装:**

```python
# iris.pyでのGradient Boostingの実装例
from sklearn.ensemble import GradientBoostingClassifier

# モデル定義
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# モデル学習
gb_model.fit(X_train, y_train)

# 予測と評価
y_pred = gb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

**実務活用:**
- **広告最適化**: クリック率予測により効率的な広告配信を実現
- **需要予測**: 小売業での在庫管理や売上予測。季節性や特別イベントの影響も考慮
- **保険リスク評価**: 契約者の属性から保険金支払いリスクを予測

**実装のポイント**:
- `learning_rate`（学習率）は小さいほど精度が上がるが、必要な木の数が増加
- 正則化パラメータ（`subsample`, `max_depth`）の調整で過学習を防止
- データ量が少ない場合は交差検証が特に重要

#### Logistic Regression（ロジスティック回帰）

**コード実装:**

```python
# iris.pyでのLogistic Regressionの実装例
from sklearn.linear_model import LogisticRegression

# モデル定義（多クラス分類のため'multinomial'を使用）
lr_model = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42)

# モデル学習
lr_model.fit(X_train, y_train)

# 係数の確認（特徴量の重要度）
coefficients = lr_model.coef_
```

**実務活用:**
- **医療診断**: 検査結果から特定の疾患リスクを予測。確率として解釈しやすい
- **クレジットスコアリング**: 申請者の属性からローン返済能力を予測
- **マーケティング**: キャンペーンのコンバージョン率予測。どの顧客がキャンペーンに反応するか予測

**実装のポイント**:
- 特徴量のスケーリングが重要（StandardScalerなどを使用）
- `C`パラメータ（正則化の強さの逆数）を調整して過学習を防止
- `coef_`属性を使って各特徴量の影響度を解釈
- `predict_proba`メソッドで確率を出力し、閾値を調整して精度と再現率のバランスを取る

#### Support Vector Machine（サポートベクターマシン）

**コード実装:**

```python
# iris.pyでのSVMの実装例
from sklearn.svm import SVC

# モデル定義（RBFカーネル使用）
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)

# モデル学習
svm_model.fit(X_train, y_train)

# 予測
y_pred = svm_model.predict(X_test)
```

**実務活用:**
- **テキスト分類**: 感情分析やスパム検出。高次元の特徴空間でも効果的
- **画像認識**: 小規模～中規模の問題に対応。顔認識など
- **バイオインフォマティクス**: DNA配列の分類やタンパク質構造予測

**実装のポイント**:
- カーネルの選択が重要（linear, rbf, poly, sigmoid）
- `C`と`gamma`パラメータのグリッドサーチでチューニング
- データのスケーリングが非常に重要
- サンプル数が多い場合は線形SVMが計算効率的

### 2. データ変換・次元削減技術

#### PCA（主成分分析）

**コード実装:**

```python
# iris.pyでのPCAの実装例
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# データのスケーリング
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA実行
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 寄与率の確認
explained_variance_ratio = pca.explained_variance_ratio_
```

**実務活用:**
- **画像圧縮**: 顔認識などの前処理で次元削減
- **ノイズ除去**: センサーデータなどのノイズ成分を除去
- **可視化**: 高次元データを2D/3Dで視覚化し、パターンを発見

**実装のポイント**:
- 前処理としてのスケーリングが重要
- `n_components`で保持する主成分数を指定（分散の寄与率に基づいて決定）
- `explained_variance_ratio_`で各主成分の説明力を確認
- 結果の解釈に`components_`（主成分の方向）を活用

#### t-SNE（t-分布確率的近傍埋め込み）

**コード実装:**

```python
# iris.pyでのt-SNEの実装例
from sklearn.manifold import TSNE

# t-SNE実行
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 可視化
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.colorbar()
```

**実務活用:**
- **遺伝子発現データ分析**: 細胞タイプのクラスタリングと可視化
- **自然言語処理**: 単語埋め込みやドキュメント類似性の可視化
- **異常検知**: 正常データと異常データの分離を視覚的に確認

**実装のポイント**:
- `perplexity`パラメータ（近傍点の数に関連）の調整が重要（通常5～50）
- 計算コストが高いため、サンプル数が多い場合は前処理としてPCAを適用
- 同じデータでも乱数シードによって結果が変わるため`random_state`を固定
- 距離の絶対値ではなく相対的な配置が重要

### 3. クラスタリング手法

#### K-means（K平均法）

**コード実装:**

```python
# iris.pyでのK-meansの実装例
from sklearn.cluster import KMeans

# モデル定義
kmeans = KMeans(n_clusters=3, random_state=42)

# クラスタリング実行
clusters = kmeans.fit_predict(X_scaled)

# クラスタ中心の取得
cluster_centers = kmeans.cluster_centers_
```

**実務活用:**
- **顧客セグメンテーション**: 購買行動に基づく顧客グループ分け
- **画像処理**: 色の量子化による画像圧縮
- **市場区分**: 競合製品の特性に基づくマーケットセグメント分析

**実装のポイント**:
- 適切なクラスタ数を決めるためにエルボー法やシルエット分析を活用
- スケーリングが結果に大きく影響するため適切な前処理が重要
- 初期クラスタ中心の設定（`init='k-means++'`がデフォルトで効果的）
- クラスタの解釈に`cluster_centers_`を活用

#### DBSCAN（密度ベースクラスタリング）

**コード実装:**

```python
# iris.pyでのDBSCANの実装例
from sklearn.cluster import DBSCAN

# モデル定義
dbscan = DBSCAN(eps=0.5, min_samples=5)

# クラスタリング実行
clusters = dbscan.fit_predict(X_scaled)

# クラスタラベルとノイズポイント(-1)の確認
unique_clusters = np.unique(clusters)
noise_points = np.sum(clusters == -1)
```

**実務活用:**
- **異常検知**: 通常のパターンから外れたデータポイントを特定
- **地理空間分析**: GPSデータからの関心領域（POI）抽出
- **ネットワーク分析**: ソーシャルネットワークのコミュニティ検出

**実装のポイント**:
- `eps`（近傍の距離閾値）と`min_samples`（コアポイントの最小近傍点数）の調整が重要
- 距離計算の効率化のため`algorithm='ball_tree'`や`leaf_size`の調整
- 多様な密度を持つクラスタを検出するにはHDBSCANも検討
- ノイズポイント（ラベル-1）の分析も重要な情報源

### 4. データスケーリング技術

#### StandardScaler（標準化）

**コード実装:**

```python
# iris.pyでのStandardScalerの実装例
from sklearn.preprocessing import StandardScaler

# スケーラーの定義と適用
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 変換パラメータの確認
mean_values = scaler.mean_
std_values = scaler.scale_
```

**実務活用:**
- **距離ベースアルゴリズムの前処理**: K-means、SVM、KNNなど
- **勾配法を用いる機械学習手法**: 最適化の収束を早める
- **特徴量の影響度を均等化**: 異なるスケールの特徴量を持つデータセット

**実装のポイント**:
- 訓練データでのみ`fit`し、そのパラメータで検証・テストデータも変換
- 外れ値の影響を受けるため、外れ値が多い場合はRobustScalerを検討
- 訓練後のデータの統計（`mean_`, `scale_`）を確認し、期待通りか検証

#### MinMaxScaler（最小最大スケーリング）

**コード実装:**

```python
# iris.pyでのMinMaxScalerの実装例
from sklearn.preprocessing import MinMaxScaler

# スケーラーの定義と適用
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
```

**実務活用:**
- **ニューラルネットワークの入力**: 0-1の範囲で活性化関数の効率が上がる
- **画像処理**: ピクセル値の正規化
- **特徴量エンジニアリング**: 直感的に解釈可能な特徴量の作成

**実装のポイント**:
- 外れ値に非常に敏感なので、事前に外れ値の処理が重要
- `feature_range`パラメータで変換後の範囲を指定可能
- 分布がガウス分布でない特徴量で効果的

### 5. モデル評価とチューニング

**コード実装:**

```python
# iris.pyでのモデル評価の実装例
from sklearn.model_selection import cross_val_score, GridSearchCV

# 交差検証
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
mean_score = cv_scores.mean()

# グリッドサーチ
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_scaled, y)
best_params = grid_search.best_params_
```

**実務活用:**
- **モデル選択**: 複数のアルゴリズムから最適なモデルを選定
- **パラメータチューニング**: ハイパーパラメータの最適化
- **過学習の防止**: トレーニングデータとテストデータのパフォーマンスギャップを監視

**実装のポイント**:
- データ量が限られている場合は交差検証が特に重要
- グリッドサーチは計算コストが高いため、パラメータ空間を絞る
- モデルの評価指標は問題に応じて適切に選ぶ（精度、F1スコア、AUC-ROCなど）
- `n_jobs=-1`で並列処理を活用し計算を高速化

### 実践的ワークフロー

iris.pyでの機械学習プロセスは以下のような実務的なワークフローに沿っています:

1. **データ読み込み・前処理**:
   ```python
   # データセット読み込み
   data_loader = DataLoader()
   data = data_loader.load_data("iris")
   X, y = data_loader.X, data_loader.y
   
   # スケーリング
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **探索的データ分析**:
   ```python
   # 相関分析
   visualizer.plot_correlation(X, "correlation_matrix.png")
   
   # ペアプロット可視化
   visualizer.plot_pair(data, "species", "pairplot.png")
   ```

3. **モデル構築と評価**:
   ```python
   # 複数モデルの評価
   models = {
       "LogisticRegression": LogisticRegression(),
       "RandomForest": RandomForestClassifier(),
       "SVM": SVC()
   }
   
   # 交差検証でモデル評価
   cv = KFold(n_splits=5, shuffle=True, random_state=42)
   results = []
   
   for name, model in models.items():
       scores = cross_val_score(model, X_scaled, y, cv=cv)
       results.append({
           "model": name,
           "mean_score": scores.mean(),
           "std_score": scores.std()
       })
   ```

4. **モデル解釈**:
   ```python
   # 特徴量重要度の可視化
   rf_model = RandomForestClassifier()
   rf_model.fit(X_scaled, y)
   visualizer.plot_feature_importances(rf_model, feature_names)
   
   # 決定木の可視化
   tree_model = DecisionTreeClassifier(max_depth=3)
   tree_model.fit(X, y)
   visualizer.plot_decision_tree(tree_model, feature_names, target_names)
   ```

このような体系的なアプローチは、実務でのデータ分析プロジェクトにも直接適用可能で、機械学習の一連のプロセスを効率的に実行できます。iris.pyの各機能を理解することで、実際のビジネス課題に対しても同様のアプローチで機械学習ソリューションを構築できるようになります。 