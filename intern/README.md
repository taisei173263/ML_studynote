* * *
​
- [セットアップ](#セットアップ)
  - [ローカル開発環境](#ローカル開発環境)
    - [Pythonのバージョン](#pythonのバージョン)
    - [VSCode](#vscode)
      - [Extension](#extension)
      - [スニペットを使ったヘッダーの挿入](#スニペットを使ったヘッダーの挿入)
    - [リンタ・フォーマッタ](#リンタフォーマッタ)
    - [Git](#git)
  - [AWSとCodeCommit](#awsとcodecommit)
    - [Git for macOS: リポジトリへのアクセスが拒否される場合(403)](#git-for-macos-リポジトリへのアクセスが拒否される場合403)
- [コーディングルール](#コーディングルール)
  - [Docstring](#docstring)
  - [TypeHints](#typehints)
    - [TypeHintsの注意点](#typehintsの注意点)
- [TODO](#todo)
# セットアップ
​
## ローカル開発環境

### Pythonのバージョン

Python >= 3.9.1
​
### VSCode
​
こだわりがなければ[VSCode](https://code.visualstudio.com/download)を推奨します。
​
#### Extension
​
以下のExtensionの追加を推奨します。
各拡張機能のIDが記されているので、Extensionの検索窓で調べてください。
​
| Extension         | Extension ID               |
| ----------------- | -------------------------- |
| black             | ms-python.black-formatter  |
| flake8            | ms-python.flake8           |
| isort             | ms-python.isort            |
| Trailing Spaces   | shardulm94.trailing-spaces |
| Path Autocomplete | ionutvmi.path-autocomplete |
| Python            | ms-python.python           |
| Pylance           | ms-python.vscode-pylance   |
| autoDocstring     | njpwerner.autodocstring    |
​
#### スニペットを使ったヘッダーの挿入
​
[VSCocdeのスニペット機能](https://code.visualstudio.com/docs/editor/userdefinedsnippets#_builtin-snippets)を使用してヘッダーが挿入されるよう設定する。
​
1. VSCodeを開く
1. 左上のCode→基本設定→ユーザースニペットの構成
1. pythonと入力しエンター
1. 以下のauthor, contactを修正し、python.jsonに貼り付ける
    ```json
    {
        "HEADER": {
            "prefix": "header",
            "body": [
                "##",
                "# @file    :   $TM_FILENAME",
                "#",
                "# @brief   :   None",
                "#",
                "# @author  :   Tanaka Taro",
                "# @contact :   t.tanaka@walc.co.jp",
                "# @date    :   $CURRENT_YEAR/$CURRENT_MONTH/$CURRENT_DATE",
                "#",
                "# (C)Copyright $CURRENT_YEAR, WALC Inc.",
                "$0",
            ],
        }
    }
    ```
​
### リンタ・フォーマッタ
​
可読性の高いコードを記述する際に、PEP8に準拠したコードを書くことが要求されます。
自動でコードを整形してくれるFormatter(black)と、文法のチェックを行ってくれるLinter(flake8)をセットアップしましょう。
​
* VSCodeへのセットアップ
​
    * command + shift + p でコマンドパレットを開く
    * Preferences: Open User Settings(JSON)を選択し、以下のjsonを貼り付ける。
```json
{
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {"source.organizeImports": true},
    },
    "black-formatter.args": ["--line-length", "119"],
    "flake8.args": ["--max-line-length=119"],
    "isort.args": ["--profile", "black"],
    "python.analysis.typeCheckingMode": "basic",
}
```
TODO:blackのlength設定がうまく行ってないかも。でも動作する
​
​
​
### Git
​
コードのバージョン管理ツールとしてGitを使用するので、[Git]をインストールする。
​
[Git]: https://git-scm.com/download/mac
​
1.  Gitがインストールされているか確認する。バージョンが返ってきたらOKである。現時点での最新は`2.38.1`である。
    ```bash
    git --version
    ```
​
## AWSとCodeCommit
​
1.  [AWS CLI](https://docs.aws.amazon.com/ja_jp/cli/latest/userguide/getting-started-install.html)をインストールする。
​
1.  Credential の発行依頼をメンターにslackまたはメールで行い、Credential 情報をメールで受取る。access_key と書かれたファイルに Creadential 情報が含まれる。
​
1.  [AWS公式：aws configure を使用したクイック設定](https://docs.aws.amazon.com/ja_jp/cli/latest/userguide/cli-configure-quickstart.html)を参照し、アクセスキーIDとシークレットアクセスキーの設定をAWS CLIで行う。リージョンはap-northeast-1にする。以下はサンプルである。
​
    ```bash
    AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
    AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
    Default region name [None]: ap-northeast-1
    Default output format [None]: json
    ```
​
1.  以下のコマンドを実行してレポジトリ一覧が見れることを確認する。
    ```bash
    aws codecommit list-repositories
    ```
​
1.  [認証情報ヘルパーを使用したセットアップ](https://docs.aws.amazon.com/ja_jp/codecommit/latest/userguide/setting-up-https-unixes.html)のステップ3からを行って、AWS CodeCommitリポジトリへの HTTPS 接続のセットアップを行う。セットアップが終わったら、レポジトリをcloneする。
​
    1.  AWSのドキュメントにも記載されているが、以下のコマンドでレポジトリをクローンする。REPOSITORY_NAMEは`InternIris<AWSIAMユーザー名>`である。AWSIAMユーザー名がSatoの場合、REPOSITORY_NAMEは`InternIrisSato`となる。
        ```bash
        git clone https://git-codecommit.ap-northeast-1.amazonaws.com/v1/repos/REPOSITORY_NAME
        ```
​
    1.  イニシャルブランチの名称をmainに変更する。
        ```bash
        git config --global init.defaultBranch main
        ```
​
    1. [.gitignore](https://git-scm.com/docs/gitignore)について調査し、理解する。またpython用の[.gitignore template](https://github.com/github/gitignore)を参照し追加する。
​
    1.  コードを書き始める。Gitの基本的な使いかたは[サル先生のGit入門](https://backlog.com/ja/git-tutorial/)を参照する。
​
### [Git for macOS: リポジトリへのアクセスが拒否される場合(403)](https://docs.aws.amazon.com/ja_jp/codecommit/latest/userguide/troubleshooting-ch.html#troubleshooting-macoshttps)
​
1.  ターミナルで`git config`コマンドを実行し、Keychain Access ユーティリティが定義されている Git 設定ファイル (gitconfig) を見つけます。ローカルシステムおよび設定によっては、複数のgitconfigファイルが存在する場合がある。
​
    ```bash
    git config -l --show-origin | grep credential
    ```
​
    このコマンドの出力で、次のような結果を検索する。
​
    ```bash
    file:/path/to/gitconfig  credential.helper=osxkeychain
    ```
​
    この行の先頭に示されているファイルが、編集する必要がある Git 設定ファイルである。
​
​
1.  helper = osxkeychain が含まれている認証情報セクションを削除する
    以下のコマンドでgitconfigを開き、パスを削除する。
    ```bash
    nano /path/to/gitconfig
    ```
​
1.  Git を使用して他のリポジトリにアクセスする場合は、CodeCommit リポジトリの認証情報が提供されないように、Keychain Access ユーティリティを設定できます。Keychain Access ユーティリティを設定するには、以下のように行う
​
    1.  Keychain Access ユーティリティを開く。(Finder を使用して位置を指定できます)
​
    1.  git-codecommit.ap-northeast-1.amazonaws.com を検索する。行をハイライト表示し、コンテキスト (右クリック) メニューを開いてから、[Get Info] を選択する。
​
    1.  [Access Control] タブを選択する。
​
    1.  [Confirm before allowing access] で、[git-credential-osxkeychain] を選択し、マイナス記号を選択してリストから削除する。
​
1.  適当な Git コマンドを実行しレポジトリへアクセスができることを確認する。
    例：リモートレポジトリの各ブランチの最新のコミットのIDを見る
    ```bash
    git ls-remote
    ```
​
​
# コーディングルール
​
基本的には[PEP8](https://peps.python.org/pep-0008)に準拠します。VSCodeにFormatterとLinterをセットアップしたので、ここではdocStringとTypeHintについて説明します。
​
## Docstring
​
DocstringにはGoogleスタイル, reStructuredTextスタイル、NumPyスタイルの3つがあります。WALCのでは[Googleスタイル](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)を使用します。
​
​
```python
def func(arg1, arg2, arg3):
  """Hello, func
​
  Lorem ipsum dolor sit amet,
​
  Args:
    arg1 (str): First argument
    arg2 (list[int]): Second argument
    arg3 (dict[str, int]): Third argument
​
  Returns:
    str or None: Return value
​
  Raises:
    ValueError: if arg1 is empty string.
  """
  ```
​
## TypeHints
​
Python3.6からType Hintsという型アノテーションの機能が実装されました。Type Hintsを書くことで、コードの可読性が向上します。またPylanceのような静的解析ツールを使用することで、バグを事前に防ぐことができます。[Python公式ドキュメント](https://docs.python.org/3/library/typing.html)を参考に記述してください。
​
```python
def f(num1: int, my_float: float = 3.5) -> float:
    return num1 + my_float
```
​
### TypeHintsの注意点
​
Python3.9以上ではTypeHintsとして組み込み関数が使用できます。Python3.7, 3.8で開発を行う場合は互換性を担保するために[futureクラス](https://docs.python.org/3.8/library/__future__.html)を使いましょう。例えば、`typing.Union[int,float]`は`int|float`と書けます。また`typing.List[int]`も`list[int]`と書くことができます。
​
​
# TODO
​
* PythonとJupyterの環境構築方法
* Extetionの説明の追加