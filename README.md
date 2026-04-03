# 競馬GB指数 予測Webアプリ

iPhoneのSafariから使える競馬GB指数計算ツール。

## デプロイ済みURL
※ Renderデプロイ後にここにURLを記載

---

## Render.comへのデプロイ手順

### 1. GitHubにリポジトリを作成してアップロード

```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/あなたのID/keiba-gb-index.git
git push -u origin main
```

### 2. Render.comでデプロイ

1. https://render.com でGitHubアカウントでサインアップ
2. ダッシュボードで「New +」→「Web Service」
3. 先ほどのGitHubリポジトリを選択
4. 設定：
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --timeout 300 --workers 1`
5. 「Create Web Service」→ 5〜10分でURL発行

### 3. iPhoneのホーム画面に追加

1. SafariでRenderのURLを開く
2. 下部の共有ボタン（四角＋矢印）をタップ
3. 「ホーム画面に追加」→ アプリとして使える

---

## ファイル構成

```
keiba_app/
├── app.py                   ← Flaskサーバー
├── requirements.txt         ← 依存ライブラリ
├── render.yaml              ← Render設定
├── templates/
│   └── index.html           ← スマホ対応UI
└── models/
    ├── turf_v2_config.json  ← 芝v2設定
    ├── dart_v2_config.json  ← ダートv2設定
    ├── turf_v2_train.npz    ← 芝v2学習データ
    └── dart_v2_train.npz    ← ダートv2学習データ
```

## 注意

- 無料プランは15分無操作でスリープ（再アクセスで1〜2分待ち）
- 初回起動時にモデル自動学習（3〜5分）
- 馬券購入は自己責任でお願いします
