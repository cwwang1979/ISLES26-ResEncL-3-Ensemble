# GitHub 上傳步驟

這份資料夾已經保留最終 Docker 的實際程式結構，並移除了原本 ZIP 裡的
test MRI 與 validation output。

## 1. 先建立 GitHub Repository

建議名稱：

`ISLES26-ResEncL-3-Ensemble`

可以設為 Public（若主辦方要求公開）或依主辦方規則設定。

## 2. 不要把三個大型模型直接 git add

真正模型放 GitHub Release assets，repo 的 `model/` 只保留 `README.md`。

最後本機模型仍應解壓成 README 指定的三個原始資料夾名稱。

## 3. 上傳程式碼

在這個資料夾開 Terminal：

```bash
git init
git add .
git commit -m "ISLES26 final ResEncL 3 Ensemble"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ISLES26-ResEncL-3-Ensemble.git
git push -u origin main
```

## 4. GitHub Release

建立一個例如 `final-models` 的 Release，把 M2/M3/M4 模型資源放上去。
之後把 release 下載方式補到 `model/README.md` 與 `scripts/download_models.sh`。

## 5. 最後檢查

- GitHub 裡沒有 ISLES/ATLAS MRI
- GitHub 裡沒有 2–3 GB Docker/model tar
- `inference.py` 仍然是 0.20 / 0.30 / 0.50
- `Dockerfile`、`app.py`、`vendor/` 都保留
- M5/M6/M7 沒有混進 final solution 描述
