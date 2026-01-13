# Contributing Guide

感謝您對 NoC Behavior Model 專案的興趣！本指南說明如何貢獻程式碼。

## Branch 工作流程

### Branch 命名規範

使用以下格式命名分支：

```
{type}/{description}
```

| Type | 用途 | 範例 |
|------|------|------|
| `feature/` | 新功能開發 | `feature/adaptive-routing` |
| `fix/` | Bug 修復 | `fix/deadlock-detection` |
| `refactor/` | 程式碼重構 | `refactor/ni-cleanup` |
| `test/` | 測試相關 | `test/coverage-improvement` |
| `docs/` | 文件更新 | `docs/api-reference` |

### 工作流程

1. **從 main 建立分支**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **進行開發並 commit**
   ```bash
   git add .
   git commit -m "feat(core): add adaptive routing support"
   ```

3. **推送並建立 PR**
   ```bash
   git push -u origin feature/your-feature-name
   gh pr create --title "feat(core): add adaptive routing support"
   ```

4. **Code Review 後合併**

---

## Commit 規範

本專案使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式。

### Commit Message 格式

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Type 類型

| Type | 說明 | 範例 |
|------|------|------|
| `feat` | 新功能 | `feat(router): add west-first routing` |
| `fix` | Bug 修復 | `fix(ni): correct strb mask calculation` |
| `docs` | 文件 | `docs(readme): update installation guide` |
| `style` | 格式化（不影響程式邏輯） | `style: apply black formatting` |
| `refactor` | 重構（不新增功能或修復 bug） | `refactor(buffer): simplify credit logic` |
| `perf` | 效能改善 | `perf(mesh): optimize flit routing` |
| `test` | 測試 | `test(integration): add deadlock tests` |
| `build` | 建置系統 | `build: update pytest to 8.0` |
| `ci` | CI/CD | `ci: add GitHub Actions workflow` |
| `chore` | 維護任務 | `chore: clean up unused imports` |

### Scope 範圍（可選）

| Scope | 涵蓋範圍 |
|-------|----------|
| `core` | router, ni, mesh, buffer |
| `flit` | flit, packet |
| `routing` | routing_selector |
| `testbench` | memory, axi_master, node_controller |
| `verify` | golden_manager, validators |
| `config` | configuration |
| `test` | test infrastructure |

### 範例

```bash
# 新功能
git commit -m "feat(router): implement adaptive routing algorithm"

# Bug 修復
git commit -m "fix(ni): handle empty packet edge case"

# 重構
git commit -m "refactor(core): extract credit flow to separate module"

# 測試
git commit -m "test(integration): add NoC-to-NoC stress tests"

# 文件
git commit -m "docs: add hardware parameters guide"
```

---

## 開發環境設定

### 1. 安裝依賴

```bash
pip install -r requirements.txt
pip install pre-commit
```

### 2. 啟用 Pre-commit Hooks

```bash
pre-commit install
pre-commit install --hook-type commit-msg  # 啟用 commit message 驗證
```

### 3. 執行測試

```bash
# 執行所有測試
py -3 -m pytest tests/ -v

# 執行並顯示覆蓋率
py -3 -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## Pull Request 檢查清單

提交 PR 前請確認：

- [ ] Branch 名稱符合規範
- [ ] Commit messages 使用 Conventional Commits 格式
- [ ] 所有測試通過 (`pytest tests/ -v`)
- [ ] 覆蓋率維持或提升
- [ ] 無 lint 錯誤
- [ ] 更新相關文件（如有必要）

---

## 問題回報

如發現問題，請在 [GitHub Issues](https://github.com/yue831107/noc-behavior-model/issues) 回報，包含：

1. 問題描述
2. 重現步驟
3. 預期行為 vs 實際行為
4. 環境資訊（Python 版本、OS）
