# 新北市 YouBike 缺車 / 滿位預警模型報告（Stage 2）

本報告針對「未來 30 分鐘」的兩個二元分類任務：

- **Shortage**（缺車風險）：`future_shortage_30m` = 下一個 30 分鐘 `available_ratio ≤ 0.15`
- **Full**（滿位風險）：`future_full_30m` = 下一個 30 分鐘 `empty_dock_ratio ≤ 0.10`

所有評估嚴格維持時間切分（Train 03-01~03-09、Validation 03-10~03-11、Test 03-12~03-14），未使用隨機切分，threshold 僅由 Validation 決定，Test 不再調整。

---

## 訓練資訊

| 項目 | Shortage | Full |
|---|---|---|
| SageMaker Job | `youbike-xgb-shortage-1789198402` | `youbike-xgb-full-1789198417` |
| Status | Completed | Completed |
| 容器 image | sagemaker-xgboost:1.7-1 | sagemaker-xgboost:1.7-1 |
| Model artifact | s3://ntpc-youbike-hackathon-2026/sagemaker/xgboost-shortage/output/youbike-xgb-shortage-1789198402/output/model.tar.gz | s3://ntpc-youbike-hackathon-2026/sagemaker/xgboost-full/output/youbike-xgb-full-1789198417/output/model.tar.gz |
| scale_pos_weight | 1（未使用） | 33.68（= 637259/18921，由 Train 實算） |

共同超參數：`objective=binary:logistic`、`eval_metric=logloss,auc`、`num_round=100`、`max_depth=5`、`eta=0.1`、`subsample=0.8`、`colsample_bytree=0.8`。Train/Val AUC 差距小（shortage 0.9676/0.9696、full 0.9855/0.9806），無明顯過擬合。

---

## 指標總覽（Test，固定 Validation 決定的 threshold）

| 任務 | 模型 | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Shortage | Persistence baseline | – | 0.817 | 0.817 | 0.817 | 0.941 | 0.818 | 42930 | 9622 | 156481 | 9614 |
| Shortage | XGBoost | 0.20 | 0.722 | 0.904 | 0.803 | 0.960 | 0.892 | 47500 | 18253 | 147850 | 5044 |
| Full | Persistence baseline | – | 0.695 | 0.695 | 0.695 | 0.949 | 0.650 | 4869 | 2132 | 209505 | 2141 |
| Full | XGBoost | 0.70 | 0.417 | 0.861 | 0.562 | 0.978 | 0.792 | 6038 | 8457 | 203180 | 972 |

> 注意：baseline 的 F1 之所以很高，是因為 30 分鐘內站點狀態高度自相關（「下一刻和現在幾乎一樣」）。這是強 baseline，任何模型都必須超越它才有意義。

---

## Early-Warning KPI（Test period）

| 任務 | 模型 | 事件 onset 數 | 預警覆蓋率 | 平均提前 | 中位數提前 | P90 提前 | False Alarm Rate | 總警報數 |
|---|---|---|---|---|---|---|---|---|
| Shortage | XGBoost | 9894 | 38.7% | 40.4 分 | 30 分 | 60 分 | 34.0% | 65414 |
| Shortage | Persistence | 9894 | 20.8% | 30 分 | 30 分 | 30 分 | 27.0% | 52234 |
| Full | XGBoost | 2194 | 34.0% | 48.3 分 | 30 分 | 90 分 | 63.7% | 14346 |
| Full | Persistence | 2194 | 8.4% | 30 分 | 30 分 | 30 分 | 40.2% | 6912 |

- **覆蓋率**：能在事件「發生前一格（30 分鐘前）」就發出警報的 onset 比例。
- **提前量**：對被成功預警的 onset，最早警報距事件發生的時間。
- **False Alarm Rate**：所有警報中，其對應的下一格實際未發生事件的比例。

---

## 九個問題的回答

### 1. XGBoost 是否真的比 Baseline 好？
**部分是，但要看指標。**
- 【模型結果支持】排序能力明確更好：Shortage PR-AUC 0.818→0.892、Full PR-AUC 0.650→0.792；ROC-AUC 同步提升。
- 【模型結果支持】早期預警能力大幅提升：Shortage 覆蓋率 20.8%→38.7%、Full 8.4%→34.0%，且提前量從固定 30 分延長到平均 40~48 分、P90 達 60~90 分。
- 【資料直接證明】但在「單點 F1」上，XGBoost 在營運 threshold 下的 F1 反而略低於 persistence（Shortage 0.803 vs 0.817、Full 0.562 vs 0.695）。這是因為我們刻意把 threshold 調低以換取 recall 與提前量，代價是 precision 下降。
- 結論：XGBoost 在「提前預警」這個真正的業務目標上明顯優於 baseline；但若只看 F1 這種對稱指標，優勢不明顯，甚至因偏好 recall 而略低。

### 2. 哪一個模型比較可靠？
**Shortage 模型比較可靠。** 【模型結果支持】Shortage 在 Test 的 precision 0.722 / recall 0.904，警報大多可信；Full 模型 precision 僅 0.417（約 6 成警報是誤報），可靠度明顯較低。

### 3. Shortage 與 Full 哪個比較難預測？
**Full 明顯較難。** 【資料直接證明】Full 正例僅約 3%，屬高度不平衡；【模型結果支持】即使用了 scale_pos_weight=33.68，Full 的 PR-AUC（0.792）與 F1（0.562）都低於 Shortage（0.892 / 0.803），FAR 也高（63.7% vs 34.0%）。

### 4. 最佳 threshold 是多少？
Threshold 僅由 Validation 決定：
- **Shortage：營運建議 0.20**（數學最高 F1 在 0.40）。理由：交通局需要「提前預警」，recall 優先。0.20 是「precision 仍 ≥ 0.70」的最低門檻，recall 拉到 0.913，兼顧覆蓋與可信度。
- **Full：營運建議 0.70**（數學最高 F1 也在 0.70）。理由：Full 事件稀少，門檻需拉高以壓低誤報；0.70 是「precision ≥ 0.50」中 recall 最高者。即便如此仍有約 56% 誤報，屬本任務的先天限制。

### 5. 平均可以提前多久預警？
【模型結果支持】對被成功預警的事件：Shortage 平均 40.4 分、中位數 30 分、P90 60 分；Full 平均 48.3 分、中位數 30 分、P90 90 分。多數預警提前一格（30 分），少數能提前 1~1.5 小時。

### 6. 哪些特徵最重要？
【模型結果支持】兩個模型都由**目前供給狀態**主導：
- Shortage：`available_ratio`（gain 4080，約次高特徵的 10 倍）遙遙領先，其後為 `empty_dock_ratio`、`available_bikes`、`hour`。
- Full：`empty_dock_ratio`（gain 12113）+ `empty_docks`（4044）主導，其後為 `hour`、`delta_30m_empty_docks`。

依訊號分類：
- **目前供給狀態**（available_ratio / empty_dock_ratio / available_bikes / empty_docks）：**主要訊號**，貢獻絕大部分 gain。
- **近期變化速度**（delta_30m / delta_60m）：次要，提供「正在往缺車/滿位移動」的動能資訊。
- **時間模式**（hour / day_of_week / is_weekend）：`hour` 分裂次數多（weight 高）代表常被用來切分，但單次 gain 不高，屬輔助。
- **站點歷史特性**（station_hist_avg_available_ratio）：中等，提供站點基準線。

【合理推論】模型本質上是「persistence 的非線性強化版」：先看現在狀態，再用變化速度與時間微調。這解釋了為何它能贏 persistence，但幅度有限。

### 7. 是否值得進入 SageMaker Endpoint？
- **Shortage：建議進入。** 排序能力強、營運 threshold 下 recall 0.90 且 precision 0.72，覆蓋率與提前量都明顯優於 baseline，具實用價值。
- **Full：暫不建議直接單獨上線。** 誤報率過高（Test FAR 63.7%），會造成調度端警報疲勞。建議先做改善（見下）再評估。
- 【尚待驗證】上線價值需以「調度成本 / 缺車損失」實際權重驗證，目前僅有統計指標。

### 8. 下一階段如何把模型轉成「站點風險排名」？
【合理推論】每個 30 分鐘週期，對所有站點取模型輸出的機率 `p_shortage`、`p_full`，直接由高到低排序即為風險排名。可再乘上站點權重（容量、歷史需求、是否轉運樞紐）得到加權風險分數，聚焦高衝擊站點。

### 9. 如何把預測結果轉成「調度優先級」？
【合理推論】建議以「風險 × 影響 × 可行動性」組合：
- 缺車調度優先級 = `p_shortage × capacity × 站點重要度`，越高越優先補車。
- 滿位調度優先級 = `p_full × capacity × 站點重要度`，越高越優先清車。
- 用提前量（平均 40~48 分）作為調度車輛的出發前置時間，並用 threshold 控制每班次警報量，避免超出人力。

---

## 證據分級總結

- **【資料直接證明】**：Full 正例約 3%（高度不平衡）；baseline 因 30 分鐘自相關而強；營運 threshold 下 XGBoost 的 F1 未超越 persistence。
- **【模型結果支持】**：XGBoost 在 PR-AUC / ROC-AUC / 覆蓋率 / 提前量上優於 baseline；特徵重要度由現況供給狀態主導。
- **【合理推論】**：模型為 persistence 的非線性強化；風險排名與調度優先級的建構方式。
- **【尚待驗證】**：真實調度成本效益；abrupt-onset 事件為何難以提前預測；加入天氣 / 活動 / 鄰站空間特徵後能否提升覆蓋率。

## 誠實限制
- 兩個模型的**事件 onset 覆蓋率僅 34~39%**：超過六成的「由正常轉為缺車/滿位」的瞬間，模型無法在前一格預見（多為突發性變化）。標準 F1/Recall 看起來漂亮，是因為分母包含大量「已經在缺車狀態、持續缺車」的容易樣本。
- Full 模型誤報偏高，直接上線會造成警報疲勞。
- 特徵目前只用單站自身歷史，缺乏空間（鄰站）、天氣、活動等外生變數，這可能是覆蓋率上不去的主因之一。
