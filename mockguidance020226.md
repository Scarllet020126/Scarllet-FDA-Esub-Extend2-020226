## 3 Mock Application Datasets (Traditional Chinese; JSON)

> 這三份與上方 `app.py` 內 `MOCK_CASES` 一致，便於你獨立使用或另存成 `*.json` 再上傳到「TW Premarket」。

### Mock Dataset A (展延案 / 一般醫材 / 輸入 / 第二等級)
```json
{
  "doc_no": "衛授食藥字第1130001234號",
  "e_no": "MDE-A-0001",
  "apply_date": "2025-01-20",
  "case_type": "許可證有效期限屆至後六個月內重新申請",
  "device_category": "一般醫材",
  "case_kind": "展延案",
  "origin": "輸入",
  "product_class": "第二等級",
  "similar": "有",
  "replace_flag": "否",
  "prior_app_no": "1120719945 變更案",
  "name_zh": "一次性無菌導尿管",
  "name_en": "Sterile Single-Use Urinary Catheter",
  "uniform_id": "24567890",
  "firm_name": "宏遠醫材股份有限公司",
  "firm_addr": "台北市中山區XX路100號10樓",
  "resp_name": "王小明",
  "contact_name": "陳怡君",
  "contact_tel": "02-2345-6789",
  "contact_email": "reg@hongyuan-med.example",
  "manu_name": "ACME Medical Devices Inc.",
  "manu_country": "UNITED STATES",
  "manu_addr": "1234 Device Ave, Irvine, CA, USA"
}
```

### Mock Dataset B (IVD / 第三等級 / 變更案)
```json
{
  "doc_no": "衛授食藥字第1130002222號",
  "e_no": "MDE-B-0002",
  "apply_date": "2025-02-05",
  "case_type": "一般申請案",
  "device_category": "體外診斷器材(IVD)",
  "case_kind": "變更案",
  "origin": "輸入",
  "product_class": "第三等級",
  "similar": "有",
  "replace_flag": "否",
  "prior_app_no": "1110815922",
  "name_zh": "肌酸酐試驗系統",
  "name_en": "Creatinine Assay System",
  "uniform_id": "12345678",
  "firm_name": "新星診斷科技有限公司",
  "firm_addr": "新北市板橋區YY路88號8樓",
  "resp_name": "林志強",
  "contact_name": "黃佩珊",
  "contact_tel": "02-8765-4321",
  "contact_email": "qa@novadiag.example",
  "manu_name": "Nova Diagnostics GmbH",
  "manu_country": "EU (Member State)",
  "manu_addr": "Musterstrasse 1, Berlin, Germany"
}
```

### Mock Dataset C (國產 / 第二等級 / 展延案；多處待補)
```json
{
  "doc_no": "衛授食藥字第1130003333號",
  "e_no": "MDE-C-0003",
  "apply_date": "2025-03-01",
  "case_type": "一般申請案",
  "device_category": "一般醫材",
  "case_kind": "展延案",
  "origin": "國產",
  "product_class": "第二等級",
  "similar": "無",
  "replace_flag": "是",
  "name_zh": "低周波治療器",
  "name_en": "Low Frequency Therapy Device",
  "uniform_id": "87654321",
  "firm_name": "康健醫電股份有限公司",
  "firm_addr": "台中市西屯區ZZ路66號6樓",
  "resp_name": "張雅雯",
  "contact_name": "周承恩",
  "contact_tel": "04-2233-4455",
  "contact_email": "reg@healthtron.example",
  "confirm_match": false,
  "manu_name": "康健醫電股份有限公司（自製）",
  "manu_country": "TAIWAN， ROC",
  "manu_addr": "台中市西屯區ZZ路66號6樓"
}
```

---

## 3 Mock Review Guidance (Traditional Chinese; Markdown)

### Guidance A (展延案形式審查重點)
```markdown
# 展延案件形式審查指引（Mock A）
## 核心原則
- 展延案重點在於：**許可證資訊一致性**、**文件有效性**、**CFS/授權/QMS 仍有效且可追溯**。

## 必檢附件（適用時）
1. 許可證有效期間展延申請書：完整填列與簽章。
2. 原許可證：許可證字號、品名、製造廠、有效期限清楚可辨識。
3. 標籤/中文核定說明書/包裝核定本：版本須與系統一致；如更完整需更新系統檔案。
4. 出產國製售證明（CFS）：需載明出具日期；影本需註明正本留存/可追溯資訊。
5. 原廠授權登記書：需在有效期限內；影本需可追溯正本留存案號。
6. QMS/QSD：需在有效期限內；範圍需涵蓋該製造廠與產品類別。
7. 器材商許可執照：名稱/地址/負責人需與申請資料一致。

## 常見缺失
- 影本未註明正本留存資訊
- 未填出具日期或效期
- 上傳文件為作廢/註銷版本仍被引用
- 標籤/說明書版本不一致
```

### Guidance B (IVD/變更案對照)
```markdown
# IVD/變更/展延相關文件審查重點（Mock B）
## 必查一致性
- 產品中文/英文名稱、製造廠名稱/地址、器材商資訊必須與原許可證一致（除核准變更項目外）。

## 文件有效性
- CFS / 授權 / QMS 需在有效期內，並能清楚追溯出具日期與證書範圍。

## 標籤/說明書
- 若變更警語、適應症、使用限制或關鍵性能聲明，需確認是否需另案變更或重新核定。

## 常見缺漏
- 未附出具日期
- 影本無可追溯資訊（正本留存案號/文件來源）
- 僅上傳作廢版本
- 系統內說明書未更新
```

### Guidance C (國產展延常見缺失)
```markdown
# 國產一般醫材展延文件自檢要點（Mock C）
## 必附/常附文件
- 展延申請書、原許可證、標籤/中文核定說明書或包裝核定本
- 器材商許可執照（製造/販賣）
- QMS/QSD（若法規/等級要求）

## 特別注意
- 若申請資料「確認與器材商證照資訊相符」未勾選，通常會被要求補正或澄清。
- 標籤/說明書若缺，常屬重大缺漏。

## 作廢/註銷
- 上傳文件若為作廢版本，須同時提供有效版本並說明差異與原因。
```

---

## 20 Comprehensive Follow-up Questions

1. 你希望「TFDA 許可證展延」模組支援 **同一個 session 管理多張許可證**（多案件清單/切換），還是每次只處理單一案件？
2. 展延附件的「檔案名稱/檔案說明/註銷/作廢」是否需要符合 TFDA 系統的 **欄位命名與輸出格式**（例如直接對應匯入）？
3. 你是否需要在 UI 中同時支援「民國日期輸入」與「西元日期輸入」，並在匯出時固定為哪一種格式？
4. 對於「正本在 XXXX 案」這類資訊，你希望拆成 **(留存案號 / 留存案型 / 留存機關)** 三欄，或維持單一文字欄位即可？
5. 目前的附件完整性分數只做基本規則；你希望加上 **權重機制**（例如 CFS/授權/QMS 權重更高）嗎？
6. Section 八（委託製造）若改為適用，你需要支援 **多組委託鏈**（多受託廠、多契約、多製程），以及每組各自的 checklist 嗎？
7. 你希望系統對「作廢/註銷」文件做更嚴格規則嗎（例如：若存在作廢文件，必須同時提供有效版本才能算完成）？
8. 是否需要支援「附件文件」的 **到期日/效期** 欄位（除了出具日期）並自動提醒即將到期？
9. 你希望展延模組能從 TW Premarket 申請書自動帶入哪些欄位做一致性檢核（例如公司名稱/地址/製造廠）？
10. 你希望「一致性檢核 Agent」輸出固定格式的 **差異矩陣表**（欄位、申請書值、文件值、結論、建議）嗎？
11. 針對 OCR/抽取，你希望系統能從 PDF 自動抓出「出具日期」「文件號」「公司名稱」並填入 metadata 嗎？
12. 你希望在 Dashboard 增加 **成本預估**（依各模型 token 單價估算）還是保持 token 估算即可？
13. 是否需要「審查歷程」支援 **匯出審查紀錄（run history）**，用於內部稽核或專案留存？
14. 目前剪貼簿是文字層級；你希望升級為 **多 Artifact 管理**（多份摘要/報告/指引的版本庫）嗎？
15. 你希望在 UI 上新增「附件整包」的一鍵輸出：**Summary.md + GapReport.md + Packet.json** 的 zip 下載嗎？
16. 對於 Anthropic/Grok 模型，你有沒有指定可用的 **實際模型 ID 白名單**（避免使用者選到不可用 ID）？
17. 你希望加入「模型自動選擇」策略嗎（例如長文指引自動建議 Gemini Pro，短文建議 GPT mini）？
18. 展延指引（Guidance）你希望系統提供「關鍵詞珊瑚色高亮」與「MUST/SHALL 規則抽取」兩段式處理嗎？
19. 你希望在 TFDA 展延模組加入「附件需求模板管理」：可由你上傳不同版本的 TFDA 清單，系統自動生成 Section 結構嗎？
20. 你預期在 HF Spaces 的實際使用情境：單次上傳檔案的大小上限、平均頁數、同時使用者數量是多少，以便調整效能與超時策略？

21. 20 follow up questions
defaultdataset.json 是否需要支援 版本升級機制（例如 schema_version，讓舊檔自動轉換到新欄位）？
你希望「載入預設案例」是 覆寫所有欄位，還是只填入目前為空的欄位（避免破壞使用者已輸入資料）？
下載 CSV 時，你希望欄位順序固定為 TW_APP_FIELDS 的順序，還是以 JSON key 的自然順序即可？
review_guidance_zh 是否需要拆成兩份：tw_screen_guidance_zh（查驗登記）與 extension_guidance_zh（展延）以避免混用？
你希望 default datasets 也能包含 PDF 原始檔（放在 repo / 或 URL），並在 app 中一鍵載入嗎？
extension_packet_seed.seed_files 中的 issue_date_raw 是否要允許直接填 ISO（YYYY-MM-DD）與民國（111/11/10）混用？
你希望 app 在載入案例時自動產生一份「申請書 Markdown 草稿」與「展延摘要 Markdown」嗎（省一步點按）？
你希望新增一個「Reset to Default Case State」按鈕，用來快速回到剛載入案例時的狀態嗎？
若 defaultdataset.json 檔案不存在，你希望 app 顯示警告、還是靜默使用內建 defaults？
你希望在 sidebar 的案例清單加入 搜尋/篩選（依 tags、等級、國產/輸入）嗎？
你希望下載功能除了 application JSON/CSV，也包含 review_guidance_zh 的 .md 下載嗎？
是否需要提供「下載整包」：application + extension packet seed + guidance 的單一 JSON（或 zip）？
你希望 default datasets 中加入「預期缺漏點」欄位，讓 demo 更像教學（例如 CASE_C 的缺漏原因清單）嗎？
你希望在 UI 顯示每個案例的摘要卡（裝置名稱、等級、案件種類、風險提示）以提升 WOW 感嗎？
application 目前包含較多欄位；你希望提供「最小必填範例」與「完整範例」兩種不同 default case 類型嗎？
展延附件 S8（委託製造）在 default_applicability 中若為「不明」，你希望系統如何計算完整性（算入分母或不算）？
你希望把 default datasets 也用於 510(k) tab 的 demo（例如提供 mock K number、device name）嗎？
是否需要在載入案例後自動把資料送到「剪貼簿」作為下一步 agent 的 input 範本？
你希望 default dataset 的欄位支持英文版本（例如 case_name_en、review_guidance_en）並跟語言切換連動嗎？
你希望未來 default datasets 支援「多筆 records」（例如同一公司多產品），並在 UI 提供多層選擇（公司→產品→案件）嗎？
