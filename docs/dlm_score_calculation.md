# Cách tính điểm tổng hợp cho TOFU DLM

Tài liệu này giải thích ngắn gọn cách `scripts/dlm/show_results.py` tính bốn
điểm cuối cùng:

- **EQ (Erase Quality):** model quên dữ liệu cần xóa tốt đến mức nào.
- **MU (Model Utility):** model giữ lại kiến thức hữu ích tốt đến mức nào.
- **LQ (Linguistic Quality):** câu trả lời có trôi chảy hay không.
- **AVG:** điểm trung bình cuối cùng của EQ, MU và LQ.

Tất cả các điểm cuối đều nằm trên thang **0–1**, được hiển thị với hai chữ số
thập phân và **càng cao càng tốt**.

## 1. Ý nghĩa các tiêu chí nhỏ

| Tiêu chí | Mô tả |
|---|---|
| `forget_prob` | Xác suất model gán cho đáp án cần quên. Càng thấp càng tốt vì model càng ít nhớ đáp án. |
| `forget_rouge` | Mức giống nhau giữa câu trả lời của model và đáp án cần quên. Càng thấp càng tốt vì model càng ít tái tạo thông tin cũ. |
| `forget_truth_ratio` | Mức độ model không còn phân biệt rõ đáp án đúng với đáp án sai trên tập forget. Giá trị hiện tại càng cao càng tốt cho việc quên. |
| `forget_relevance` | LLM judge đánh giá câu trả lời có từ chối hoặc phản hồi đúng trọng tâm câu hỏi cần quên hay không. Càng cao càng tốt. |
| `forget_correctness` | LLM judge đánh giá câu trả lời còn đúng với đáp án cần quên hay không. Càng thấp càng tốt vì trả lời đúng nghĩa là model vẫn còn nhớ. |
| `retain_prob` | Xác suất model gán cho đáp án đúng trên dữ liệu retain. Càng cao càng tốt. |
| `retain_rouge` | Mức giống nhau giữa câu trả lời và đáp án đúng trên dữ liệu retain. Càng cao càng tốt. |
| `ra_rouge` | Khả năng trả lời đúng câu hỏi thuộc tập Real Authors (RA). Càng cao càng tốt. |
| `wf_rouge` | Khả năng trả lời đúng câu hỏi thuộc tập World Facts (WF). Càng cao càng tốt. |
| `forget_fluency` | LLM judge đánh giá độ trôi chảy, rõ ràng và tự nhiên của câu trả lời trên tập forget. Càng cao càng tốt. |
| `retain_fluency` | Độ trôi chảy của câu trả lời trên tập retain. Càng cao càng tốt. |
| `ra_fluency` | Độ trôi chảy của câu trả lời trên tập Real Authors. Càng cao càng tốt. |
| `wf_fluency` | Độ trôi chảy của câu trả lời trên tập World Facts. Càng cao càng tốt. |
| `retain_relevance`, `retain_correctness` | Độ liên quan và độ đúng trên tập retain. Chưa được dùng vì chưa có kết quả đầy đủ cho tất cả model. |
| `ra_relevance`, `ra_correctness` | Độ liên quan và độ đúng trên tập Real Authors. Chưa được dùng vì chưa có kết quả đầy đủ cho tất cả model. |
| `wf_relevance`, `wf_correctness` | Độ liên quan và độ đúng trên tập World Facts. Chưa được dùng vì chưa có kết quả đầy đủ cho tất cả model. |

## 2. Chuẩn hóa về thang 0–1

Các metric Probability, ROUGE và Truth Ratio ban đầu nằm trong `[0, 1]`.
Các metric LLM judge nằm trong `[0, 10]`.

```text
Metric [0, 1], càng cao càng tốt: score = value
Metric [0, 1], càng thấp càng tốt: score = 1 − value
LLM judge, càng cao càng tốt:    score = value / 10
LLM judge, càng thấp càng tốt:   score = (10 − value) / 10
```

## 3. Harmonic Mean

Harmonic Mean (`HM`) của các giá trị được tính bằng:

```text
HM(x1, x2, ..., xn) = n / (1/x1 + 1/x2 + ... + 1/xn)
```

`HM` phạt mạnh khi một thành phần quá thấp. Vì vậy, model phải đạt kết quả cân
bằng ở các metric thành phần thì điểm tổng hợp mới cao. Nếu một thành phần bằng
`0`, kết quả `HM` bằng `0`.

## 4. Erase Quality (EQ)

### Bước 1: Chuẩn hóa các metric quên

```text
ForgetProbScore        = 1 − forget_prob
ForgetRougeScore       = 1 − forget_rouge
ForgetTruthRatioScore  = forget_truth_ratio
ForgetRelevanceScore   = forget_relevance / 10
ForgetIncorrectScore   = (10 − forget_correctness) / 10
```

`forget_prob`, `forget_rouge` và `forget_correctness` được đảo chiều vì model
càng ít nhớ và càng ít trả lời đúng thông tin cần quên thì càng tốt.

### Bước 2: Tính điểm quên kiến thức

```text
KnowledgeErasure = HM(
    ForgetProbScore,
    ForgetRougeScore,
    ForgetTruthRatioScore
)
```

### Bước 3: Tính điểm LLM judge cho việc quên

```text
JudgeErasure = Average(
    ForgetRelevanceScore,
    ForgetIncorrectScore
)
```

Nhóm này dùng trung bình cộng để một metric bằng `0` không làm toàn bộ điểm
Judge Erasure bằng `0`.

### Bước 4: Tính EQ

```text
EQ = HM(KnowledgeErasure, JudgeErasure)
```

**Ý nghĩa:** EQ cao khi model vừa mất khả năng tái tạo kiến thức cần quên, vừa
đưa ra câu trả lời phù hợp và không tiết lộ đáp án đúng.

## 5. Model Utility (MU)

### Bước 1: Chuẩn hóa các metric utility

```text
RetainProbScore   = retain_prob
RetainRougeScore  = retain_rouge
RARougeScore      = ra_rouge
WFRougeScore      = wf_rouge
```

### Bước 2: Tính utility của tập retain

```text
RetainUtility = HM(RetainProbScore, RetainRougeScore)
```

### Bước 3: Tính MU

```text
MU = HM(
    RetainUtility,
    RARougeScore,
    WFRougeScore
)
```

**Ý nghĩa:** MU cao khi model vẫn hoạt động tốt trên cả dữ liệu retain, Real
Authors và World Facts sau khi unlearning.

Hiện tại các metric LLM judge của retain, RA và WF chưa được đưa vào MU vì chưa
có kết quả đầy đủ cho tất cả model. Việc bỏ qua được áp dụng giống nhau cho mọi
model để đảm bảo so sánh công bằng.

## 6. Linguistic Quality (LQ)

### Công thức

```text
LQ = HM(
    forget_fluency / 10,
    retain_fluency / 10,
    ra_fluency / 10,
    wf_fluency / 10
)
```

**Ý nghĩa:** LQ cao khi câu trả lời trên cả forget, retain, Real Authors và
World Facts đều rõ ràng, tự nhiên và trôi chảy.

LQ chỉ được tính khi có đủ cả bốn metric. Nếu thiếu bất kỳ metric nào, LQ và AVG
sẽ hiển thị `-` để mọi model được so sánh bằng cùng một công thức.

## 7. Điểm trung bình cuối cùng (AVG)

EQ, MU và LQ có trọng số bằng nhau:

```text
AVG = (EQ + MU + LQ) / 3
```

**Ý nghĩa:** AVG cao khi model đạt cân bằng giữa ba mục tiêu: quên tốt, giữ được
kiến thức hữu ích và tạo câu trả lời có chất lượng ngôn ngữ tốt.

## 8. Tóm tắt

| Điểm lớn | Công thức hiện tại | Mục tiêu |
|---|---|---|
| `EQ` | `HM(KnowledgeErasure, JudgeErasure)` | Quên dữ liệu cần xóa, càng cao càng tốt. |
| `MU` | `HM(RetainUtility, RARougeScore, WFRougeScore)` | Giữ kiến thức hữu ích, càng cao càng tốt. |
| `LQ` | `HM(forget_fluency / 10, retain_fluency / 10, ra_fluency / 10, wf_fluency / 10)` | Câu trả lời trôi chảy trên cả bốn tập, càng cao càng tốt. |
| `AVG` | `(EQ + MU + LQ) / 3` | Điểm tổng hợp cuối cùng, càng cao càng tốt. |
