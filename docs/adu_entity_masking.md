# Giải thích ADU và các tùy chọn Entity Masking

## 1. Tổng quan thuật toán ADU

`DiffusionADU` kết hợp hai mục tiêu huấn luyện:

```text
L_ADU = auto_weight × L_auto + (1 − auto_weight) × L_gendistill
```

Trong đó:

- **`L_auto` (Auto-unlearning):** làm giảm các dự đoán mà model chỉ có thể đưa ra nhờ thông tin trong query.
- **`L_gendistill` (Generative distillation):** dạy model sinh một câu trả lời thay thế không chứa thông tin cần quên.
- **Teacher model:** model gốc đã freeze, chỉ dùng để tạo target.
- **Student model:** model đang được cập nhật để thực hiện unlearning.

Implementation chính nằm trong:

- `src/trainer/dlm/unlearn/adu.py`
- `src/trainer/dlm/unlearn/repel_gendistill.py`

## 2. Các bước của ADU

### Bước 1: Chọn token cần quên `A_hat`

ADU so sánh xác suất của gold token trong hai trường hợp:

```text
Conditional:   p(token | query, answer đã mask)
Unconditional: p(token | query bị mask, answer đã mask)
```

Nếu xác suất của một answer token tăng mạnh khi có query, token đó được xem là phụ thuộc vào thông tin trong query và có khả năng chứa thông tin cần quên.

```text
score_i = log p_cond(gold_token_i) − log p_uncond(gold_token_i)
```

Các token có score cao được chọn vào tập `A_hat` bằng `topk`, `otsu` hoặc ngưỡng `tau`.

### Bước 2: Tạo secret-free answer

ADU che các token nhận diện trong query, sau đó dùng teacher để sinh một answer thay thế. Answer mới được dùng làm target cho student thay vì bắt student học lại gold answer cần quên.

### Bước 3: Tính Auto-unlearning loss

Teacher dự đoán trên hai context:

```text
log_p_cond   = teacher(query gốc + answer đã noise)
log_p_uncond = teacher(query bị mask + answer đã noise)
```

Phần xác suất được query bổ sung là:

```text
prompt_added = max(log_p_cond − log_p_uncond, 0)
```

ADU tạo target mới bằng cách giảm phần xác suất phụ thuộc vào query:

```text
beta = lam / (1 − lam)
log_q_auto = LogSoftmax(log_p_cond − beta × prompt_added)
```

Student được tối ưu bằng reverse KL:

```text
L_auto = KL(p_student || q_auto)
```

### Bước 4: Tính Generative Distillation loss

Nếu `distill_mode=sft`:

```text
L_gendistill = CrossEntropy(student_logits, secret_free_answer)
```

Nếu `distill_mode=teacher`:

```text
L_gendistill = CrossEntropy(
    phân phối top-k của teacher,
    phân phối của student
)
```

### Bước 5: Kết hợp hai loss

```text
L_ADU = auto_weight × L_auto + (1 − auto_weight) × L_gendistill
```

- `auto_weight=1`: chỉ dùng Auto-unlearning.
- `auto_weight=0`: chỉ dùng Generative Distillation.
- `auto_weight=0.5`: hai loss có trọng số bằng nhau.

## 3. Tổng quan các tùy chọn Entity Masking

| Tùy chọn | Chức năng |
|---|---|
| `query_blank_identity` | Chọn cách phát hiện và che thông tin nhận diện trong query. |
| `keep_answer` | Chọn sinh lại toàn bộ answer hoặc chỉ sinh lại các token nhạy cảm. |
| `keep_subject_name` | Giữ tên subject trong target cuối nhưng giấu tên khỏi teacher trong lúc sinh các token khác. |
| `suppress_gold` | Không cho gold token xuất hiện trong phân phối teacher tại các vị trí `A_hat`. |

## 4. `query_blank_identity`

Tùy chọn này điều khiển cách tạo mask cho query trước khi teacher sinh secret-free answer.

### Khi `query_blank_identity=True`

ADU che nhiều loại token nhận diện:

- Từ riêng viết hoa, ví dụ tên người hoặc địa danh.
- Token chứa số hoặc ngày tháng.
- Từ xuất hiện trong cả query và answer.

Ví dụ:

```text
Query gốc:
What occupation did Wei Jun's father have?

Query sau khi mask:
What occupation did [MASK] [MASK]'s father have?
```

Mục đích là không cho teacher nhìn thấy danh tính thật và dùng danh tính đó để khôi phục secret.

### Khi `query_blank_identity=False`

ADU vẫn mask query, nhưng chỉ mask các từ xuất hiện trong cả query và answer.

Nếu tên `Wei Jun` không xuất hiện trong answer, tên này có thể không bị mask. Vì vậy teacher vẫn có khả năng suy ra thông tin cần quên từ subject name.

### Khuyến nghị

```text
query_blank_identity=True
```

Cấu hình này che identity rộng hơn và an toàn hơn cho unlearning.

## 5. `keep_answer`

Tùy chọn này quyết định phần nào của gold answer được giữ lại khi tạo secret-free target.

### Khi `keep_answer=False`

Toàn bộ answer bị mask và được teacher sinh lại từ đầu:

```text
Gold answer:
Wei Jun's father worked as a disc jockey.

Input cho teacher:
[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]
```

Ưu điểm:

- Loại bỏ gold answer mạnh hơn.
- Teacher có thể tạo một answer hoàn toàn mới.

Nhược điểm:

- Có thể làm mất cấu trúc tốt của answer gốc.
- Có thể giảm relevance hoặc fluency.
- Teacher dễ sinh câu chung chung hoặc không liên quan.

### Khi `keep_answer=True`

ADU chỉ sinh lại:

```text
A_hat ∪ entity names trong answer
```

Các token không nhạy cảm được giữ nguyên.

Ví dụ:

```text
Gold answer:
Wei Jun's father worked as a disc jockey.

Answer được mask một phần:
[MASK] [MASK]'s father worked as a [MASK] [MASK].
```

Ưu điểm:

- Giữ được cấu trúc ngữ pháp của answer.
- Thường giữ relevance và fluency tốt hơn.
- Tập trung thay đổi các token liên quan trực tiếp đến secret.

## 6. `keep_subject_name`

Tùy chọn này chỉ có ý nghĩa khi:

```text
keep_answer=True
```

Giả sử:

```text
Query:
What occupation did Wei Jun's father have?

Gold answer:
Wei Jun's father worked as a disc jockey.
```

### Khi `keep_subject_name=False`

Tên subject trong answer có thể được sinh lại cùng các entity khác:

```text
Secret-free target có thể là:
Alex's father worked as a teacher.
```

Teacher được phép thay cả tên subject và thông tin cần quên.

### Khi `keep_subject_name=True`

Tên subject được xử lý theo ba bước:

1. Tên subject bị mask khỏi input trong lúc teacher sinh answer.
2. Teacher không thể dùng tên đó để suy ra secret.
3. Sau khi sinh xong, tên subject gốc được khôi phục vào target cuối.

Ví dụ:

```text
Input trong lúc generation:
[MASK] [MASK]'s father worked as a [MASK] [MASK].

Target cuối:
Wei Jun's father worked in an unspecified profession.
```

Như vậy, model vẫn có thể nhắc đúng subject nhưng không được liên kết subject đó với thông tin cần quên.

Nếu một token subject cũng thuộc `A_hat`, token đó vẫn được sinh lại thay vì được giữ nguyên.

## 7. `suppress_gold`

Khi `suppress_gold=True`, tại mỗi vị trí thuộc `A_hat`, logit của gold token trong teacher target được đặt thành âm vô cùng:

```text
teacher_logits[position, gold_token] = −∞
```

Sau softmax:

```text
P_teacher(gold_token) = 0
```

Ví dụ, nếu gold token cần quên là `jockey`, teacher không được dùng `jockey` làm distillation target tại vị trí đó.

### Tại sao cần `suppress_gold`?

Dù query đã bị che, teacher vẫn có thể nhớ gold token từ:

- Các token answer xung quanh.
- Một phần answer chưa bị mask.
- Kiến thức đã được ghi nhớ trong trọng số teacher.

`suppress_gold` buộc teacher chuyển xác suất sang các token khác.

### Giới hạn quan trọng

`suppress_gold` chỉ ảnh hưởng trực tiếp khi:

```text
distill_mode=teacher
```

Ở chế độ này, student học từ phân phối logits của teacher.

Khi:

```text
distill_mode=sft
```

student học trực tiếp từ `secret_free_answer`, nên `suppress_gold` không được sử dụng trong loss SFT hiện tại.

## 8. Ví dụ kết hợp các tùy chọn

Cấu hình:

```text
keep_answer=True
query_blank_identity=True
keep_subject_name=True
suppress_gold=True
```

Dữ liệu ban đầu:

```text
Query:
What occupation did Wei Jun's father have?

Gold answer:
Wei Jun's father worked as a disc jockey.
```

Quá trình xử lý:

```text
1. Query bị che identity:
   What occupation did [MASK] [MASK]'s father have?

2. Answer chỉ che entity và token nhạy cảm:
   [MASK] [MASK]'s father worked as a [MASK] [MASK].

3. Teacher sinh phần thay thế mà không nhìn thấy tên Wei Jun.

4. Tên subject được khôi phục:
   Wei Jun's father worked in an unspecified profession.
```

Kết quả mong muốn là giữ được cấu trúc và subject của câu trả lời nhưng xóa liên kết giữa subject với occupation cần quên.

## 9. Cấu hình mặc định cần lưu ý

File YAML `configs/trainer/DiffusionADU.yaml` đặt:

```text
auto_weight=0.5
keep_answer=False
query_blank_identity=True
keep_subject_name=False
suppress_gold=True
```

Tuy nhiên, `scripts/dlm/tofu/adu.sh` override mặc định thành:

```text
AUTO_WEIGHT=0.0
DISTILL_MODE=sft
KEEP_ANSWER=True
QUERY_BLANK_IDENTITY=True
KEEP_SUBJECT_NAME=True
SUPPRESS_GOLD=True
```

Vì vậy, khi chạy `adu.sh` mà không truyền biến môi trường:

- `L_auto` có trọng số bằng `0`.
- Training thực tế chỉ dùng `L_gendistill`.
- Chỉ các token nhạy cảm và entity trong answer được sinh lại.
- Subject name được khôi phục vào target cuối.
- `suppress_gold=True` không ảnh hưởng trực tiếp vì mode hiện tại là `sft`.

Để chạy ADU với hai loss có trọng số bằng nhau:

```bash
AUTO_WEIGHT=0.5 \
KEEP_ANSWER=True \
QUERY_BLANK_IDENTITY=True \
KEEP_SUBJECT_NAME=True \
bash scripts/dlm/tofu/adu.sh llada 0,1,2,3 forget01
```
