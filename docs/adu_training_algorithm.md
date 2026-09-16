# DiffusionADU: Thuật toán huấn luyện

## 1. Mục tiêu

`DiffusionADU` kết hợp hai mục tiêu unlearning cho masked diffusion language model:

1. **Auto-unlearning loss**: làm giảm phần phân phối mà prompt quên bổ sung cho mô hình.
2. **Generative distillation loss**: đưa mô hình học sinh về một câu trả lời không chứa thông tin cần quên do mô hình giáo viên sinh ra.

Loss cho một lần lấy mẫu diffusion mask là:

\[
\mathcal{L}_{ADU}
= \alpha_{auto}\mathcal{L}_{auto}
+ (1-\alpha_{auto})\mathcal{L}_{gendistill}
\]

Trong đó:

- \(\alpha_{auto}\) là `auto_weight`, nằm trong \([0,1]\).
- `auto_weight = 1` chỉ dùng auto-unlearning.
- `auto_weight = 0` chỉ dùng generative distillation.
- Nhánh RBF repulsion của `DiffusionRepelGenDistill` không được sử dụng.

## 2. Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| \(x\) | Chuỗi input sạch gồm prompt và answer |
| \(A\) | Tập vị trí thuộc answer, xác định bởi `labels != -100` |
| \(\hat{A}\) | Các vị trí cần quên được chọn bằng PCSS |
| \(M\) | Các vị trí answer được mask tại timestep hiện tại |
| \(y_t\) | Chuỗi đã được noise tại timestep \(t\) |
| \(p_\theta\) | Phân phối của student model |
| \(p_0\) | Phân phối của frozen teacher model |
| \(y_{gen}\) | Secret-free answer do frozen teacher sinh |
| \(q_{auto}\) | Target distribution của auto-unlearning |
| \(\lambda\) | Mức suppress prompt-dependent probability |
| \(\beta\) | \(\lambda/(1-\lambda)\) |

## 3. Chuẩn bị frozen teacher

Student model ban đầu được sao chép thành một reference model cố định:

```text
teacher <- COPY(student)
FREEZE_ALL_PARAMETERS(teacher)

beta <- lam / (1 - lam)
```

Teacher chỉ tạo target và không nhận gradient.

## 4. Chọn vị trí cần quên bằng PCSS

Với mỗi example, thuật toán ước lượng mức phụ thuộc vào prompt của từng answer token:

\[
s_i = \mathbb{E}_{M}
\left[
\log p_0(x_i \mid x_{prompt}, y_t)
-
\log p_0(x_i \mid x_{empty}, y_t)
\right]
\]

Trong đó `x_empty` là prompt đã được thay bằng mask token.

```text
FUNCTION SELECT_FORGET_POSITIONS(input_ids, answer_mask, attention_mask):
    prompt_mask <- attended non-answer and non-padding positions
    score_sum <- zeros_like(answer_mask)
    sample_count <- zeros_like(answer_mask)

    REPEAT sel_mc TIMES:
        t <- Uniform(time_epsilon, 1)
        mask_probability <- 1 - scheduler(t)
        sampled_mask <- Bernoulli(mask_probability) AND answer_mask
        y_t <- replace sampled_mask positions with MASK

        conditional_log_prob <- teacher(y_t)
        promptless_y_t <- replace prompt_mask positions in y_t with MASK
        unconditional_log_prob <- teacher(promptless_y_t)

        FOR each sampled answer position i:
            score_sum[i] += conditional_log_prob[i, gold_token_i]
                            - unconditional_log_prob[i, gold_token_i]
            sample_count[i] += 1

    score <- score_sum / MAX(sample_count, 1)
    A_hat <- THRESHOLD(score, method = topk | otsu | tau)
    A_hat <- A_hat MINUS special-token positions

    IF drop_question_overlap:
        A_hat <- A_hat MINUS answer words already present in the question

    RETURN A_hat
```

`A_hat` được cache theo chuỗi token sạch để không phải tính lại ở mỗi optimizer step.

## 5. Sinh secret-free answer

Teacher sinh một answer mới sau khi các token nhận diện danh tính trong prompt bị che.

```text
FUNCTION GENERATE_SECRET_FREE(input_ids, answer_mask, A_hat):
    identity_mask <- identity tokens in the prompt

    IF keep_answer:
        answer_name_mask <- entity-name positions inside the answer
        fill_mask <- (A_hat OR answer_name_mask) AND answer_mask

        IF keep_subject_name:
            subject_mask <- answer words overlapping the prompt
            hidden_subject <- subject_mask MINUS A_hat
            fill_mask <- fill_mask MINUS hidden_subject
    ELSE:
        fill_mask <- answer_mask

    state <- input_ids
    state[identity_mask OR fill_mask] <- MASK

    REPEAT gen_steps TIMES:
        remaining <- masked answer positions excluding hidden_subject
        STOP globally when no distributed rank has remaining positions

        logits <- teacher(state)
        suppress EOS, padding, MASK, and invalid vocabulary tokens
        prediction <- ARGMAX(logits)
        confidence <- MAX(SOFTMAX(logits))

        reveal the highest-confidence fraction of remaining positions

    IF keep_subject_name:
        restore hidden subject tokens from the original answer

    RETURN state as y_gen
```

`y_gen` cũng được cache theo example.

## 6. Tạo context dùng chung

Context được tạo một lần cho mỗi batch và dùng lại cho tất cả `loss_mc` mask draws.

```text
FUNCTION BUILD_CONTEXT(student, forget_batch):
    input_ids, labels, attention_mask <- PREPROCESS(forget_batch)
    answer_mask <- labels != -100

    A_hat, y_gen <- LOAD_OR_BUILD_CACHE(input_ids, answer_mask)

    identity_mask <- identity tokens in prompt
    anchor_input <- input_ids
    anchor_input[identity_mask] <- MASK
    anchor_input[answer_mask] <- y_gen[answer_mask]

    teacher_anchor_logits <- teacher(anchor_input)

    IF suppress_gold:
        teacher_anchor_logits[A_hat, gold_token] <- -infinity

    RETURN {
        input_ids,
        attention_mask,
        answer_mask,
        A_hat,
        y_gen,
        teacher_anchor_logits
    }
```

## 7. Lấy mẫu diffusion state

```text
FUNCTION SAMPLE_DIFFUSION_STATE(context):
    t <- Uniform(time_epsilon, 1) for each example
    mask_probability <- 1 - scheduler(t)
    M <- Bernoulli(mask_probability) AND answer_mask
    y_t <- replace positions M in input_ids with MASK
    RETURN M, y_t
```

Student chỉ forward một lần trên `y_t` cho mỗi mask draw.

## 8. Auto-unlearning loss

Teacher được chạy trên cùng trạng thái answer đã noise, với hai loại context:

```text
conditional_input   <- y_t with the original prompt visible
unconditional_input <- y_t with prompt positions replaced by MASK

log_p_cond   <- LOG_SOFTMAX(teacher(conditional_input))[M]
log_p_uncond <- LOG_SOFTMAX(teacher(unconditional_input))[M]
```

Phần xác suất được bổ sung bởi prompt:

\[
\Delta_+(v)
= \max\left(0, \log p_{cond}(v)-\log p_{uncond}(v)\right)
\]

Target auto-unlearning:

\[
\log q_{auto}(v)
= \operatorname{LogSoftmax}
\left(\log p_{cond}(v)-\beta\Delta_+(v)\right)
\]

Student được tối ưu bằng reverse KL tại các vị trí mask:

\[
\mathcal{L}_{auto}
= \operatorname{MeanSequence}
\left[
D_{KL}\left(p_\theta(\cdot\mid y_t)\;||\;q_{auto}\right)
\right]
\]

```text
FUNCTION AUTO_LOSS(student_logits, y_t, M):
    promptless_y_t <- replace prompt positions in y_t with MASK

    WITH NO_GRAD:
        log_p_cond <- teacher_log_probs(y_t)[M]
        log_p_uncond <- teacher_log_probs(promptless_y_t)[M]
        prompt_added <- MAX(log_p_cond - log_p_uncond, 0)
        log_q_auto <- LOG_SOFTMAX(log_p_cond - beta * prompt_added)

    log_p_student <- LOG_SOFTMAX(student_logits[M])
    token_loss <- SUM(EXP(log_p_student) * (log_p_student - log_q_auto))
    RETURN mean token loss per active sequence
```

## 9. Generative distillation loss

### SFT mode

```text
position_loss <- CROSS_ENTROPY(
    student_logits[answer_mask],
    y_gen[answer_mask]
)
```

### Teacher mode

```text
top_values, top_indices <- TOP_K(
    teacher_anchor_logits[answer_mask],
    distill_topk
)
target_probability <- SOFTMAX(top_values)

student_log_probability <- LOG_SOFTMAX(student_logits[answer_mask])
selected_student_log_probability <- GATHER(student_log_probability, top_indices)

position_loss <- -SUM(
    target_probability * selected_student_log_probability
)
```

Các vị trí thuộc `A_hat` có thể được tăng trọng số:

```text
weight[i] <- loss_d_ahat_weight if i belongs to A_hat else 1
L_gendistill <- SUM(position_loss * weight) / SUM(weight)
```

## 10. Một mask draw của ADU

```text
FUNCTION ADU_DRAW(student, context):
    M, y_t <- SAMPLE_DIFFUSION_STATE(context)
    student_logits <- student(y_t)

    IF auto_weight > 0:
        L_auto <- AUTO_LOSS(student_logits, y_t, M)
    ELSE:
        L_auto <- 0

    IF auto_weight < 1:
        L_gendistill <- GENDISTILL_LOSS(student_logits, context)
    ELSE:
        L_gendistill <- 0

    RETURN auto_weight * L_auto
           + (1 - auto_weight) * L_gendistill
```

Hai loss dùng chung student forward và cùng diffusion mask `M`. GenDistill được tính trên toàn bộ answer span, còn Auto loss chỉ được tính tại các answer positions đang bị mask.

## 11. Toàn bộ training step

```text
FUNCTION ADU_TRAINING_STEP(student, batch):
    context <- BUILD_CONTEXT(student, batch.forget)
    forget_loss <- 0

    REPEAT loss_mc TIMES:
        draw_loss <- ADU_DRAW(student, context)
        forget_loss <- forget_loss + draw_loss / loss_mc

    retain_loss <- OPTIONAL_RETAIN_NLL(student, batch.retain)
    total_loss <- gamma * forget_loss + alpha * retain_loss

    BACKPROPAGATE(total_loss)
```

Khi `loss_mc > 1`, implementation backward từng draw ngay lập tức để chỉ giữ một student computation graph trong bộ nhớ. Trong chế độ này, code hiện tại không áp dụng optional retain loss.

## 12. Tham số chính

| Tham số | Mặc định | Vai trò |
|---|---:|---|
| `auto_weight` | `0.5` | Cân bằng Auto loss và GenDistill loss |
| `lam` | `0.5` | Mức loại bỏ ảnh hưởng do prompt bổ sung |
| `select_method` | `topk` | Phương pháp chọn `A_hat` |
| `topk_frac` | `0.8` | Tỷ lệ answer positions được chọn bởi top-k |
| `sel_mc` | `16` | Số mask draws để ước lượng PCSS |
| `distill_mode` | `sft` | Chọn hard target hoặc soft teacher target |
| `distill_topk` | `64` | Số token giữ lại trong teacher mode |
| `gen_steps` | `128` | Số bước sinh secret-free answer |
| `loss_d_ahat_weight` | `1.0` | Trọng số GenDistill tại `A_hat` |
| `loss_mc` | `1` | Số diffusion mask draws trên mỗi step |
| `gamma` | `1.0` | Trọng số tổng của forget loss |

## 13. Luồng tổng quát

```text
forget batch
    |
    +--> PCSS selection ----------> A_hat --------+
    |                                             |
    +--> secret-free generation --> y_gen --------+--> shared context
                                                  |
                                                  v
                                         sample diffusion mask
                                                  |
                                                  v
                                             student forward
                                             /             \
                                            v               v
                                       Auto loss      GenDistill loss
                                            \               /
                                             \             /
                                              v           v
                                                ADU loss
                                                   |
                                                   v
                                               backward
```
