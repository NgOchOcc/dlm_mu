# AIR Dataset Review — MUSE (Machine Unlearning Six-Way Evaluation)

## Form Fields

| Field | Value |
|---|---|
| **Project** | AIR Approval Requests (AIR) |
| **Issue Type** | Dataset Review |
| **Summary** | MUSE (Machine Unlearning Six-Way Evaluation) — Books & News |
| **Priority** | Medium |
| **VP Sponsor** | *(fill in your VP sponsor's name)* |
| **Requested Due Date** | *(fill in your target date, format: d/MMM/yy)* |

---

## Description

| **Questions** | **Responses** |
|---|---|
| **General:** | |
| License Type | cc-by-4.0 |
| License Link | https://huggingface.co/datasets/muse-bench/MUSE-Books/blob/main/README.md |
| What do you need to download? | Data and code |
| **Dataset:** | |
| How was the dataset generated? | MUSE (Machine Unlearning Six-Way Evaluation) is a benchmark for evaluating whether an LLM has successfully forgotten specific training data without harming overall performance. It is collected from Harry Potter books and news articles. |
| Where are the dataset authors/proprietors from? | 10 authors affiliated with University of Washington, Princeton University, University of Southern California, University of Chicago, and Google Research. Paper: https://arxiv.org/pdf/2407.06460 |
| Did the dataset author/proprietor generate the dataset themselves? | Yes, in the US |
| Did the dataset author/proprietor take content from other sites or datasets? | Yes — Harry Potter books and news articles |
| Where would the dataset be downloaded from? | Hugging Face (https://huggingface.co/datasets/muse-bench/MUSE-Books and https://huggingface.co/datasets/muse-bench/MUSE-News) |
| Do we need to submit a signature or agreement? | No |
| **Usage:** | |
| What is the business objective? | Internal research idea benchmarking and paper submission |
| Model description | The dataset is used to evaluate the efficiency of an unlearning process for Diffusion Language Models (LLaDA, LLaDA 1.5, Dream). Models are first fine-tuned on MUSE data to obtain a "memorized" checkpoint, then unlearning methods (GradAscent, GradDiff, NPO, SimNPO, WGA) are applied to make the model forget specific book/news content. Evaluation measures forget quality, retain quality, and model utility. |
| Is there a paper associated with the dataset? | Yes — https://arxiv.org/pdf/2407.06460 |
| Will any samples be visible publicly? | If published, yes — samples may appear in the paper. |
| Will you be training any GenAI related models? | Yes — LLaDA (https://arxiv.org/abs/2502.09992), LLaDA 1.5 (https://arxiv.org/abs/2505.19223), Dream 7B (https://arxiv.org/abs/2508.15487) |
| **Attestation questions:** | |
| Dataset does not contain known pirated materials | Confirmed |
| Dataset is not used to create a generative model where output is the same type as dataset data | Confirmed |
| Dataset does not contain geospatial data | Confirmed |
| Dataset does not contain red teaming / security testing data | Confirmed |
| Dataset is hosted on Huggingface, Github, Kaggle, Roboflow, or site without separate ToU | Confirmed |
| Dataset does not require submission of an agreement | Confirmed |

---

## Name of Dataset

MUSE

## Links

- https://huggingface.co/datasets/muse-bench/MUSE-Books
- https://huggingface.co/datasets/muse-bench/MUSE-News

## Country of Origin

United States

## Dataset Provider Location

United States

---

## Item Description

| Item Description | Response |
|---|---|
| 1) Nature of content | Text only; not AI-generated. Consists of book corpora (Harry Potter) and news article corpora. |
| 2) Any personal information | No. News articles may contain names of public individuals mentioned in articles, but the benchmark is not built as a personal information dataset. |
| 3) Data collection process | Researchers collected existing book corpora and news article corpora to create realistic "forget" and "retain" datasets. The benchmark evaluates whether unlearning methods can remove knowledge from the designated forget set while preserving performance on the retain set. |

---

## Training Models

| Training Model | Response |
|---|---|
| 1) If any trained models involve Generative AI (Y/N, explain) | Yes — LLaDA, LLaDA 1.5, and Dream 7B are Diffusion Language Models (masked-diffusion generative text models). |
| 2) Additional context | MUSE is primarily a machine unlearning benchmark. It is used to evaluate whether a model can forget specific book and news content while maintaining overall performance, privacy protection, and model utility. The models are fine-tuned on MUSE data and then subjected to unlearning algorithms; no new generative model is released as a product. |

---

## Additional Fields

| Field | Value |
|---|---|
| **Component/s** | AI Research |
| **License Name** | cc-by-4.0 |
| **License URL** | https://huggingface.co/datasets/muse-bench/MUSE-News/blob/main/README.md |
| **Data Collection Practices** | No known limitations or biases are documented by the dataset authors beyond the scope of the benchmark (book and news corpora used as forget/retain splits for unlearning evaluation). |
| **Employee Location(s)** | Vietnam |
| **Previously Reviewed** | No |
| **Product or Service Incorporation** | No |

---

## Specific Use Case

Internal research purposes including testing and validating models trained for specific use cases, and training models. Specifically: fine-tuning pre-trained Diffusion Language Models (LLaDA, LLaDA 1.5, Dream 7B) on MUSE data to create memorized checkpoints, then applying and benchmarking machine unlearning methods (GradAscent, GradDiff, NPO, SimNPO, WGA). Results are intended for paper submission.

---

## Additional Information

Dataset will be stored on Qualcomm development servers (filer/HPC cluster, e.g., `/prj/corp/airesearch/...`). Access is restricted to the AI Research team members working on this project. Data will be purged upon project completion. No local copies will be made outside of approved Qualcomm infrastructure.

---

## Remaining Fields

| Field | Value |
|---|---|
| **Internal Project Name** | open-unlearning (Diffusion LM Unlearning) |
| **Internal Use Only** | Yes — training models and benchmarking unlearning methods; results intended for paper submission (public disclosure of use checked below). |
| **Public Disclosure of Use** | Yes — paper submission planned. |
| **Is it possible to work from a copy stored on a U.S. server?** | Yes |
| **Demos** | No |
| **Fee** | No |
| **Dataset Origination** | United States |
| **Data Types** | Text |
| **Business Use Case** | Internal R&D / no product or service incorporation; Public Disclosure of Use |
| **Dataset Use Case** | Training; Testing/Benchmarking; Research |

---

## Fields to Fill In Manually

- **VP Sponsor** — your team's VP name
- **Requested Due Date** — your target review date (format: d/MMM/yy)
- **cc: Users / cc: Groups** — your team members or distribution list
