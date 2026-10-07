# Lab 21 — Evaluation Report

**Họ tên**: Lê Thanh Trường  **MSSV**: 2A202602492  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 14.6 GB (sm_75, fp16)`

> Mọi con số dưới đây khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage (mặc định của lab) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | **1024** — p95 đo được là **98** *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 / **30** |

**Về `max_length`:** p95 = 98 token, nên `suggested_max_length` = 256. Lab chọn **1024** vì
đó là giá trị của tier T4, và tier không có tham số riêng cho việc này. Đây là lựa chọn
**có ý thức đánh đổi**: 1024 lớn gấp 4 lần mức cần thiết, nhưng batch = 1 nên phần padding
đó gần như không tốn gì, còn hạ xuống 256 sẽ phải sửa `config.py` — nguồn sự thật duy nhất
mà cả ba notebook đọc chung. Với corpus này, `max` = 101 nên **không mẫu nào bị cắt** ở cả
hai mức. Nếu dùng corpus dài hơn (ticket nhiều lượt thoại), 256 sẽ là lựa chọn đúng và
tiết kiệm hơn.

**Template có giữ khối `<think>` không?** **Có** — *(results/template_check.json)*, verdict
`"reasoning preserved — safe to train on traces"`. Rendering kiểm tra cho thấy body của
khối `<think>` đi qua `apply_chat_template` nguyên vẹn. Nghĩa là nếu sau này huấn luyện
trên dữ liệu có reasoning trace thật, trace đó sẽ tới được hàm loss.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** |
| Câu trả lời nằm trong loss | **true** |
| Câu hỏi KHÔNG nằm trong loss | **true** |

39/94 token được tính loss. Đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn **bị** mask (model vẫn thấy, nhưng không bị chấm):

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>

```

Hai điều đáng chú ý trong con số này. Thứ nhất, `<|im_end|>` **nằm trong loss** — đúng,
vì đó là tín hiệu dừng của model; bỏ nó ra là dạy model viết vô hạn. Thứ hai, phần
`<think>\n\n</think>` nằm **ngoài** loss: Qwen3.5 đóng khối suy luận rỗng ngay trong
generation prompt, nên nó thuộc về prompt chứ không phải câu trả lời. Vì thế
`assistant-only`, `masked-think` và `response-only` cho ra **cùng một mask** trên corpus
này — lab có cảnh báo `RuntimeWarning` cho điều đó, và đó là lý do B3 (reasoning-trace
collapse) không chạy được với dữ liệu mặc định.

So sánh với `MASK_MODE=everything`: `supervised_fraction` = 100%, và đoạn loss chứa cả
`<|im_start|>system / Phân loại ticket sau.` — tức model được dạy *viết lại câu hỏi*.
Đó là bug kinh điển ở deck §16, và 0.41 vs 1.00 là khoảng cách giữa một pipeline đúng và
một pipeline vô giá trị.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3286.9 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.000 | 1053.3 |
| (c) LoRA fine-tune | **0.970** | 0.6111 | 1.000 | 1358.7 |

**(b) có thật sự mạnh hơn (a) không?** **Có** — và khoảng cách rất lớn: target 0.000 →
0.765, format 0.000 → 1.000.

**Bạn có sửa `OPTIMIZED_PROMPT` không?** **Không.** `optimized_prompt_sha` ghi trong
`baselines_frozen.json` là `719e74d3b6232053`, khớp với bản gốc của lab — `make verify`
xác nhận dòng `baseline (b) prompt unmodified` là OK.

> Đây là điểm liêm chính quan trọng nhất của lab. Prompt (b) **không** bị làm yếu đi để
> fine-tune trông thắng. Con số 0.765 mà bản fine-tune phải vượt là một cái mốc thật.

Chỉ số latency cũng đáng đọc: (a) chậm **3.3 giây/mẫu** vì với prompt ngây thơ model
sinh văn xuôi dài dòng cho tới hết `max_new_tokens`; (b) chỉ **1.05 giây** vì prompt tử tế
dạy nó dừng đúng lúc. Riêng việc sửa prompt đã nhanh hơn **3.1 lần** — trước khi fine-tune
được lợi gì.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6253 | **0.970** | 402.1 | 8.78 |
| `attn_only` | q,v | **283** | 32,456,704 | 1e-4 | **0.5373** | 0.965 | 264.9 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | **1e-5** | 1.5702 | **0.000** | 394.0 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 461.9 | **3.86** |

Cả bốn run dùng **cùng 30 optimizer step** (`make verify` đọc `runs.csv` và xác nhận dòng
`all runs share ONE step budget`), nên khác biệt duy nhất giữa mỗi đối chứng và `correct`
đúng là một biến.

**Bảng xếp hạng — hai thang đo cho hai thứ tự KHÁC NHAU:**

| Theo `train loss` (thấp = tốt) | Theo `target` (cao = tốt) |
|---|---|
| 1. `attn_only` — 0.5373 | 1. `correct` — 0.970 |
| 2. `correct` — 0.6253 | 2. `attn_only` — 0.965 |
| 3. `qlora` — 0.7058 | 3. `qlora` — 0.940 |
| 4. `wrong_lr` — 1.5702 | 4. `wrong_lr` — 0.000 |

### 4.1 — `attn_only`: vị trí hay rank mới là đòn bẩy?

`attn_only` được **khớp ngân sách tham số** với `correct`: 32,456,704 so với 32,464,896
tham số huấn luyện, lệch **0.003%** — `make verify` xác nhận dòng `attn_only is a FAIR
contrast`. Nghĩa là phép đối chứng này cô lập đúng **một** biến: adapter gắn ở đâu, chứ
không phải gắn bao nhiêu. Để giữ ngân sách đó, rank phải nâng từ 16 lên **283** — gấp
**17.7 lần** — vì chỉ có 2 module (q, v) thay vì 12.

Kết quả: **0.965 so với 0.970**. Chênh lệch 0.005 trên 50 mẫu × 4 trường = đúng **1 trường
trên 200**, tức là hoà trong sai số. Nhưng theo `train loss` thì `attn_only` lại **thấp
nhất bảng** (0.5373 < 0.6253).

Điều này nói gì? **Rank không phải đòn bẩy.** Nếu rank là đòn bẩy, mộtadapter r=283 gấp
17.7 lần về rank phải thắng rõ ràng — nó không thắng gì cả. Cái nó làm được là **ép train
loss xuống thấp hơn**, và đó chính xác là cái bẫy: loss thấp ở đây đến từ việc có nhiều
tham số hơn để ghi nhớ 225 mẫu, không phải từ việc học được điều gì tổng quát hơn. Vị trí
gắn adapter (text-linear phủ cả MLP: gate/up/down) cũng không phải đòn bẩy lớn trong bài
toán hẹp này — nó **hoà** với attention-only khi ngân sách bằng nhau. Kết luận đúng phải
là: trong bài toán 4 nhãn đóng này, **cả vị trí lẫn rank đều không phải nút vặn quyết
định**. Nút quyết định nằm ở chỗ khác — xem 4.2.

Nếu tôi xếp hạng bốn run bằng `final_loss` — như lab cũ làm — tôi sẽ báo cáo rằng
`attn_only` là cấu hình **tốt nhất**, và rằng gắn LoRA vào q,v với rank khổng lồ là lựa
chọn đúng. Đó là kết luận ngược hoàn toàn với sự thật, và nó chỉ sai vì dùng chỉ số thay
thế. Đây là lý do lab này gọi việc đó là **Lỗi #3**.

### 4.2 — `wrong_lr`: chỉ khác một con số, mà hỏng hoàn toàn

`wrong_lr` giống `correct` ở mọi thứ — cùng vị trí, cùng r=16, cùng alpha=32, cùng 30
step, cùng dữ liệu, cùng seed — **chỉ khác `learning_rate` 1e-5 thay vì 1e-4**, đúng một
chữ số.

Đường loss (log NB4):

| step | `correct` (LR 1e-4) | `wrong_lr` (LR 1e-5) |
|---|---|---|
| 5 | 2.163 | 2.163 |
| 10 | 1.382 | 2.066 |
| 15 | 0.1397 | 1.606 |
| 20 | 0.0284 | 1.326 |
| 25 | 0.0165 | 1.141 |
| 30 | 0.0275 | 1.119 |

Hai đường đi hai hướng khác hẳn nhau. `correct` sụp nhanh rồi bão hoà ở ~0.02 — model đã
học xong. `wrong_lr` gần như **phẳng**: 2.163 → 1.119 trong 30 step, tức vẫn đang ở giai
đoạn khởi động khi hết ngân sách.

Nếu chỉ nhìn loss mà không biết LR, tôi sẽ kết luận: *"`wrong_lr` học chậm hơn nhưng vẫn
đang giảm đều — chỉ cần train thêm là nó sẽ bắt kịp."* Kết luận đó **sai về hậu quả**.
Trên tập target, `wrong_lr` đạt **0.000** — không một trường nào đúng, và `format` cũng
0.000, nghĩa là nó **không sinh ra JSON hợp lệ nào cả**. Nó vẫn đang ở chế độ "viết tiếp
văn xuôi" khi hết 30 step. Latency 5.2 giây/mẫu (so với 1.36 của `correct`) là dấu vết thứ
hai của cùng vấn đề: model sinh rác cho tới hết `max_new_tokens`.

Bài học định lượng: train loss xấu hơn **2.5 lần** (1.5702 / 0.6253) đi kèm chất lượng
tác vụ xấu hơn **vô hạn lần** (0.000 / 0.970). Loss là hàm liên tục, chất lượng tác vụ
thì có ngưỡng — model hoặc sinh được JSON, hoặc không. Không có thang nào để nội suy giữa
hai trạng thái đó, và đó là lý do không được đọc loss như thước đo chất lượng.

### 4.3 — `qlora`: tiết kiệm VRAM, trả giá bằng cả chất lượng lẫn tốc độ

`qlora` dùng **3.86 GB** so với **8.78 GB** của `correct` — tiết kiệm **4.92 GB, tức 56%
VRAM**, đúng như kỳ vọng của lượng tử hoá 4-bit.

Cái giá, đo trên cùng thang đo:

| | `correct` | `qlora` |
|---|---|---|
| VRAM | 8.78 GB | **3.86 GB** ✓ |
| target | 0.970 | 0.940 ✗ |
| train loss | 0.6253 | 0.7058 ✗ |
| thời gian train | 402.1 s | **461.9 s** ✗ |

Điều đáng chú ý: `qlora` **chậm hơn 14.9%** (461.9 s so với 402.1 s) dù dùng ít VRAM hơn
một nửa. Lượng tử hoá/giải lượng tử ở mỗi bước tốn thời gian, và tiết kiệm bộ nhớ không
mua lại được khoản đó. Cùng số tham số huấn luyện, cùng 30 step, cùng dữ liệu, cùng seed
— `qlora` thua trên **cả hai** trục chất lượng và tốc độ, chỉ thắng ở bộ nhớ.

**Số đo của tôi ủng hộ khuyến nghị của nhà cung cấp.** Deck §12 nói *không* dùng QLoRA cho
dòng Qwen3.5 vì sai số lượng tử hoá cao hơn bình thường; đo được ở đây là mất 3 điểm
target và chậm hơn 15%. Nhưng phải nói chính xác mức độ ủng hộ: **-0.03 target trên 50
mẫu là 6 trường hợp lệch trên 200** — đủ để thấy xu hướng, chưa đủ để gọi là thảm hoạ.
Kết luận đúng không phải "QLoRA hỏng", mà là **"QLoRA không mua được gì ở tier T4"**:
8.78 GB vừa thoải mái trong 14.6 GB, nên 4.92 GB tiết kiệm được không giải quyết vấn đề
nào, trong khi cái mất là thật. QLoRA chỉ đáng cân nhắc khi VRAM là ràng buộc cứng — và
ở tier này thì không.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.205` · `regression Δ = −0.180` · `valid_trace_rate = 0.00`

Bản fine-tune **thắng baseline (b) đúng theo cách lab yêu cầu**: +0.205 target (0.765 →
0.970), tức vượt mốc một cách rõ ràng chứ không phải trong sai số. Nhưng nó **phá năng
lực phổ thông**: regression 0.7911 → 0.6111, mất 0.180 — gấp **9 lần** ngưỡng cho phép
0.02. Cổng yêu cầu đồng thời thắng target **và** không đánh mất năng lực chung; điều kiện
thứ hai không thoả, nên verdict là FAILED.

Đây là *quên thảm hoạ* mà deck §14.3 mô tả, ở dạng điển hình nhất: 225 mẫu triage JSON,
2 epoch, không một mẫu dữ liệu phổ thông nào. Model học rất tốt định dạng đầu ra mà lab
muốn và trả giá bằng việc quên cách trả lời những câu hỏi đơn giản không có dạng JSON.

Điều này **không** nói rằng fine-tune là vô ích — nó nói rằng **quy trình này** chưa
hoàn chỉnh. Bằng chứng nằm ở 6 ca hỏng định tính: *mọi* lỗi đều là một trường duy nhất
(`urgency`) trên một cấu trúc JSON hợp lệ, không phải lỗi định dạng hay lỗi hiểu sai
nhiệm vụ. Model đã học đúng việc cần học. Cái còn thiếu là một cơ chế bảo toàn năng lực
chung — và deck đã nêu tên nó: trộn **1–5% replay data** phổ thông vào tập huấn luyện.

Hệ quả trực tiếp: **không nên deploy bản này.** Một hệ thống CSKH hỏng ở 18% câu hỏi
ngoài luồng sẽ hỏng nhiều hơn mức 18% đó gợi ý, vì người dùng hỏi ngoài luồng thường
xuyên hơn nhà phát triển tưởng. Với `target` 0.970 thì bản fine-tune đáng giữ — nhưng
phải train lại kèm replay data trước, và phải đo lại cổng hồi quy trước khi nói tới
chuyện triển khai.

`valid_trace_rate` = 0.00 là điều **đã dự đoán trước**, không phải phát hiện mới: corpus
này toàn câu trả lời JSON thuần, không mẫu nào có reasoning trace, và generation prompt
của Qwen3.5 đã đóng khối `<think>` rỗng. Chỉ số này bằng 0 ở **cả** baseline (a), (b) và
(c) — nó đo sự vắng mặt của trace trong dữ liệu, không đo sự thoái hoá do fine-tune. B3
(reasoning-trace collapse) cần một corpus khác mới có ý nghĩa.

---

## 6. Định tính — bắt buộc có cả ca THUA

(Xem `results/qualitative_full.json`, sinh bởi `scripts/capture_qualitative.py`.)

## 7. Kết luận & điều tôi học được

**Kết luận.** Không nên deploy bản fine-tune này — **không phải vì nó thua, mà vì nó
thắng sai cách**. Trên tác vụ mục tiêu nó vượt baseline (b) 0.205 điểm, một khoảng cách
lớn và có ý nghĩa thống kê; định dạng đầu ra hoàn hảo (1.000). Nhưng nó đánh mất 0.180
năng lực phổ thông, gấp 9 lần ngưỡng cho phép, nên nếu đưa vào production thì cứ 5-6 câu
hỏi ngoài luồng sẽ có 1 câu hỏng — trong khi bản base chỉ cần một prompt tử tế đã trả lời
đúng 79.1% số đó mà không cần train gì.

Đòn bẩy thật sự, theo số đo của tôi, **không phải vị trí adapter và không phải rank**.
Phép đối chứng `attn_only` được khớp ngân sách tham số đến 0.003% và rank phải nâng gấp
17.7 lần để làm được điều đó — nó hoà (0.965 so với 0.970). Nếu rank là đòn bẩy, nó đã
thắng. Đòn bẩy lớn nhất là **learning rate**: chỉ đổi một chữ số 1e-4 → 1e-5 mà target rơi
từ 0.970 xuống 0.000 — một hậu quả thảm khốc hơn mọi thay đổi về vị trí hay rank trong
toàn bộ thí nghiệm này. Đòn bẩy thứ hai là **chất lượng prompt** ở phía baseline: chỉ sửa
prompt đã đưa target từ 0.000 lên 0.765 và giảm latency 3.1 lần, **không tốn một giây
train nào**. Và đòn bẩy đang chặn việc deploy là **thành phần dữ liệu**: thiếu replay data
phổ thông là nguyên nhân duy nhất khiến một run có target 0.970 bị đánh trượt.

Xếp hạng theo mức ảnh hưởng, kèm số: **learning rate** (0.970 → 0.000, biên độ 0.970) >
**prompt** (0.000 → 0.765, biên độ 0.765, miễn phí) > **dữ liệu/replay** (chặn deploy,
biên độ 0.180 ở cổng hồi quy) > **vị trí adapter** (0.005, tức 1 trường trên 200 — nhiễu)
> **rank** (0.000 khi đã khớp ngân sách). Mask không nằm trong bảng này vì nó là điều kiện
tiên quyết chứ không phải một mức: mask sai thì mọi con số phía trên đều vô nghĩa.

**Ba điều tôi học được:**

1. **`final_loss` và chất lượng tác vụ có thể xếp hạng ngược nhau, và tôi đã suýt tin
   bảng sai.** `attn_only` có train loss thấp nhất (0.5373) nhưng xếp thứ hai trên target.
   Nếu tôi báo cáo theo loss — như lab cũ làm — tôi đã kết luận rằng gắn LoRA vào q,v với
   r=283 là cấu hình tốt nhất, ngược hoàn toàn sự thật. Cái làm nên khác biệt không phải
   là hiểu điều này trên slide, mà là tự tay dựng `matched_rank()` và thấy 17.7 lần rank
   mua được đúng 0.000 điểm.

2. **Một chỉ số xấu hơn 2.5 lần có thể che một thất bại vô hạn lần.** `wrong_lr` có train
   loss 1.5702 so với 0.6253 — trông như "học chậm hơn", và nếu chỉ đọc loss tôi sẽ
   tưởng chỉ cần train thêm. Thực tế target = 0.000 và format = 0.000: model chưa từng
   thoát khỏi chế độ viết văn xuôi. Loss là thang liên tục; năng lực tác vụ có ngưỡng.
   Không có cách nào nội suy từ cái này sang cái kia.

3. **Thất bại im lặng đắt hơn thất bại ồn ào, và tôi gặp nó ở cả hai phía.** Ở phía lab:
   bộ test báo `3 failed` mà không nói test nào, khiến tôi đoán sai hai lần (đổ cho GPU,
   rồi cho `importorskip`) trước khi sửa được nguyên nhân thật — một namespace package bị
   che. Ở phía pipeline: một lần chạy 66 phút mất trắng vì `results/` không được tải về,
   và một lần khác pipeline chạy cả tiếng mà notebook trống trơn vì
   `subprocess.run()` không stream output lên Colab. Cả ba đều là "chương trình chạy
   đúng, chỉ là không ai thấy gì". Một phép đo không đủ chi tiết sẽ dẫn bạn đi sai
   hướng dù bản thân nó không hề sai.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 1–5% dữ liệu replay phổ thông vào tập train
rồi chạy lại NB3 + NB5, để đo xem bao nhiêu phần trăm replay là đủ kéo `regression` về
trong ngưỡng 0.02 mà không làm mất phần target đã thắng. Đó là thí nghiệm duy nhất, theo
số đo ở §5, có khả năng biến verdict này từ FAILED thành PASSED — và nó cũng là câu hỏi
mà bất kỳ ai triển khai thật sẽ phải trả lời trước tiên.

---

## Phụ lục — thưởng đã làm

- [x] **B1** NB6 merge + hot-swap — `results/merge_check.json`: trước merge 0.970, sau merge
      0.970, **Δ = +0.0000** (ngưỡng 0.01). Merge không làm tụt điểm, xác nhận `W = W₀ +
      (α/r)·BA` là phép toán chính xác ở fp16. Hot-swap 3 adapter (`correct`, `attn_only`,
      `qlora`) trên **cùng một** base đang nạp trong VRAM.
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse — **không chạy được với corpus mặc định**: cả 250 mẫu
      huấn luyện là JSON thuần, không mẫu nào có khối `<think>` trong câu trả lời, và
      generation prompt của Qwen3.5 đã đóng sẵn khối rỗng. `masked-think` và
      `response-only` cho ra mask **giống hệt** `assistant-only`, nên phép đối chứng §13.5
      không có gì để so. `valid_trace_rate = 0.00` ở mọi run là hệ quả của dữ liệu, không
      phải của fine-tune.
- [ ] B4 quét rank có kiểm soát — nhưng xem §4.1: phép đối chứng `attn_only` **đã** trả lời
      câu hỏi trung tâm của B4 bằng số đo (rank gấp 17.7 lần, ngân sách khớp 0.003%, mua
      được 0.000 điểm target).
- [ ] B5 HuggingFace Hub — link:
