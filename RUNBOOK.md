# RUNBOOK — chạy Lab 21 trên Colab T4

> Dành cho máy chỉ có GPU 4 GB (RTX 3050) không chạy được Qwen3.5-4B. Toàn bộ phần
> chuẩn bị đã xong; việc còn lại là bấm chạy trên Colab.

**Tổng thời gian: ~100–130 phút**, chia được thành nhiều phiên (xem §5).

---

## 0. Đẩy code lên fork TRƯỚC KHI mở Colab

Colab **clone repo của bạn**, không phải repo của giảng viên. Mọi thay đổi đã sửa
trong `colab/Lab21_RUN_ALL.ipynb` chỉ có tác dụng sau khi push.

```bash
git add -A
git commit -m "Lab21: runbook + fix bootstrap trỏ đúng fork, EVAL_LIMIT mặc định full, thêm EPOCHS"
git push origin main
```

**Kiểm tra đã push chưa** — mở link này, phải thấy `EVAL_LIMIT   = ""` ở ô số 3:

```
https://github.com/Truongjava/Day21-Track3-LeThanhTruong-2A202602492-Finetuning-Lab/blob/main/colab/Lab21_RUN_ALL.ipynb
```

> Nếu link vẫn hiện `EVAL_LIMIT = "8"`, chưa push xong — quay lại bước trên. Chạy với
> `EVAL_LIMIT=8` sẽ ra bài **không nộp được** (`verify.py` báo FAIL ở mục "full eval set used").

---

## 1. Mở notebook

```
https://colab.research.google.com/github/Truongjava/Day21-Track3-LeThanhTruong-2A202602492-Finetuning-Lab/blob/main/colab/Lab21_RUN_ALL.ipynb
```

**Runtime → Change runtime type → T4 GPU → Save.**

> Colab đọc mã notebook từ GitHub **một lần** lúc mở tab. Sau khi push commit mới,
> phải **đóng tab và mở lại** — reconnect thôi là không đủ.

---

## 2. Ô 1 — Setup (~1 phút)

Clone repo + cài dependency. Cuối ô phải in ra:

```
commit : <sha>
GPU    : Tesla T4
VRAM   : 15.0 GB
```

Nếu `GPU: NONE` → quay lại §1, chưa chọn T4.

---

## 3. Ô 2 — Smoke (~30 giây)

```
7 passed · 0 warnings · 0 failures
Ready to submit.
```

Ô này chỉ kiểm tra import + seed data + unit test, **không cần GPU**.

---

## 4. Ô 3 — Pipeline (~90–130 phút) ← phần chính

Tham số ở đầu ô:

| Tham số | Để nguyên | Khi nào đổi |
|---|---|---|
| `COMPUTE_TIER` | `"T4"` | đúng tier của lab |
| `EVAL_LIMIT` | `""` | **để trống** — đây là bài nộp được |
| `EPOCHS` | `"2"` | đổi `"1"` nếu sắp hết giờ (giảm nửa NB3 **và** NB4) |
| `STAGES` | `nb1 nb2 nb3 nb4 nb5 nb6` | bỏ `nb6` nếu không cần điểm thưởng B1 |

Chạy NB1 → NB6 liên tiếp. **Đừng đóng tab.** Output chạy liên tục, mỗi batch sinh văn
bản đều in dòng ETA — thấy nó nhích là bình thường, không phải treo.

Ước lượng từng chặng (T4 free, đo thật — là **khoảng**, không phải một con số):

| Chặng | Thời gian | Nội dung |
|---|---|---|
| NB1 | ~1 ph | mask proof · template check · p95 → max_length · split seed 42 |
| NB2 | ~17–23 ph | đóng băng eval + đo baseline (a) và (b) **trước khi train** |
| NB3 | ~15–25 ph | train `correct` |
| NB4 | ~45–60 ph | 3 run đối chứng: `attn_only` · `wrong_lr` · `qlora` |
| NB5 | ~21 ph | 4 nhóm + cổng hồi quy + chấm 3 đối chứng |
| NB6 | ~10 ph | merge + hot-swap (điểm thưởng B1) |

---

## 5. Nếu Colab đứt giữa chừng

Không mất gì. Adapter nào train xong đã nằm trên đĩa, và NB4 **tự bỏ qua** adapter đã có.

Mở lại tab, chạy lại ô 1 (setup) rồi chạy tiếp từ chặng bị đứt:

```python
STAGES = "nb4 nb5 nb6"     # ví dụ: đứt ở NB4
```

Chạy lại cả `nb4` cũng an toàn — nó skip những adapter đã train xong. Muốn train lại
từ đầu: xoá `adapters/<key>/` hoặc đặt `FORCE_RETRAIN=1`.

### NB6 báo `ValueError: We need an offload_dir to dispatch this model`

Đã sửa. Nguyên nhân: `del merged` ở cuối §2 không giải phóng được gì, vì `PeftModel`
trong biến `model` vẫn giữ tham chiếu tới đúng model đó — nên 9,3 GB vẫn nằm trên GPU.
Sang §3 nạp thêm bản thứ hai, tổng vượt 14,6 GB, và `device_map="auto"` **âm thầm** đẩy
các lớp cuối xuống CPU thay vì báo lỗi. Chính `load_adapter` của PEFT mới là chỗ ném
lỗi, muộn hơn một nhịp, với thông báo nói về `offload_dir` — sai hoàn toàn so với
nguyên nhân thật.

Nếu vẫn gặp: `Runtime → Restart session` (giữ nguyên `/content`), chạy lại ô 1 rồi ô 3
với `STAGES = "nb6"`. Bản merged đã lưu sẽ được bỏ qua, không tốn lại ~16 phút.

---

## 6. Ô 4 — Gatekeeper

Phải in ra:

```
Ready to submit.
```

Kèm bảng `results/` và `runs.csv`. **WARN thì đọc**, nhất là:

- `baseline (b) beats (a)` — nếu WARN, prompt (b) chưa mạnh hơn (a) và phần thắng ở NB5 là ảo
- `verdict recorded` FAILED — **không sao**, FAILED được chấm điểm đầy đủ nếu phân tích đúng

---

## 7. Kết quả tự về máy

**Không cần làm gì.** Ô 3 đóng gói `results/` + `adapters/correct` thành
`lab21_2A202602492.zip` rồi tự gọi `files.download()` — trong `finally`, nên **chạy dù
pipeline thành công hay đứt giữa chừng**.

Ô 4 gọi lại lần nữa (sau `verify.py`), hữu ích khi bạn muốn bản mới hơn hoặc lần tải
trước bị trình duyệt chặn.

Nếu trình duyệt chặn popup: mở ngăn file bên trái trong Colab (biểu tượng thư mục) →
tải `lab21_2A202602492.zip` bằng tay, hoặc chạy riêng `!python scripts/pack_results.py`.

> **Đây là bước dễ mất cả buổi nhất.** Colab thu hồi máy ảo khi phiên kết thúc, và
> `results/` chính là thứ grader dùng để kiểm chéo mọi con số trong REPORT.md. Một lần
> chạy 66 phút đã mất trắng vì bước này bị bỏ qua — nên giờ nó nằm trong `finally`.

**Đừng** zip cả thư mục `adapters/`: bản `adapters/merged` của NB6 nặng ~9,3 GB.

Sau khi tải về, giải nén vào đúng thư mục repo (đè `results/` và `adapters/`), rồi
chạy lại `make verify` ở máy để chắc chắn.

---

## 8. Việc còn lại

Sau khi có `results/`, viết `submission/REPORT.md` + `submission/REFLECTION.md` từ **số
đo thật**. Mọi con số trong report phải khớp file trong `results/` — grader kiểm chéo,
nên không được gõ tay từ trí nhớ.

Checklist nộp bài: xem [rubric.md](rubric.md) §"Ba lựa chọn định dạng nộp".
