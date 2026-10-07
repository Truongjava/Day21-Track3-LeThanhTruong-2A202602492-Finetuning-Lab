# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Hai thứ, và cả hai đều ngược với kỳ vọng của tôi.

Thứ nhất: **rank không phải đòn bẩy.** Tôi vào lab này tin rằng `r` là nút vặn chính —
kiểu "tăng rank lên là model học tốt hơn". Phép đối chứng `attn_only` cho thấy điều ngược
lại: để giữ ngân sách tham số bằng `correct`, rank phải nâng từ 16 lên **283**, gấp 17.7
lần. Kết quả trên target: 0.965 so với 0.970 — đúng **một trường trên 200**, tức là hoà.
Mười bảy lần rank mua được số điểm bằng 0.

Thứ hai, và bất ngờ hơn: **learning rate chỉ khác một chữ số mà hỏng hoàn toàn.** 1e-5
thay vì 1e-4 — `wrong_lr` ra target 0.000 và format 0.000. Không phải "kém hơn", mà là
**không sinh nổi một JSON nào**. Cùng dữ liệu, cùng seed, cùng 30 step, cùng mọi thứ khác.
Tôi đã nghĩ LR là tham số tinh chỉnh; hoá ra nó là tham số quyết định sống còn.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không hề. Tôi tưởng sẽ mất thời gian ở phần LoRA — chọn rank, chọn target modules. Tôi
mất thời gian ở **ba thất bại im lặng**, không cái nào liên quan tới ML:

1. Lần chạy full đầu tiên mất **66 phút** rồi mất trắng kết quả, vì bước tải `results/`
   về là một đoạn code nằm trong tài liệu mà tôi không chạy. Colab thu hồi máy ảo ngay
   khi phiên kết thúc.

2. Lần chạy thứ hai tôi nhìn màn hình trống **suốt một tiếng**. Pipeline vẫn chạy thật,
   nhưng `subprocess.run()` — thứ được thêm vào để có `try/finally` bảo vệ kết quả — không
   stream output lên Colab. Tôi đã tối ưu cho một rủi ro và vô tình tạo ra một rủi ro lớn
   hơn: mất khả năng nhìn thấy tiến trình.

3. Bộ test báo `3 failed, 115 passed` mà **không nói test nào**. Tôi đoán sai hai lần —
   đổ cho GPU, rồi cho `importorskip("torch")` — trước khi sửa được gatekeeper để nó in
   tên test, và tìm ra nguyên nhân thật trong vài phút: một package tên `tests` trong
   site-packages của Colab che module `fake_tokenizer` của repo.

Cả ba đều có cùng hình dạng: **chương trình chạy đúng, chỉ là không ai thấy gì.** Đó là
bài học tốn kém nhất và cũng là bài học thật nhất của lab này.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi tin **loss là thước đo chất lượng**. Không phải thước đo duy nhất — tôi biết phải
xem cả eval — nhưng tôi tin nó chỉ đúng hướng: loss thấp hơn nghĩa là model tốt hơn.

Bảng ở §4 phá vỡ niềm tin đó bằng số cụ thể. `attn_only` có train loss **thấp nhất**
(0.5373) nhưng chỉ xếp thứ hai trên target. `wrong_lr` có loss 1.5702 — trông như "học
chậm, cần thêm step" — nhưng thực tế target bằng 0.000. Nếu tôi xếp hạng bốn run bằng
`final_loss`, tôi sẽ báo cáo rằng `attn_only` r=283 là cấu hình **tốt nhất** và khuyên
gắn LoRA vào q,v với rank khổng lồ. Đó là kết luận ngược hoàn toàn sự thật.

Điều sâu hơn tôi nhận ra: loss là **thang liên tục**, còn năng lực tác vụ có **ngưỡng**.
`wrong_lr` đi từ loss 2.163 xuống 1.119 — một cải thiện thật, đo được — trong khi vẫn
chưa thoát khỏi chế độ viết văn xuôi. Nó đang tiến bộ và vẫn hoàn toàn vô dụng cùng lúc.
Không có cách nào nội suy từ cái này sang cái kia, và đó chính xác là lý do phải chấm
bằng tác vụ.

Tôi cũng hết tin rằng "fine-tune thắng base" là một kết luận đủ. Bản của tôi thắng (b)
**+0.205** — một chiến thắng rõ ràng, không phải trong sai số — và vẫn bị đánh trượt, vì
nó phá 0.180 năng lực phổ thông. Thắng trên tác vụ mục tiêu không bù được cho việc mất
mọi thứ khác.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng nó để đọc hiểu lab, sửa lỗi pipeline, và viết report từ số đo. Nó sai theo ba
cách khác nhau, và cả ba đều đáng nhớ:

- **Sửa đúng vấn đề bằng cách tạo ra vấn đề lớn hơn.** Nó thêm `try/finally` để bảo vệ
  kết quả khỏi việc đứt giữa chừng — đúng. Nhưng đổi từ `!python` sang `subprocess.run()`,
  và output biến mất khỏi notebook. Cái `finally` đó hoá ra không cần thiết: `!` của
  IPython không raise khi lệnh lỗi, nên một dòng thứ hai là đủ.

- **Viết ra con số sai mà không tự phát hiện.** Trong script lấy dữ liệu định tính, nó
  hardcode `NAIVE_PROMPT` cho cả hai model — nghĩa là baseline (b) sẽ tái lập baseline (a)
  ở mức 0.000, và mọi dòng trong bảng so sánh sẽ đọc như "fine-tune thắng vang dội". Nó tự
  bắt được lỗi này khi đọc lại code, trước khi chạy. Đáng chú ý: đó là **cùng một lỗi** mà
  lab này gọi là F-31 (prompt lúc train khác prompt lúc eval).

- **Đoán sai hai lần vì dữ liệu không đủ chi tiết.** Khi gatekeeper báo `3 failed`, nó
  đoán là GPU, rồi đoán là `importorskip`. Cả hai đều sai. Sau khi sửa gatekeeper để in
  tên test, nguyên nhân thật lộ ra trong vài phút.

Điều tôi rút ra không phải "AI sai" mà là: **nó sai theo cách rất khó phát hiện, và cách
duy nhất để bắt là tự kiểm chứng bằng phép đo của mình.** Ba lỗi trên đều bị bắt bằng cách
chạy thật và đọc kết quả, không bằng cách đọc lại code.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

**Đo baseline đã được prompt tử tế, trước khi viết một dòng code train nào.**

Lab này cho tôi con số cụ thể cho việc đó: chỉ sửa prompt, target đi từ **0.000 lên 0.765**
và latency giảm **3.1 lần** — 3.29 giây xuống 1.05 giây mỗi mẫu. Không tốn một giây GPU
nào, không một mẫu dữ liệu nào. Nếu tôi bỏ qua bước này, tôi sẽ train một model đạt 0.970
và báo cáo "thắng 0.970 điểm" trong khi sự thật là phần lớn khoảng cách đó đến từ một
prompt tốt hơn mà khách hàng có thể có miễn phí ngay hôm nay.

Bước thứ hai, ngay sau đó: **dựng cổng hồi quy trước khi train, không phải sau.** Bản
fine-tune của tôi mất 18% năng lực phổ thông, và nếu tôi chỉ đo tập target như thói quen
phổ biến, tôi sẽ không bao giờ thấy điều đó — nó sẽ được triển khai và hỏng dần trong
production. Cần biết ngưỡng ngay từ đầu, vì sau khi thấy kết quả thì rất khó giữ mình
không nới nó ra.
