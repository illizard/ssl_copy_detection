import time
import torch
import timm

model = timm.create_model("deit_tiny_patch16_224.fb_in1k", num_classes=0, pretrained=True)
model.eval()

input_data = torch.randn(1, 3, 224, 224)  # 예제 입력

# 타이머 시작
start_time = time.time()

# 모델 여러 번 실행
num_runs = 100
for _ in range(num_runs):
    with torch.no_grad():
        outputs = model(input_data)

# 타이머 멈춤
end_time = time.time()

# 평균 처리 시간 계산 (ms)
average_processing_time = ((end_time - start_time) / num_runs) * 1000  # 초를 밀리초로 변환
print(f"Average processing time per image: {average_processing_time:.2f} ms")