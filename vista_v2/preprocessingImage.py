import base64
from io import BytesIO

# 将图片转为 API 可以接受的 Base64 格式
def image_to_data_url(image):
    buffer = BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=90,
    )

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

