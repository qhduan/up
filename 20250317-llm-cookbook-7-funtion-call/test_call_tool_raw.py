import requests
import json

url = "http://localhost:11434/api/chat"
model = 'qwq:32b-q8_0'


def get_current_temperature(location: str, unit: str) -> float:
    """
    获取指定位置的当前温度。
    
    Args:
        location: 获取温度的位置，格式为"城市, 国家"
        unit: 返回温度的单位。(选项: ["celsius", "fahrenheit"])
    Returns:
        以指定单位表示的指定位置的当前温度，浮点数。
    """
    return 22.  # 真实的函数应该实际获取温度！

def get_current_wind_speed(location: str) -> float:
    """
    获取指定位置的当前风速（单位：公里/小时）。

    Args:
        location: 获取温度的位置，格式为"城市, 国家"
    Returns:
        指定位置的当前风速（单位：公里/小时），浮点数。
    """
    return 6.  # 真实的函数应该实际获取风速！

tools = [get_current_temperature, get_current_wind_speed]


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('qwq_32b')

ret = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "伦敦当前的摄氏温度是多少？"}
    ],
    tools=tools,
    tokenize=False,
    add_generation_prompt=True
)

print(ret)

response = requests.post(
    'http://localhost:11434/v1/completions',
    json={
        "model": model,
        "prompt": ret,
        "stream": True
    },
    stream=True
)

# 逐行读取并处理响应
for line in response.iter_lines():
    if line:
        if line.startswith(b'data: {'):
            obj = json.loads(line[6:])
            print(obj['choices'][0]['text'], end='', flush=True)
print()
