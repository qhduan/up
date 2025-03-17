import requests
import json

response = requests.post(
    # 'http://192.168.71.6:8101/v1/chat/completions',
    'http://localhost:11434/v1/chat/completions',
    json={
        # "model": "QwQ-32B",
        "model": 'qwq:32b-q8_0',
        "messages": [
            {
                "role": "user",
                "content": "巴黎今天的天气怎么样？"
            }
        ],
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "获取指定位置的当前天气",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "要获取天气的位置，例如：旧金山，加利福尼亚"
                            },
                            "format": {
                                "type": "string",
                                "description": "返回天气的格式，例如：'摄氏度'或'华氏度'",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location", "format"]
                    }
                }
            }
        ]
    },
    stream=False
)

print(json.dumps(response.json(), ensure_ascii=False, indent=4))
