
import json
import traceback
import asyncio
import json

import aiohttp

from todo import get_todos, create_todo, done_todo, update_todo, delete_todo


async def llmchat(history, engine):
    """
    Async function to chat with Ollama LLM using aiohttp
    
    Args:
        history (list): List of message dictionaries with role and content
        engine (str): Name of the LLM model to use
    
    Returns:
        str: Combined response text from the LLM
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:11434/api/chat',
            json={
                'model': engine,
                'messages': history,
                'stream': False,
            }
        ) as response:
            return (await response.json())['message']


prompt = '''# 你要做的事情

你是一个TODO助手，你需要跟用户对话，实现TODO功能，你必须利用所有可以调动的工具函数来实现功能，你只能回复一个工具调用，例如"need_more_info('我需要更多信息')"

## 你能使用的工具

def create_todo(text: str) -> Dict[str, Any]:
    """Create a new todo item."""

def done_todo(id: str) -> Optional[Dict[str, Any]]:
    """Mark a todo as done by its ID."""

def update_todo(id: str, new_text: str) -> Optional[Dict[str, Any]]:
    """Update a todo's text by its ID."""

def delete_todo(id: str) -> bool:
    """Delete a todo by its ID."""

def get_todos(is_checked: Optional[bool] = None) -> List[Dict[str, Any]]:
    """
    Get todos filtered by checked status.
    If is_checked is None, returns all todos.
    """

def need_more_info(message: str) -> str:
    """
    向用户请求更多的信息，例如用户说删除某一条，你需要知道用户提供要删除的todo的信息
    Args:
        message: 你需要给用户信息，提醒用户做什么
    """

'''


def need_more_info(message: str) -> str:
    print("Bot:", message)


async def main():
    messages = [None]

    while True:
        messages[0] = {"role": "system", "content": prompt + f"""
# 当前所有的TODO:

""" + json.dumps(get_todos(), ensure_ascii=False)}

        user_input = input('user: ')
        if len(user_input) <= 0:
            continue
        messages.append({"role": "user", "content": user_input})
        print('模型输入：')
        print(json.dumps(messages, indent=4, ensure_ascii=False))
        print()
        ret = await llmchat(history=messages, engine='qwen2.5:32b')
        messages.append(ret)
        print('模型返回：')
        print(ret)
        print()
        try:
            eval(ret['content'])
        except:
            traceback.print_exc()
        # break


if __name__ == '__main__':
    asyncio.run(main())
