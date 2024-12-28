import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

def _load_todos() -> List[Dict[str, Any]]:
    """Load todos from the JSONL file."""
    try:
        with open('todo.jsonl', 'r', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        return []

def _save_todos(todos: List[Dict[str, Any]]) -> None:
    """Save todos to the JSONL file."""
    with open('todo.jsonl', 'w', encoding='utf-8') as f:
        for todo in todos:
            f.write(json.dumps(todo, ensure_ascii=False) + '\n')

def create_todo(text: str) -> Dict[str, Any]:
    """Create a new todo item."""
    todos = _load_todos()
    new_todo = {
        "checked": False,
        "id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "text": text
    }
    todos.append(new_todo)
    _save_todos(todos)
    return new_todo

def done_todo(id: str) -> Optional[Dict[str, Any]]:
    """Mark a todo as done by its ID."""
    todos = _load_todos()
    for todo in todos:
        if todo["id"] == id:
            todo["checked"] = True
            todo["updated_at"] = datetime.now().isoformat()
            _save_todos(todos)
            return todo
    return None

def update_todo(id: str, new_text: str) -> Optional[Dict[str, Any]]:
    """Update a todo's text by its ID."""
    todos = _load_todos()
    for todo in todos:
        if todo["id"] == id:
            todo["text"] = new_text
            todo["updated_at"] = datetime.now().isoformat()
            _save_todos(todos)
            return todo
    return None

def delete_todo(id: str) -> bool:
    """Delete a todo by its ID."""
    todos = _load_todos()
    initial_length = len(todos)
    todos = [todo for todo in todos if todo["id"] != id]
    if len(todos) != initial_length:
        _save_todos(todos)
        return True
    return False

def get_todos(is_checked: Optional[bool] = None) -> List[Dict[str, Any]]:
    """
    Get todos filtered by checked status.
    If is_checked is None, returns all todos.
    """
    todos = _load_todos()
    if is_checked is None:
        return todos
    return [todo for todo in todos if todo["checked"] == is_checked]
