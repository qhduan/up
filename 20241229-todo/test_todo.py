import unittest
import os
from todo import create_todo, done_todo, update_todo, delete_todo, get_todos

class TestTodo(unittest.TestCase):
    def setUp(self):
        """Setup test environment - ensure clean todo.jsonl"""
        if os.path.exists('todo.jsonl'):
            os.remove('todo.jsonl')
    
    def tearDown(self):
        """Cleanup after tests"""
        if os.path.exists('todo.jsonl'):
            os.remove('todo.jsonl')
    
    def test_create_todo(self):
        """Test creating a new todo"""
        todo = create_todo("Test task")
        self.assertEqual(todo["text"], "Test task")
        self.assertFalse(todo["checked"])
        self.assertIsNotNone(todo["id"])
        self.assertIsNotNone(todo["created_at"])
        self.assertIsNotNone(todo["updated_at"])
        
        # Verify it's in the list
        todos = get_todos()
        self.assertEqual(len(todos), 1)
        self.assertEqual(todos[0]["text"], "Test task")
    
    def test_done_todo(self):
        """Test marking a todo as done"""
        # Create a todo first
        todo = create_todo("Test task")
        todo_id = todo["id"]
        
        # Mark it as done
        updated_todo = done_todo(todo_id)
        self.assertIsNotNone(updated_todo)
        self.assertTrue(updated_todo["checked"])
        
        # Verify in the list
        todos = get_todos(is_checked=True)
        self.assertEqual(len(todos), 1)
        self.assertTrue(todos[0]["checked"])
        
        # Test non-existent todo
        result = done_todo("non-existent-id")
        self.assertIsNone(result)
    
    def test_update_todo(self):
        """Test updating a todo's text"""
        # Create a todo first
        todo = create_todo("Original text")
        todo_id = todo["id"]
        
        # Update it
        updated_todo = update_todo(todo_id, "Updated text")
        self.assertIsNotNone(updated_todo)
        self.assertEqual(updated_todo["text"], "Updated text")
        
        # Verify in the list
        todos = get_todos()
        self.assertEqual(len(todos), 1)
        self.assertEqual(todos[0]["text"], "Updated text")
        
        # Test non-existent todo
        result = update_todo("non-existent-id", "New text")
        self.assertIsNone(result)
    
    def test_delete_todo(self):
        """Test deleting a todo"""
        # Create a todo first
        todo = create_todo("Test task")
        todo_id = todo["id"]
        
        # Delete it
        result = delete_todo(todo_id)
        self.assertTrue(result)
        
        # Verify it's gone
        todos = get_todos()
        self.assertEqual(len(todos), 0)
        
        # Test deleting non-existent todo
        result = delete_todo("non-existent-id")
        self.assertFalse(result)
    
    def test_get_todos(self):
        """Test getting todos with different filters"""
        # Create some todos
        create_todo("Task 1")
        todo2 = create_todo("Task 2")
        create_todo("Task 3")
        
        # Mark one as done
        done_todo(todo2["id"])
        
        # Test getting all todos
        all_todos = get_todos()
        self.assertEqual(len(all_todos), 3)
        
        # Test getting unchecked todos
        unchecked = get_todos(is_checked=False)
        self.assertEqual(len(unchecked), 2)
        
        # Test getting checked todos
        checked = get_todos(is_checked=True)
        self.assertEqual(len(checked), 1)
        self.assertEqual(checked[0]["text"], "Task 2")

if __name__ == '__main__':
    unittest.main()
