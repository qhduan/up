帮我写一个python版本的最简单的todo list程序的API，只需要写api到todo.py
todo本身存储为jsonl格式，就是每行一个json，每个json类似这样：
{
    "checked": true or false,
    "id": uuidv4,
    "created_at": "创建时间",
    updated_at: "修改时间",
    "text": 内容
}
数据库保存在todo.jsonl
包含的api函数有：
create_todo(text)
done_todo(id) # 把一个todo编程checked
update_todo(id, new_text) # 更新一个todo
delete_todo(id)
get_todos(is_checked) # 如果is_checked是None返回所有todo，否则返回对应checked一样的todo
这些函数可以读取todo.jsonl中的内容，作出修改，并保存
