CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    user_input TEXT,
    bot_response TEXT,
    timestamp DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(id)
);