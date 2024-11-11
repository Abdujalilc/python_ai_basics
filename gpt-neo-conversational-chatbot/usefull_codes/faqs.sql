CREATE TABLE faqs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT,
    answer TEXT,
    question_embedding BLOB  -- This stores the precomputed embedding
);
