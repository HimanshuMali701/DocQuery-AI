import sqlite3
from pathlib import Path

DATABASE_NAME = "chatbot.db"

class DatabaseManager:
    """
    Handles all SQLite database operations.
    """

    def __init__(self, database_name=DATABASE_NAME):

        self.database_name = database_name

        self.initialize_database()
        
    def get_connection(self):
        """
        Create a SQLite connection.
        """

        return sqlite3.connect(self.database_name)

    def initialize_database(self):
        """
        Create required tables if they do not exist.
        """

        connection = self.get_connection()

        cursor = connection.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            title TEXT NOT NULL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            conversation_id INTEGER,

            role TEXT,

            content TEXT,

            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY(conversation_id)
            REFERENCES conversations(id)

        )
        """)

        connection.commit()

        connection.close()        
    
    def show_tables(self):
        """
        Display all database tables.
        """

        connection = self.get_connection()

        cursor = connection.cursor()

        cursor.execute("""

        SELECT name

        FROM sqlite_master

        WHERE type='table'

        """)

        tables = cursor.fetchall()

        connection.close()

        return tables
    def create_conversation(self, title="New Chat"):
        """
        Create a new conversation.

        Parameters
        ----------
        title : str
            Conversation title.

        Returns
        -------
        int
            Newly created conversation ID.
        """

        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO conversations (title)
            VALUES (?)
            """,
            (title,)
        )

        conversation_id = cursor.lastrowid

        connection.commit()
        connection.close()

        return conversation_id

        

        
database = DatabaseManager()

chat_id = database.create_conversation("ChatBot")
print(database.show_tables())       

print(chat_id)
  