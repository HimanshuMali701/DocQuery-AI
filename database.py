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

    def get_all_conversations(self):
        """
        Return all conversations.

        Returns
        -------
        list
        """

        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                id,
                title,
                created_at
            FROM conversations
            ORDER BY created_at DESC
        """)

        conversations = cursor.fetchall()

        connection.close()

        return conversations

    def get_conversation(self, conversation_id):
        """
        Return one conversation.

        Parameters
        ----------
        conversation_id : int

        Returns
        -------
        tuple | None
        """

        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                title,
                created_at
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,)
        )

        conversation = cursor.fetchone()

        connection.close()

        return conversation
    def update_conversation_title(
        self,
        conversation_id,
        new_title
    ):
        """
        Update conversation title.
        """

        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            UPDATE conversations
            SET title = ?
            WHERE id = ?
            """,
            (new_title, conversation_id)
        )

        connection.commit()
        connection.close()

    def delete_conversation(self, conversation_id):
        """
        Delete a conversation and all related messages.
        """

        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM messages
            WHERE conversation_id = ?
            """,
            (conversation_id,)
        )

        cursor.execute(
            """
            DELETE FROM conversations
            WHERE id = ?    
            """,
            (conversation_id,)
        )

        connection.commit()
        connection.close()   
    
    def save_message(
        self,
        conversation_id,
        role,
        content
    ):
        """
        Save a single message to the database.

        Parameters
        ----------
        conversation_id : int
        role : str
            "user" or "assistant"
        content : str
        """

        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO messages (
                conversation_id,
                role,
                content
            )
            VALUES (?, ?, ?)
            """,
            (
                conversation_id,
                role,
                content
            )
        )

        connection.commit()
        connection.close()

    def get_messages(self, conversation_id):
        """
        Retrieve all messages for a conversation.

        Parameters
        ----------
        conversation_id : int

        Returns
        -------
        list
        """

        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                role,
                content,
                timestamp
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,)
        )

        messages = cursor.fetchall()

        connection.close()

        return messages
    def delete_messages(
        self,
        conversation_id):
        """
        Delete all messages for a conversation.
        """

        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM messages
            WHERE conversation_id = ?
            """,
            (conversation_id,)
        )

        connection.commit()
        connection.close()

    def clear_chat_history(self, conversation_id):
        """
        Alias for delete_messages.
        """
        return self.delete_messages(conversation_id)

    def load_chat_history(
        self,
        conversation_id
    ):
        """
        Load conversation history in Streamlit format.

        Returns
        -------
        list
        """

        messages = self.get_messages(
            conversation_id
        )

        history = []

        for role, content, timestamp in messages:

            history.append({

                "role": role,

                "content": content

            })

        return history   
    def reset_database(self):
        """
        Delete all tables and recreate them.
        """

        connection = self.get_connection()
        cursor = connection.cursor()

        cursor.execute("DROP TABLE IF EXISTS messages")
        cursor.execute("DROP TABLE IF EXISTS conversations")

        self.initialize_database()

        connection.commit()
        connection.close()
if __name__ == "__main__":
    database = DatabaseManager()
    history = database.load_chat_history(7)
    print(history)


