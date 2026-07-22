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
        conn = sqlite3.connect(self.database_name)
        conn.row_factory = sqlite3.Row
        return conn
    
    def initialize_database(self):
        """
        Create required tables if they do not exist.
        """

        connection = self.get_connection()

        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (

            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id)
            REFERENCES users(id)
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

        # Migration: Check if user_id column exists in conversations table
        cursor.execute("PRAGMA table_info(conversations)")
        columns = [row["name"] for row in cursor.fetchall()]
        if "user_id" not in columns:
            cursor.execute("ALTER TABLE conversations ADD COLUMN user_id INTEGER NOT NULL DEFAULT 1")

        connection.commit()

        connection.close()        
    
    def create_user(self, name, email, password_hash):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO users(name, email, password_hash)
            VALUES (?, ?, ?)
            """,
            (name, email, password_hash)
        )

        conn.commit()
        conn.close()

    def get_user_by_email(self, email):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM users
            WHERE email = ?
            """,
            (email,)
        )

        user = cursor.fetchone()
        conn.close()
        return user

    def get_user(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM users
            WHERE id = ?
            """,
            (user_id,)
        )

        user = cursor.fetchone()
        conn.close()
        return user

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
    def create_conversation(self, title="New Chat", user_id=None):
        """
        Create a new conversation.

        Parameters
        ----------
        title : str
            Conversation title.
        user_id : int, optional
            The user ID this conversation belongs to.

        Returns
        -------
        int
            Newly created conversation ID.
        """

        connection = self.get_connection()
        cursor = connection.cursor()

        # If user_id is not provided, try to find a default user or use a dummy ID like 1 if no user exists.
        if user_id is None:
            try:
                import streamlit as st
                if "user_id" in st.session_state and st.session_state.user_id is not None:
                    user_id = st.session_state.user_id
            except ImportError:
                pass

        if user_id is None:
            cursor.execute("SELECT id FROM users LIMIT 1")
            row = cursor.fetchone()
            if row:
                user_id = row[0]
            else:
                user_id = 1 # Fallback dummy ID

        cursor.execute(
            """
            INSERT INTO conversations (user_id, title)
            VALUES (?, ?)
            """,
            (user_id, title)
        )

        conversation_id = cursor.lastrowid

        connection.commit()
        connection.close()

        return conversation_id

    def get_all_conversations(self, user_id=None):
        """
        Return all conversations.

        Parameters
        ----------
        user_id : int, optional
            Filter conversations by this user ID.

        Returns
        -------
        list
        """

        if user_id is None:
            try:
                import streamlit as st
                if "user_id" in st.session_state and st.session_state.user_id is not None:
                    user_id = st.session_state.user_id
            except ImportError:
                pass

        connection = self.get_connection()
        cursor = connection.cursor()

        if user_id is not None:
            cursor.execute("""
                SELECT
                    id,
                    title,
                    created_at
                FROM conversations
                WHERE user_id = ?
                ORDER BY created_at DESC
            """, (user_id,))
        else:
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


database = DatabaseManager()


def create_user(name, email, password_hash):
    return database.create_user(name, email, password_hash)


def get_user_by_email(email):
    return database.get_user_by_email(email)


def get_user(user_id):
    return database.get_user(user_id)