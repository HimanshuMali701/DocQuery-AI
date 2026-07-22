import bcrypt
from database import create_user, get_user_by_email

def hash_password(password: str) -> str:
    """
    Hash a plain-text password using bcrypt.
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a plain-text password against a stored hash.
    """
    return bcrypt.checkpw(
        password.encode("utf-8"),
        password_hash.encode("utf-8")
    )

def register_user(name: str, email: str, password: str):
    """
    Register a new user.

    Returns:
        (True, message) on success
        (False, message) on failure
    """

    email = email.strip().lower()

    # Check if email already exists
    if get_user_by_email(email):
        return False, "Email already registered."

    password_hash = hash_password(password)

    try:
        create_user(
            name=name,
            email=email,
            password_hash=password_hash
        )

        return True, "Registration successful."

    except Exception as e:
        return False, str(e)    

def login_user(email: str, password: str):
    """
    Authenticate a user.

    Returns:
        (True, user) on success
        (False, message) on failure
    """

    email = email.strip().lower()

    user = get_user_by_email(email)

    if not user:
        return False, "User not found."

    password_hash = user["password_hash"]

    if not verify_password(password, password_hash):
        return False, "Incorrect password."

    return True, user        