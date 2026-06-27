"""
Utility Helper Functions

Reusable helper functions used throughout the project.
"""

from pathlib import Path

def create_directory(path: str) -> Path:
    
    """
    Create a directory if it does not exist.

    Args:
        path: Directory path

    Returns:
        Path object of the created directory.
    """
    
    directory_path = Path(path)
    directory_path.mkdir(parents=True, exist_ok=True)
    
    return directory_path

def save_text_file(path: str, content: str) -> Path:
    """
    Save text to a UTF-8 encoded file.

    Args:
        path: File path.
        content: Text content to save.

    Returns:
        Path object of the saved file.
    """
    
    file_path = Path(path)
    
    create_directory(file_path.parent)
    
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)
        
    return file_path