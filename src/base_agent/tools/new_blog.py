import os
import re
from pathlib import Path

async def new_blog(filename: str, contents: str) -> str:
    """Create a new blog post markdown file in the blogs directory.
    
    Args:
        filename: Name of the blog file to create (must be valid filename)
        contents: Markdown contents of the blog post
    
    Returns:
        Success message or error message
    """
    # Validate filename
    if not re.match(r'^[\w\-. ]+$', filename):
        return "Error: Invalid filename. Use only letters, numbers, spaces, hyphens, and periods."
    
    if not filename.endswith('.md'):
        filename += '.md'
    
    blogs_dir = Path('blogs')
    blogs_dir.mkdir(exist_ok=True)
    
    blog_path = blogs_dir / filename
    if blog_path.exists():
        return f"Error: Blog '{filename}' already exists"
        
    try:
        blog_path.write_text(contents)
        return f"Successfully created blog: {filename}"
    except Exception as e:
        return f"Error creating blog: {str(e)}"
