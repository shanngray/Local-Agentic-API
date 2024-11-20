from pathlib import Path

async def read_blog(filename: str) -> str:
    """Read the contents of a blog post markdown file.
    
    Args:
        filename: Name of the blog file to read
    
    Returns:
        Contents of the blog file or error message
    """
    if not filename.endswith('.md'):
        filename += '.md'
    
    blog_path = Path('blogs') / filename
    if not blog_path.exists():
        return f"Error: Blog '{filename}' does not exist"
        
    try:
        return blog_path.read_text()
    except Exception as e:
        return f"Error reading blog: {str(e)}"
