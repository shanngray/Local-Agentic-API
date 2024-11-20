from pathlib import Path

async def edit_blog(filename: str, contents: str) -> str:
    """Edit an existing blog post markdown file.
    
    Args:
        filename: Name of the blog file to edit
        contents: New markdown contents for the blog post
    
    Returns:
        Success message or error message
    """
    if not filename.endswith('.md'):
        filename += '.md'
    
    blog_path = Path('blogs') / filename
    if not blog_path.exists():
        return f"Error: Blog '{filename}' does not exist"
        
    try:
        blog_path.write_text(contents)
        return f"Successfully updated blog: {filename}"
    except Exception as e:
        return f"Error updating blog: {str(e)}"
