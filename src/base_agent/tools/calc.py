async def calc(a: float, b: float, calc_type: str) -> dict:
    """Perform basic arithmetic calculations.
    
    Args:
        a: First number
        b: Second number
        calc_type: Type of calculation ('add', 'sub', 'mul', 'div')
    """
    operations = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y if y != 0 else "Error: Division by zero"
    }
    
    if calc_type not in operations:
        return {"error": f"Invalid calculation type: {calc_type}"}
    
    try:
        result = operations[calc_type](a, b)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
