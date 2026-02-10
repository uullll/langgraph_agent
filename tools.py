from langchain_core.tools import tool
from datasets import load_dataset
from pathlib import Path
import os
import traceback
import subprocess
WORKSPACE = Path("workspace")
@tool
def create_file(file_name, file_contents):
    """
    Create a new file with the provided contents at a given path in the workspace.
    
    args:
        file_name (str): Name to the file to be created
        file_contents (str): The content to write to the file
    """
    try:
        WORKSPACE.mkdir(exist_ok=True)

        save_path = os.path.basename(file_name)
        file_path = WORKSPACE /save_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_contents)

        return {
            "message": f"Successfully created file at {file_path}"
        }

    except Exception as e:
        return {
            "error": str(e)
        }

@tool
def str_replace(file_name, old_str, new_str):
    """
    Replace specific text in a file.
    
    args:
        file_name (str): Name to the target file
        old_str (str): Text to be replaced (must appear exactly once)
        new_str (str): Replacement text
    """
    try:
        WORKSPACE.mkdir(exist_ok=True)
        safe_name = os.path.basename(file_name)
        file_path = WORKSPACE / safe_name
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        count = content.count(old_str)
        if count == 0:
            return {"error": f"Text not found: '{old_str}'"}
        if count > 1:
            return {"error": f"Text appears {count} times, expected exactly once"}
        new_content = content.replace(old_str, new_str, 1)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return {"message": "Successfully replaced '{old_str}' with '{new_str}' in {file_path}"}
    except Exception as e:
        return {"error": f"Error replacing '{old_str}' with '{new_str}' in {file_path}: {str(e)}"}

@tool
def send_message(message: str):
    """
    send a message to the user
    
    args:
        message: the message to send to the user
    """
    
    return message

@tool
def shell_exec(command: str) -> dict:
    """
    在指定的 shell 会话中执行命令。

    参数:
        command (str): 要执行的 shell 命令

    返回:
        dict: 包含以下字段：
            - stdout: 命令的标准输出
            - stderr: 命令的标准错误
    """
  
    try:
        # 执行命令
        result = subprocess.run(
            command,
            shell=True,          
            cwd=os.getcwd(),        
            capture_output=True,
            text=True,    
            check=False
        )

        # 返回结果
        return {"message":{"stdout": result.stdout,"stderr": result.stderr}}

    except Exception as e:
        return {"error":{"stderr": str(e)}}
    
@tool
def load_student_dataset() -> dict:
    """Load StudentPerformance dataset from HuggingFace and return a small summary."""
    ds = load_dataset("riyadahmadov/StudentPerformance")
    train = ds["train"]
    return {
        "num_rows": train.num_rows,
        "features": list(train.features.keys()),
        "sample": train.select(range(min(5, train.num_rows))).to_dict(),
    }