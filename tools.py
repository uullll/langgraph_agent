from langchain_core.tools import tool
import textwrap
from datasets import load_dataset
from pathlib import Path
import os
import traceback
import subprocess
from datasets import load_dataset
from datetime import datetime, timezone
import uuid
from config import WORKSPACE, get_setting


SHELL_EXEC_MAX_OUTPUT_CHARS = int(get_setting("SHELL_EXEC_MAX_OUTPUT_CHARS", "tools.shell_exec.max_output_chars", default=1000))
def _safe_path(rel_path: str) -> Path:
    p = (WORKSPACE / rel_path).resolve()
    if not str(p).startswith(str(WORKSPACE.resolve())):
        raise ValueError(f"Path escape blocked: {rel_path}")
    return p

def _truncate_output(text: str, max_chars: int) -> tuple[str, bool]:
    if text is None:
        return "", False
    if len(text) <= max_chars:
        return text, False
    head = max_chars // 2
    tail = max_chars - head
    truncated = (
        text[:head]
        + f"\n... [truncated {len(text) - max_chars} chars] ...\n"
        + text[-tail:]
    )
    return truncated, True


def _write_shell_exec_log(command: str, stdout: str, stderr: str, return_code: int) -> str:
    logs_dir = WORKSPACE / "tool_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = logs_dir / f"shell_exec_{stamp}_{uuid.uuid4().hex[:8]}.log"
    content = (
        f"$ {command}\n"
        f"exit_code: {return_code}\n"
        "\n===== STDOUT =====\n"
        f"{stdout or ''}\n"
        "\n===== STDERR =====\n"
        f"{stderr or ''}\n"
    )
    log_path.write_text(content, encoding="utf-8")
    return str(log_path)
@tool
def load_hf_dataset():
    """Download the dataset from Huggingface"""
    dataset_name = get_setting("HF_DATASET", "tools.load_data.dataset", required=True)
    dataset_config = get_setting("HF_DATASET_CONFIG", "tools.load_data.dataset_config")
    split = get_setting("HF_SPLIT", "tools.load_data.split", default="train")

    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    return ds
@tool
def save_dataset(ds):
    """Save the dataset to specific file path"""
    save_path = WORKSPACE / "dataset.parquet"
    ds.to_parquet(str(save_path))

def clean_code(code: str) -> str:
    return textwrap.dedent(code).lstrip()
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

        file_path = _safe_path(file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(clean_code(file_contents))

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
        file_path = _safe_path(file_name)
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

        return {"message": f"Successfully replaced '{old_str}' with '{new_str}' in {file_path}"}
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
    Execute a shell command in the current working directory.

    Args:
        command (str): Shell command to execute.

    Returns:
        dict: Contains:
            - stdout: standard output
            - stderr: standard error
    """
  
    try:
        # Execute command
        result = subprocess.run(
            command,
            shell=True,          
            cwd=WORKSPACE,          
            capture_output=True,
            text=True,    
            check=False
        )

        # Return results
        log_path = _write_shell_exec_log(
            command=command,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )

        stdout_preview, stdout_truncated = _truncate_output(result.stdout, SHELL_EXEC_MAX_OUTPUT_CHARS)
        stderr_preview, stderr_truncated = _truncate_output(result.stderr, SHELL_EXEC_MAX_OUTPUT_CHARS)

        # Return bounded results for ToolMessage to avoid oversized context payloads.
        return {
            "message": {
                "stdout": stdout_preview,
                "stderr": stderr_preview,
                "exit_code": result.returncode,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "full_output_log": log_path,
            }
        }

    except Exception as e:
        return {"error":{"stderr": str(e), "type": type(e).__name__}}
    
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
