import platform
import os
import subprocess

def run_cmd(cmd):
    """Execute a shell command and return its output."""
    try:
        return subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    except:
        return None
    
def get_repo_root():
    """
    从当前脚本位置向上查找直到找到.git目录来确定repo根目录
    返回: str - repo的根目录的绝对路径
    """
    current_path = os.path.abspath(os.path.dirname(__file__))
    while current_path != '/':  # 对Windows系统需要修改这个条件
        if os.path.exists(os.path.join(current_path, '.git')):
            return current_path
        current_path = os.path.dirname(current_path)
    raise Exception("未找到.git目录，请确保在git仓库中运行")

def get_env_info():
    """
    Detect the current running environment.
    Returns: str, str - One of "Kaggle", "Mac", "AutoDL", "Linux", or "Unknown" and project root
    """
    # Check Kaggle first
    hostname = run_cmd('hostname')  # machine hostname
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None:
        kaggle_repo_root = "./repo_root"
        # mkdir if not exists
        os.makedirs(kaggle_repo_root, exist_ok=True)
        return "Kaggle", kaggle_repo_root
    
    # Check if Mac or Linux
    system = platform.system()
    repo_root = get_repo_root()
    if system == "Darwin":
        return "Mac", repo_root
    elif system == "Linux":
        if hostname and 'autodl' in hostname:
            return "AutoDL", repo_root
        else:
            return "Linux", repo_root
    else:
        return "Unknown", None