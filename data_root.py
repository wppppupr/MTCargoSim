import platform
from pathlib import Path

def data_root():
    current_os = platform.system()
    
    if current_os == "Darwin":    # Macの場合
        # Macは自動的に /Volumes/ラベル名 にマウントされます
        return Path("/Volumes/My Passport")
    
    elif current_os == "Linux":   # Ubuntuの場合
        # 先ほど設定した固定マウントパスを指定します
        # (例: /media/あなたのユーザー名/data)
        return Path("/media/sasaki/data") 
    
    else:
        raise OSError(f"未対応のOSです: {current_os}")