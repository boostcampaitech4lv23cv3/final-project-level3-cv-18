import os
from typing import Dict, List
import argparse
import time

FILES:Dict[str,str] = {
    "safe_driving_assistance_system_for_novice_drivers_kitti.pth" : "work_dirs",
    "safe_driving_assistance_system_for_novice_drivers_finetuned.pth" : "work_dirs",
    "safe_driving_assistance_system_for_novice_drivers_onnx.onnx" : "work_dirs",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Input file id')
    parser.add_argument('file_id', help='file id', type=str)
    args = parser.parse_args()
    return args

def download_objects(file_id:str, file_list:List[str]):
    """
    file_id : fild_id를 입력하세요.
    file_list : file_id에서 다운받을 파일의 목록을 입력하세요.
    작성자 : 김형석
    """
    command_template = "wget --load-cookies ~/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILEID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILEID}\" -O {FILENAME} && rm -rf ~/cookies.txt"
    for idx, file in enumerate(file_list):
        command = command_template.format(FILEID = file_id, FILENAME = file, idx = idx)
        print("[{current}/{total}] Downloading >> {file}".format(current=idx+1, total=len(file_list), file=file))
        os.system(command)
        time.sleep(2)

def move_objects(files:Dict[str,str]):
    """
    files : 옮길 대상과 옮길 위치 쌍을 입력하세요.
    """
    command_template = "mv {file} {dir}"
    for file in files:
        dir = files[file]
        command = command_template.format(file=file, dir=dir)
        print(command)
        os.system(command)

def main(args:argparse.Namespace):
    download_objects(args.file_id, FILES.keys()) # type: ignore
    move_objects(FILES)

if __name__ == "__main__":
    args = parse_args()
    main(args=args)


