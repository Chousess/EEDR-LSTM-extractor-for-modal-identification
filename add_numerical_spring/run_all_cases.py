"""
依次运行全部四种损失函数消融实验，等价于：
  python train_spring_ablation.py 1
  python train_spring_ablation.py 2
  python train_spring_ablation.py 3
  python train_spring_ablation.py 4
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_spring_ablation import main, CASE_NAMES

if __name__ == "__main__":
    for case_num in sorted(CASE_NAMES.keys()):
        print(f"\n{'='*50}")
        print(f"开始 Case {case_num}: {CASE_NAMES[case_num]}")
        print(f"{'='*50}")
        main(case_num)

    print("\n全部消融实验完成。")
    print(f"输出目录: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')}")
    print(f"日志目录: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')}")
