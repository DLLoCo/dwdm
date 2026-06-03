"""
查看 ST-SSL 输入窗口的真实结构 — 搞清楚 35 步到底是什么

用法: python inspect_input_window.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from lib.utils import get_project_path

def main():
    data_dir = get_project_path('data', 'processed', 'NYCTaxi')
    train_path = os.path.join(data_dir, 'train.npz')

    npz = np.load(train_path, allow_pickle=True)
    print("=== NPZ keys ===")
    for k in npz.keys():
        v = npz[k]
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")

    # x_offsets 是关键：记录每个时间步相对于预测目标的偏移量
    if 'x_offsets' in npz:
        offsets = npz['x_offsets']
        # 可能是 object array 包装的 list
        if offsets.ndim == 0:
            offsets = offsets.item()
        offsets = np.array(offsets).flatten()
        
        print(f"\n=== x_offsets (共 {len(offsets)} 步) ===")
        print(f"  Raw: {offsets}")
        
        # 换算成实际时间
        print(f"\n=== 换算为实际时间 (30min 间隔) ===")
        print(f"  {'Step':>4s}  {'Offset':>7s}  {'Hours':>8s}  {'Days':>6s}  含义")
        print("-" * 65)
        
        prev_offset = None
        for i, off in enumerate(offsets):
            hours = off * 0.5  # 每个 offset 单位 = 30min
            days = hours / 24
            
            # 检测跳变
            jump_mark = ""
            if prev_offset is not None:
                gap = off - prev_offset
                if abs(gap) > 2:  # 跳了超过 1 小时
                    jump_mark = f"  ← 跳变! gap={gap} slots ({gap*0.5:.1f}h)"
            
            # 判断属于哪个时段
            if days < -2.5:
                period = "3天前"
            elif days < -1.5:
                period = "2天前"
            elif days < -0.5:
                period = "1天前"
            else:
                period = "近期"
            
            print(f"  {i:>4d}  {off:>7d}  {hours:>7.1f}h  {days:>5.1f}d  {period}{jump_mark}")
            prev_offset = off
        
        # 分组统计
        print(f"\n=== 分组统计 ===")
        offsets_hours = offsets * 0.5
        
        groups = {}
        for i, h in enumerate(offsets_hours):
            d = h / 24
            if d < -2.5:
                g = "3天前"
            elif d < -1.5:
                g = "2天前"
            elif d < -0.5:
                g = "1天前"
            else:
                g = "近期"
            groups.setdefault(g, []).append(i)
        
        for g, indices in groups.items():
            print(f"  {g}: {len(indices)} 步 (step {indices[0]}-{indices[-1]})")
    
    else:
        print("\n[WARNING] 没有找到 x_offsets，尝试从数据推断...")
        print("  可用的 keys:", list(npz.keys()))
    
    # 也看看 Y_offset
    if 'Y_offset' in npz:
        y_off = npz['Y_offset']
        if y_off.ndim == 0:
            y_off = y_off.item()
        y_off = np.array(y_off).flatten()
        print(f"\n=== Y_offset ===")
        print(f"  Raw: {y_off}")
        print(f"  含义: 预测目标在当前时刻之后 {y_off * 0.5} 小时")


if __name__ == '__main__':
    main()