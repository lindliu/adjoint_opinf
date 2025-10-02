#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 20:39:39 2025

@author: dliu
"""

import os
import re

# 要修改的文件夹路径
folder = "./"   # 改成你的目标路径

# 正则模式：匹配 sam 后跟数字
pattern = re.compile(r"(sam\d+)")

for filename in os.listdir(folder):
    old_path = os.path.join(folder, filename)

    if os.path.isfile(old_path):
        # 查找 "samXXXX"
        match = pattern.search(filename)
        if match:
            sam_str = match.group(1)
            # 在 samXXXX 后加 "_p75"
            new_filename = filename.replace(sam_str, sam_str + "_ratio0p75")
            new_path = os.path.join(folder, new_filename)

            # 重命名
            os.rename(old_path, new_path)
            print(f"{filename} -> {new_filename}")