# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_all

# 收集必要的数据文件
datas = [
    ('config.json', '.'),
    ('best-seg.onnx', '.'),
    ('best_seg_DW.onnx', '.'),
    ('best_CM.onnx', '.'),
    ('ic4-gentl-u3v_x64.cti', '.'),
    ('MvImport', 'MvImport'),
    ('1.ico', '.'),
]

binaries = []
hiddenimports = [
    'serial.tools.list_ports',
    'FpPlc',
    'ultralytics.nn.modules',
    'ultralytics.nn.tasks',
]

# 收集 ultralytics, onnxruntime, cv2 的依赖
# 这些库通常包含大量二进制文件和 Python 脚本，PyInstaller 有时无法自动检测完整
for lib in ['ultralytics', 'onnxruntime', 'cv2']:
    tmp_ret = collect_all(lib)
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]

a = Analysis(
    ['main_new_hik_beifen5.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'pandas', 'scipy', 'notebook', 'ipython', 'anaconda-navigator',
        'IPython', 'jupyter', 'nbconvert', 'nbformat', 'qtconsole', 'traitlets', 'pygments',
        'openvino', 'openvino-telemetry', 'h5py',
        'tkinter',
        'ensurepip', 'venv', 'pip', 'wheel',
        'test', 'tests', 'testing', '_pytest', 'pytest', 'nose',
        'PyQt6', 'PySide2', 'PySide6',
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main_new_hik_CC_beifen3',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['1.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='dist_main_new_hik_beifen5.py',
)
