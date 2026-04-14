# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main_new_hik_CC.py'],
    pathex=[],
    binaries=[],
    datas=[('best-seg.onnx', '.'), ('best_seg_DW.onnx', '.'), ('best_CM.onnx', '.'), ('config.json', '.'), ('MvImport', 'MvImport')],
    hiddenimports=['ultralytics'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='高更盛检测系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='高更盛检测系统',
)
