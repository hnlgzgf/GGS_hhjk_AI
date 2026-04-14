# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main_new2.py'],
    pathex=[],
    binaries=[],
    datas=[('best-seg.onnx', '.'), ('best_seg_DW.onnx', '.'), ('ic4-gentl-u3v_x64.cti', '.'), ('1.ico', '.')],
    hiddenimports=['imagingcontrol4', 'ultralytics', 'cv2', 'torch'],
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
    a.binaries,
    a.datas,
    [],
    name='main_new2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['1.ico'],
)
