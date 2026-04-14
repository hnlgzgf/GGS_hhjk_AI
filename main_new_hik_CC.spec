# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

# 收集必要的数据文件
datas = [
    ('config.json', '.'),
    ('best-seg.onnx', '.'),
    ('best_seg_DW.onnx', '.'),
    ('best_CM.onnx', '.'),
    ('ic4-gentl-u3v_x64.cti', '.'),
    ('MvImport', 'MvImport'),
]

binaries = []
hiddenimports = ['serial.tools.list_ports', 'FpPlc']

# 只收集必要的依赖，避免全量收集
tmp_ret = collect_all('ultralytics')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

a = Analysis(
    ['main_new_hik_CC.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'setuptools',
        'pkg_resources',
        'backports',
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
        'PIL',
        'numpy.random._examples',
        'unittest',
        'xmlrpc',
        'pydoc',
        'doctest',
        'argparse',
        'difflib',
        'pdb',
        'profile',
        'cProfile',
        'timeit',
        'trace',
        'py_compile',
        'compileall',
        'pickletools',
        'pyclbr',
        'tabnanny',
        'ensurepip',
        'venv',
        'pip',
        'wheel',
        'setuptools',
        'distutils',
        'test',
        'tests',
        'testing',
        '_pytest',
        'pytest',
        'nose',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main_new_hik_CC',
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
    name='main_new_hik_CC',
)
