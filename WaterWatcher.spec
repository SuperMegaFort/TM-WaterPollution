# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['standalone/web_wrapper.py'],
    pathex=['.'],
    binaries=[],
    datas=[('UI_V2/index.html', 'UI_V2'), ('UI_V2/style.css', 'UI_V2'), ('UI_V2/app.js', 'UI_V2'), ('pipeline', 'pipeline'), ('standalone/best_model.pth', 'standalone')],
    hiddenimports=['UI_V2.server', 'torch', 'torchvision', 'piexif', 'scipy.signal'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'pandas', 'IPython', 'notebook', 'tkinter', 'caffe2', 'tensorboard', 'triton', 'pyarrow', 'cv2', 'torch.test'],
    noarchive=False,
    optimize=0,
)
# windows
pyz = PYZ(a.pure)


exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='WaterWatcher',
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
)
