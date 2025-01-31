# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('known_faces', 'known_faces'), ('templates', 'templates'), ('attendance.db', '.'), ('auth_database.db', '.'), ('users.db', '.'), ('db_startup.py', '.'), ('fine_tuned_classifier.pth', '.'), ('imports.py', '.'), ('load_post_check.py', '.'), ('model_loader.py', '.')],
    hiddenimports=['flask'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['starkbank'],
    noarchive=True,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [('v', None, 'OPTION')],
    name='app',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
