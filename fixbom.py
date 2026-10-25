import io, sys, glob
targets = sys.argv[1:] if len(sys.argv) > 1 else glob.glob('tools/*.cpp')
for p in targets:
    raw = open(p, 'rb').read()
    if raw.startswith(b'\xef\xbb\xbf'):
        continue
    try:
        s = raw.decode('utf-8')
    except UnicodeDecodeError:
        print('skip(non-utf8)', p); continue
    if all(ord(c) < 128 for c in s):
        continue
    io.open(p, 'w', encoding='utf-8-sig', newline='\n').write(s)
    print('BOM+', p)
