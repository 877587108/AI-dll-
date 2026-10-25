import io
p = 'tools/hole_test.cpp'
s = io.open(p, encoding='utf-8-sig').read()
old = '"%8s %10s %10s %12s %8s\\n"'
new = '"%8s %10s %10s %12s %8s %8s %6s %6s\\n"'
if old in s:
    s = s.replace(old, new)
old2 = '"有墨比例", "断口数", "最长缺口°", "清px");'
new2 = '"有墨比例", "断口数", "最长缺口°", "清px", "断点分", "档位", "判定");'
if old2 in s:
    s = s.replace(old2, new2)
io.open(p, 'w', encoding='utf-8-sig', newline='\n').write(s)
print('ok')
