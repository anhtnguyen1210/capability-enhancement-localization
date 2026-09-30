"""Audit working, staged and committed content plus anonymous commit metadata."""
import argparse
import re
import subprocess
from pathlib import Path


def audit(root, forbidden):
    def git(*args):
        return subprocess.check_output(['git','-C',str(root),*args])
    files=list(filter(None,git('ls-files','-z').decode().split('\0')))
    failures=[]
    scanned=set()
    checks=[r'/(?:home|Users)/[A-Za-z0-9_.-]+', r'/data/(?:user_data|locus|group_data)/',
            r'-----BEGIN (?:OPENSSH|RSA|EC|DSA|PRIVATE).*KEY-----',
            r'\b(?:ghp_|github_pat_|hf_)[A-Za-z0-9]{20,}',
            r'https?://[^\s/]+:[^\s/]+@']
    def scan(label, raw):
        import hashlib
        key=(label,hashlib.sha256(raw).hexdigest())
        if key in scanned:return
        scanned.add(key)
        try:text=raw.decode('utf-8')
        except UnicodeDecodeError:
            failures.append(f'{label}: unreviewed binary content');return
        if any(word.lower() in text.lower() or word.lower() in label.lower() for word in forbidden):
            failures.append(f'{label}: forbidden identity term')
        if any(re.search(pattern,text) for pattern in checks):
            failures.append(f'{label}: identity/secret pattern')
        emails=re.findall(r'[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',text)
        if any(email!='anonymous@example.invalid' for email in emails):
            failures.append(f'{label}: non-anonymous email')
    def path_check(name):
        if name.startswith(('.agent/','.agents/','.codex/','data/','runs/','.env','logs/','results/')):
            failures.append(f'{name}: private/generated file tracked')
    for name in files:
        path_check(name)
        path=root/name
        if path.is_symlink():
            failures.append(f'{name}: symbolic link');continue
        scan(name,path.read_bytes())
        scan(name,git('show',':'+name))
    commits=git('rev-list','--all').decode().splitlines()
    for commit in commits:
        scan('commit message',git('show','-s','--format=%B',commit))
        for row in git('ls-tree','-rz',commit).split(b'\0'):
            if not row:continue
            metadata,name=row.split(b'\t',1)
            mode,kind,oid=metadata.decode().split()
            name=name.decode()
            path_check(name)
            if mode!='100644' or kind!='blob':
                failures.append(f'{name}: unexpected object type/mode');continue
            scan(name,git('cat-file','blob',oid))
    identities=git('log','--all','--format=%an <%ae>%n%cn <%ce>').decode().splitlines()
    if any(x!='Anonymous Authors <anonymous@example.invalid>' for x in identities):
        failures.append('Non-anonymous author or committer in history')
    if failures:raise SystemExit('\n'.join(sorted(set(failures))))
    print(f'PASS: {len(files)} tracked files, index and {len(commits)} commits; anonymous metadata')
    print('This checks content, not hosting-account or review-link anonymity.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--forbid',action='append',default=[],help='Private identity string to reject; do not commit it')
    a=p.parse_args()
    audit(Path(__file__).resolve().parent,a.forbid)
