"""Audit tracked release content and commit author metadata; no network access."""
import argparse
import re
import subprocess
from pathlib import Path


def audit(root, forbidden):
    def git(*args):
        return subprocess.check_output(['git','-C',str(root),*args])
    files=git('ls-files','-z').decode().split('\0')
    failures=[]
    checks=[r'/(?:home|Users)/[A-Za-z0-9_.-]+', r'/data/(?:user_data|locus|group_data)/',
            r'-----BEGIN (?:OPENSSH|RSA|EC|DSA|PRIVATE).*KEY-----',
            r'\b(?:ghp_|github_pat_|hf_)[A-Za-z0-9]{20,}',
            r'https?://[^\s/]+:[^\s/]+@']
    for name in filter(None,files):
        path=root/name
        if path.is_symlink():
            failures.append(f'{name}: symbolic link')
            continue
        text=path.read_text()
        for word in forbidden:
            if word.lower() in text.lower() or word.lower() in name.lower():
                failures.append(f'{name}: forbidden identity term')
        for pattern in checks:
            if re.search(pattern,text):failures.append(f'{name}: identity/secret pattern')
        if name.startswith(('.agent/','.agents/','data/','runs/','.env')):
            failures.append(f'{name}: private/generated file tracked')
    identities=git('log','--all','--format=%an <%ae>%n%cn <%ce>').decode().splitlines()
    if any(x != 'Anonymous Authors <anonymous@example.invalid>' for x in identities):
        failures.append('Non-anonymous author or committer in history')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'PASS: {len(list(filter(None,files)))} tracked files; anonymous author/committer metadata')
    print('This checks content, not hosting-account or review-link anonymity.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--forbid',action='append',default=[],help='Private identity string to reject; do not commit it')
    a=p.parse_args()
    audit(Path(__file__).resolve().parent,a.forbid)
