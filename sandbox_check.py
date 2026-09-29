"""Check passing/failing programs, time limits, and denied network/file writes."""
import argparse
from pathlib import Path
from runtime import score_programs


def check(image, output):
    fixtures = [
        ('pass', 'def add(a,b): return a+b', ['assert add(2,3)==5'], True),
        ('fail', 'def add(a,b): return a-b', ['assert add(2,3)==5'], False),
        ('timeout', 'while True: pass', ['assert True'], False),
        ('network', 'import socket\nsocket.socket(socket.AF_INET,socket.SOCK_STREAM)', ['assert True'], False),
        ('write', 'open("/tmp/forbidden-write", "w").write("x")', ['assert True'], False),
    ]
    rows = [{'sample_id':name,'test_setup_code':'','program':program,'test_list':tests}
            for name,program,tests,_ in fixtures]
    scored = score_programs(rows,output,image)
    if [r['correct'] for r in scored] != [f[3] for f in fixtures]:
        raise RuntimeError('Sandbox functional/isolation check failed')
    if scored[2]['execution']['execution_outcome'] != 'timeout':
        raise RuntimeError('Sandbox timeout check failed')
    print('Sandbox checks passed')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    check(a.image,a.output)
