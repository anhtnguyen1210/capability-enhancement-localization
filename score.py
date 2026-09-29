"""MBPP classifier; invoked only inside the isolated sandbox."""
import json
import os
import sys
from common import publish
from secure_execute import _secure_humaneval_execution


def main():
    if os.environ.get('LLMCOMP_HUMANEVAL_SANDBOX') != '1':
        raise RuntimeError('Scoring requires the sandbox launcher')
    rows = json.load(open(sys.argv[1]))
    records = []
    for row in rows:
        outcome = _secure_humaneval_execution(row['test_setup_code']+'\n'+row['program'],
                                              '\n'.join(row['test_list']), 3)
        if outcome['execution_outcome'] == 'sandbox_error':
            raise RuntimeError('Sandbox setup failed')
        records.append({**row, 'correct':outcome['execution_outcome']=='passed', 'execution':outcome})
    publish(sys.argv[2], {'records':records})


if __name__ == '__main__':
    main()
