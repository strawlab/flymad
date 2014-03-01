#!/usr/bin/env python
import argparse
import os

def ensure_line_in_file( line, filename, create_ok=True ):
    if not os.path.exists(filename):
        if not create_ok:
            raise RuntimeError('the file %r does not exist, but will not create it.')
        else:
            fd = open(filename,mode='a+') # do not delete if just created
            fd.close()

    line_in_file = False
    with open(filename, mode='r+') as fd:
        for this_line in fd.readlines():
            this_line_rstrip = this_line.rstrip()
            if line==this_line_rstrip:
                line_in_file = True
                break

        if not line_in_file:
            fd.write(line + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('line', help='the line to be included')
    parser.add_argument('filename', help='the file in which to include it')

    args = parser.parse_args()

    ensure_line_in_file( args.line, args.filename )
