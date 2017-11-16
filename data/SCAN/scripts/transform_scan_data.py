import os
rootdir = 'input'

def clean_data(rootdir, newroot):
    extensions = ('.txt')

    # create new root for cleaned data
    if not os.path.exists(newroot):
        os.mkdir(newroot)

    # walk over all subdirs and files
    for subdir, dirs, files in os.walk(rootdir):
        newsubdir = os.path.join(newroot, os.path.basename(subdir))
        if not os.path.exists(newsubdir):
            os.mkdir(newsubdir)
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext!= '' and ext in extensions:
                base = os.path.basename(file)
                orig_file = open(os.path.join(subdir, base), 'r')
                cleaned_file_path=os.path.join(newsubdir, base)
                cleaned_file=open(cleaned_file_path,'w')


                for line in orig_file:
                    line = line[3:]
                    line = line.replace(' OUT: ', '\t')
                    cleaned_file.write(line)
                cleaned_file.close()
                orig_file.close()

if __name__ == '__main__':
    clean_data('../../../data/SCAN', '../../../data/CLEANED-SCAN')