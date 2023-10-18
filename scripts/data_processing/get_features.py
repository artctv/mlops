import sys
import os
import io

if len(sys.argv) != 2:
    sys.stderr.write("Argument error\n")
    sys.exit(1)

f_input = sys.argv[1]
f_output = os.path.join("datasets", "stage1", "train.csv")
os.makedirs(os.path.join("datasets", "stage1"), exist_ok=True)

def process_data(fd_in, fd_out):
    fd_in.readline()
    for line in fd_in:
        line = line.rstrip("\n")
        line = line.split(',')
        p_surv = line[1]
        p_class = line[2]
        if line[4][0] == '"':
            p_sex = line[6]
            p_age = line[7]
        else:
            p_sex = line[5]
            p_age = line[6]
        fd_out.write(
            "{},{},{},{}\n".format(p_surv, p_class, p_sex, p_age)
        )

with io.open(f_input, encoding="utf8") as fd_in:
    with io.open(f_output, "w", encoding="utf8") as fd_out:
        process_data(fd_in, fd_out)