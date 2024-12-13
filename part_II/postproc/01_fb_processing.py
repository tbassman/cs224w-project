# import gzip
# import shutil

# ## STEP 0:
# ## https://stackoverflow.com/questions/31028815/how-to-unzip-gz-file-using-python

# with gzip.open("freebase-rdf-2015-08-09-00-01.gz", "rb") as f_in:
#     with open("freebase-rdf-2015-08-09-00-01.txt", "wb") as f_out:
#         shutil.copyfileobj(f_in, f_out)

import gzip
import shutil

## STEP 0:
## https://stackoverflow.com/questions/31028815/how-to-unzip-gz-file-using-python

num_lines_per_file = 250000

with gzip.open("freebase-rdf-2015-08-09-00-01.gz", "rb") as f_in:
    x = 0
    y = 0
    filename_out = f"freebase-rdf-2015-08-09-00-0{y}.txt"
    f_out = open(filename_out, "wb")
    for line in f_in:
        if x == num_lines_per_file:
            print(f"starting new file {y+1}")
            f_out.close()
            y += 1
            filename_out = f"freebase-rdf-2015-08-09-00-0{y}.txt"
            f_out = open(filename_out, "wb")
            x = 0
        if "type.object.name" in str(line) and "@en" in str(line):
            f_out.write(line)
            x += 1
f_out.close()
