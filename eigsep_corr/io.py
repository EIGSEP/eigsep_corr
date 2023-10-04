import json

# the immutable header of the file
HEADER = {
    "nfiles": 60,
    "dtype": "int32",
    "byteorder": ">",
    # "header_size": ,
    # "metadata_size": ,
}


class File:
    def __init__(self, fname, metadata):
        self.fname = fname
        self.data = {"header": HEADER, "metadata": metadata}
        self.max_cnt = None

    def add_data(self, data, cnt):
        self.data[f"{cnt}"] = data
        if self.max_cnt is None:  # this is the first file
            self.max_cnt = cnt + HEADER["nfiles"]

    def write(self):
        with open(self.fname, "w") as f:
            json.dump(self.data, f)

    def read_header(self):
        pass

    def read_data(self):
        # use header size to seek to the right place
        with open(self.fname, "r") as f:
            data = json.load(f)
        return data
