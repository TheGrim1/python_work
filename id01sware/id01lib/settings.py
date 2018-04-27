import numpy as np




edf_header_types = {
    'ByteOrder': bytes,
    'DataType': bytes,
    'Dim_1': int,
    'Dim_2': int,
    #'HeaderID': bytes, # do not use it according to PB
    'Image': int,
    'Offset_1': int,
    'Offset_2': int,
    'Size': int,
    'BSize_1': int,
    'BSize_2': int,
    'ExposureTime': float,
    'Title': bytes,
    'TitleBody': bytes,
    'acq_frame_nb': int,
    'time': np.string_, # fixed length str
    'time_of_day': float,
    'time_of_frame': float
    }



detector_aliases = { # only those where x!=y
        "eiger2M":"ei2m",
    }


