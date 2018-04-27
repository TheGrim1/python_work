# Software package for ID01 @ ESRF
This readme is a work in progress.
Have a look at the [wiki](https://gitlab.esrf.fr/zatterin/id01sware/wikis/home)!

## Structure

- `/bin` contains ready to use applications:
    * `CamView`
    * `pscan_align`
    * `id01_microscope_cofm`
    * `id01_microscope_contrast`
    * `kmap_showroi`
    * `pscan_detector_average`
    * `pscan_live`
- `/examples` perhaps it contains examples;
- `/id01lib` is the python module;
- `/scripts` ?;
- `/tests` some tests.

## Dependencies

- h5py
- matplolib
- numpy
- scipy
- xrayutilities
- silx
- (PIL)
- (SpecClient)

## Simple installation

- Download package as zip or clone the git repository:
```
    git clone https://gitlab.esrf.fr/zatterin/id01sware.git
```


- Install package using pip or setup script. One of:
```
    python setup.py install
```
```
    python setup.py install --user # for the current user only
```


