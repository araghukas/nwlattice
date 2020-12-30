# nwlattice
This package is for generating atom coordinate files to initialize structures in atomistic simulations of nanowires.

It's written to work with the `read_data` command in [LAMMPS](https://lammps.sandia.gov/). However, output files can be visualized directly in [OVITO](https://www.ovito.org/) (free version works OK) or [VMD](http://www.ks.uiuc.edu/Research/vmd/) (for example). The format of the file corresponds to what's called *xyz* in the LAMMPS docs. This is just a text file listing every atom as well as its type, coordinates, and velocity.

## Installation
Download or clone

    git clone https://github.com/araghukas/nwlattice.git
    
Navigate to the `nwlattice` directory, which contains `setup.py`, then

    pip install .
    
## Usage
Import the nanowire type of interest and instantiate it. Then call the `write_points(file_path=None)` instance method. If `file_path` is not specified, this will write a file at `'./TypeName_structure.data'`, with `'TypeName'` substituted for the actual class name. The resulting file can be imported directly by LAMMPS using, say,

    read_data ./TypeName_structure.data
    
to initialize atom locations and the simulation cell dimensions.

### print information
For a summary describing the nanowire classes, import the `get_info(type_name=None)` method:

    from nwlattice import get_info
    
Calling `get_info` without specifying `type_name` will print doc information for all implemented nanowire classes.

### writing data files
If you're after a data file representing, say, atoms in a GaAs nanowire that is 10 nm wide and 300 nm long, run something like:

    from nwlattice.nw import ZBPristine111
    
    scale = 0.56531  # GaAs lattice constant @ 300K (in nm)
    
    wire = ZBPristine111(scale, width=10, length=300)
    wire.write_points("~/Desktop/my_GaAs_nanowire.data")
    
**Keep in mind that** the number of atoms in the nanowire scales with the **square** of the width, so the output files (being uncompressed text) can quickly become several GB in size. 
However file sizes should be manageable for dimensions suitable to MD.

The argument set `scale`, `width`, and `length` is enough to instantiate the simplest nanowire classes. Other classes may additionally require `period` or `fraction` to be specified, if applicable. For example, a *faceted* twinning nanowire can be instantiated using

    wire = ZBTwinFaceted(0.56531, 10, 300, 20)
    
where the arguments specify `scale`, `width`, `length`, and `period`, respectively.

See the class docstrings in `nwlattice/nw.py` or use the `get_info` method mentioned above for instructions on instantiating with integer atom-wise dimension parameters. 

### extracting actual wire dimensions
Note that because the lattice is discrete, input dimensions such as width, length, and period will be approximated in general. The actual resulting dimensions are stored in the `size` attribute of each instance.

    actual_scale = wire.size.scale
    actual_width = wire.size.width
    actual_length = wire.size.length
    actial_period = wire.size.period  # error for not periodic nanowire types
    actual_area = wire.size.area

## Disclaimer
Please note that this project is a work in progress. Use at your own risk.
