# Input Files

To use Edcrop, it is crucial to use consistent units for all input. Since all the default values 
set in Edcrop use millimeter for length and day for time, it is beneficial to use these as units. 
Otherwise, all required data and parameters need to be specified in the input in order to avoid use of default values set in the code.
All input to Edcrop, except the climate time series, is given in a required text file named edcrop.yaml. The
file naming uses the extension “yaml” because the file is structured as a YAML file, and it is read (loaded) by
the Python YAML module. The edcrop.yaml file must be available in the working directory when Edcrop is
run.

## Structure of input file 'edcrop.yaml'

The edcrop.yaml file is a text file where each line of content is either a blank line, a comment, a key, a key
and a value, or the continuation of a value if this needs (or is preferred) to span more than one line of text.

A comment line begins with the hash character,”#”.

A key is a text string which is succeeded by a colon, “:”. The colon is not part of the key.

A value can be:
- A text string – e.g. JB1.
- An integer number – e.g. 23.
- A floating point (real) number – e.g. 7.831, using dot (“.”) as decimal separator; or 3.52e-7 for an
exponential number.
- A date in the format %Y-%m-%d – e.g. 1998-08-31.
- A Boolean – i.e. either True or False.
- A list of either of the above – e.g. a list of three real numbers [3.2, 7.32, 7.5e-3].
- A dictionary – e.g. a dictionary with two entries {‘key_1’: 1.3, ‘key_2’: 2.7}, where an entry has a key
(a text string) and a value (here a real number, but it could be of any type).

A list begins by “[“, and ends by “]”. A comma separates entries.

A dictionary begins by “{“, and ends by “}”. A colon followed by a space must separate the key and the
value, and a comma must separate two entries.

The file information is further structured by line indentation. **It is a requirement to carry out line
indentation by use of space characters, not by use of tabs!** 

Using tabs will create an error message and program failure

Indentation is used to define a block of information; indented lines contain information belonging to the
block. Further indentation defines information belonging to a sub block.

Table 1 shows an example with an input block named “Models”. The block begins with the line with the key
“Models”, defining the block name to be “Models”. This line is succeeded by an indented line with the keyDocumentation of Edcrop
“M1” defining a sub block of information to “Models”. The following line with key “M2” defines a new sub
block to “Models”.

It is required to use exact same indentation for all sub blocks! Otherwise, the file loading fails with an error
message! The error message is likely to point to the place in edcrop.yaml causing the loading problem.
In Table 1, sub block “M2” has its own sub-sub-block with key “wbfunc” followed by a value, which is the
string “evacrop”.

Table 1 Example of input block in edcrop.yaml file.

```
#This is a comment – beginning with the hash character
Models:
M1:
M2:
wbfunc: evacrop
```

## How Edcrop uses the edcrop.yaml input during execution

If Edcrop loads the “Models” block with the sub blocks in Table 1, it will execute two model runs. First
Edcrop will make model run short-named “M1”: because there is no sub block to “M1:” Edcrop will run
using default setup and default parameter values; as default, Edcrop uses the conceptualization for
evapotranspiration and drainage described in Chapter 4. Second Edcrop will make model run short-named
“M2”: this has a sub block with key “wbfunc” and value “evacrop”, which causes execution to use the
Evacrop conceptualization for evapotranspiration and drainage described in Chapter 3.

Table 2 shows another example of an edcrop.yaml file containing three blocks, each with one or more sub
blocks.

The first block is named “Climates”. Its sub block has a key named “C1”, and this has its own sub block with
key “filename” and value (text string) “climate.dat”. This instructs Edcrop, when it executes, to read the
required climate data from a file named “climate.dat”. “C1” would typically be a short name that identifies
the climate station.

The second block is named “Soils”. It has two sub blocks named “JB1” and “JB7”, respectively. Neither of
the sub blocks have its own sub block(s). This instructs Edcrop to execute sequential simulation for each of
two default soil types named JB1 and JB2, respectively, without changing any soil parameter value from its
default value.

The third block is named “Crops”, having two sub blocks named “SB” and “WW”, respectively. These sub
blocks also do not have own sub blocks. This instructs Edcrop to execute simulation for two default crop
types named SB (spring barley) and WW (winter wheat), respectively, without changing any crop parameter
value from its default value.

The edcrop.yaml file in Table 2 thus instructs Edcrop to execute in total four simulations; one simulation for
each combination of two soils (JB1 and JB7) and two crops (SB and WW).Documentation of Edcrop

Table 2 Example with the three mandatory input blocks of edcrop.yaml file.
```
#This is an edcrop.yaml file with the following three mandatory blocks
#Block defining file with climate data set
Climates:
   C1:
      filename: climate.dat
#Block defining soil types to be simulated
Soils:
   JB1:
   JB7:
#Block defining crops to be simulated
Crops:
   SB:
   WW:
```

If the edcrop.yaml file contained both the “Models” block in Table 1 and the three blocks in Table 2, Edcrop
would execute in total eight simulations; a simulation for each combination of two model setups, two soils,
and two crops.

It is mandatory that the edcrop.yaml file contains a block for “Climates”, “Soils”, and “Crops”, respectively,
like in Table 2.

If the “Models” block is missing, Edcrop uses the default model setup when executing simulations.
In Edcrop, the edcrop.yaml file is loaded in the function named `read_inp_file()`. This also stores the
“Models”, “Climates”, “Soils”, and “Crops” information in respective Python dictionaries. These dictionaries
are used by Edcrop to change settings and parameter values from their default as explained below.

The following sub-chapters give more information about the input regarding model setup, climate data,
soils and crops. The sub-chapter titles and content follow the block names in the edcrop.yaml file, i.e.
“Models”, “Climates”, “Soils”, and “Crops”.

## “Models” input
The user should use the “Models” block of the edcrop.yaml file to specify model setup and model
parameter values only to the extent they need to deviate from their default. If the “Models” block is not
present, Edcrop uses the default model setup when executing simulations.

As illustrated in Table 1 and explained in Chapter 5.1, each sub block (indented block) of “Models” specifies
a model setup that will be executed when running Edcrop with the respective edcrop.yaml input file. The
key of the sub block gives the name of that model execution (e.g. M1 in Table 1), which is also used as
model case short name in the naming of output files from this execution (explained in Chapter 6).

A sub block to a sub block (twice-indented block) specifies the desired setting or value by a key and a value,
as for key “wbfunc” in Table 1. There can be several of such sub blocks to a sub block, if more than one
setting or parameter needs to deviate from its default.Documentation of Edcrop

In Chapter 7.1, Table 6 lists all the model settings or model parameter values that can be changed by the
edcrop.yaml “Models” input. The Table also gives the respective key and type of value to be specified in the
edcrop.yaml file. Table 7 shows a second example of “Models” block input of an edcrop.yaml file.

In Edcrop, default model settings and parameter values are set in `ModelParameters.initialize()`. Changing
default settings or values by using the edcrop.yaml input happens in `ModelParameters.read_initialize()`.
As default, Edcrop uses “wbfunc : ed”, the alternative water balance function described in chapter 4. To use
the water balance function from Evacrop instead, set key and value “wbfunc: evacrop”.

## “Climates” input
The mandatory “Climates” block of the edcrop.yaml file specifies what file(s) to read to obtain time series
for climate data, i.e. for daily temperature, precipitation, and reference evapotranspiration. If more than
one file is specified, Edcrop will execute simulation using each data set sequentially.

As illustrated in Table 3, each sub block (indented block) of “Climates” specifies a climate data input that
will be used to execute a simulation when running Edcrop. The key of the sub block gives the climate case
short name of that model execution (e.g. Clim_1 in Table 3), which is also used to name the output files
from this execution (explained in Chapter 6). The short name typically identifies the corresponding climate
station.

A sub block to the sub block (twice-indented block) MUST specify a set of climate data input by using the
key “filename” with value being a string, where the string is a valid filename containing the climate data
time series.

Another sub block may specify by key “dtformat” a value being a string defining the format of dates to be
read from the climate data file. In Table 3, Clim_2 contains this sub block. This instructs Edcrop that dates in
the climate file are in the format '%Y%m%d', which could for example be 20200128.

As for Clim_1 in Table 3, if a sub block with key “dtformat” is not specified, Edcrop will read the climate
data file using the default format '%Y-%m-%d', which could for example be 2020-01-28.

The climate data file is a CSV file, which is read by Edcrop using `pandas.read_csv()`.

Edcrop skips reading the first line of the file. This is because such a line often just names the input of the
column beneath. Edcrop uses its own internal naming.

The file must contain four columns of daily data in this order: date, temperature, precipitation, and
reference evapotranspiration.

During reading, the date column data are transformed by `pandas.to_datetime()` using the date format
mentioned above.

Edcrop checks that the succeeding dates in the file increase by 1 day. If this is not the case, Edcrop will not
use the file; instead, it writes a message to the screen and in the edcrop.log file, and continues with the
next “Climates” input block.

The beginning and end dates of the climate data determine the simulation period.
In Edcrop, a climate data file is read by `TimeSeries.read()`.Documentation of Edcrop


Table 3 Example of “Climates” input block of edcrop.yaml file.

```
Climates:
    Clim_1:
        filename: climate_station_1.dat
    Clim_2:
        filename: climate_station_2.dat
        dtformat: '%Y%m%d'
```

Table 4 Example of “Soils” input block of edcrop.yaml file. JB1a is a new soil type defined from one of the
default soil types, JB1.

```
Soils:
    JB1:
    JB2:
        kqr: 0.2
    JB1a:
        soiltype: JB1
        name: Very coarse sandy
        thf: [.12, .06, .06, .06]
        kqr: 1.2
```

## “Soils” input
The mandatory “Soils” block of the edcrop.yaml file specifies the soil types simulated during Edcrop
execution.

Table 4 shows an example of a “Soils” block specifying, by the first level of indented lines, that three soil
types should be simulated; the soil types are named JB1, JB2, and JB1a, respectively. JB1 and JB2 are two of
seven predefined soil types in Edcrop (see Chapter 7.2), while JB1a is a new soil type defined for the actual
simulation. The user is free to choose nay name (key) for a new soil type. During execution, Edcrop uses
JB1, JB2, or JB1a, respectively, as soil case short name in the naming of output files from that simulation
(see Chapter 6).

Because there are no second level indented lines following the JB1-line in Table 4, the JB1 soil is simulated
using entirely default parameter values set in Edcrop.

For the JB2 soil, a second-level indented line follows the JB2-line, having key “kqr” and value equal to 0.2.
This instructs Edcrop to simulate the JB2 soil with the drainage rate
kqr set equal to 0.2; for all remaining soil parameters Edcrop uses default values.

A new soil type needs to be defined from one of the predefined soil types. In Table 4, a new very coarse
sandy soil type, short-named JB1a, is defined from the predefined JB1 soil, which is a coarse sandy type
(see Chapter 7.2). The second-level indented line following the JB1a line, using “soiltype” as key and the
string “JB1” as value, instructs this. The following second-level indented lines instruct Edcrop to change
some soil parameter values from the default values of the JB1 soil.

As an alternative, a new soil type can be coded into Edcrop. This is done in `SoilParameters.initialize()`.
Thereby the soil type will be “predefined” in Edcrop.Documentation of Edcrop

In Chapter 7.2, Table 9 lists all the soil parameter values that can be changed by the edcrop.yaml “Soils”
input. The Table also gives the respective key and type of value to be specified in the edcrop.yaml file.
Table 10 shows a second example of “Soils” block input of an edcrop.yaml file.

As default, Edcrop uses the linear drainage model for all soil types. However, if “wbfunc: ed” in the
“Models” block, the nonlinear drainage model can be chosen instead by setting “soilmodel: mvg” as
illustrated in Table 10 (Chapter 7.2). Also, as default Edcrop uses 0
Kmp = , which means there is no macro
pore drainage.

## “Crops” input
The mandatory “Crops” block of the edcrop.yaml file specifies the vegetation types simulated during
Edcrop execution. The “Crops” block is structured, and used by Edcrop, as described above for the “Soils”
block.

Table 5 shows an example of a “Crops” block specifying, by the first level of indented lines, that three
vegetation types should be simulated; the types are short-named SB (spring barley), DF (deciduous forest),
and WR (winter rape), respectively. SB and DF are two of thirteen predefined vegetation types in Edcrop
(see Chapter 7.3), while WR is a new type defined for the actual simulation. (The user is free to choose nay
name (key) for a new vegetation type.) For SB and DF, during execution they will have some of their
parameter values or settings changed from the default. WR is defined from the predefined type, WW,
which is winter wheat, but dates of sow and harvest are changed. During execution, Edcrop uses SB, DF, or
WR, respectively, as vegetation case short name in the naming of output files from that simulation (see
Chapter 6).

In Table 5, notice that the year is 1900 in the dates for sow and harvest of WR and for “leaflife” of DF.
Edcrop does not use the year; it only uses month and day for every simulated year. Therefore, an arbitrary
year can be used for the value of these dates.

For permanent use, a new vegetation type can be coded into Edcrop and thereby be “predefined”. This is
done in `CropParameters.initialize()`.

In Chapter 7.3, Table 12 lists all the vegetation parameter values that can be changed by the edcrop.yaml
“Crops” input. The Table also gives the respective key and type of value to be specified in the edcrop.yaml
file.

In Edcrop, default vegetation parameter values and growth models are set in `CropParameters.initialize()`.
Changing default settings or values by using the edcrop.yaml input happens `CropParameters.read_initialize()`.Documentation of Edcrop

Table 5 Example of “Crops” input block of edcrop.yaml file. SB and DF are predefined vegetation types for
which Edcrop during execution changes some parameter values from the default. WR is a new
crop type defined from one of the predefined types, WW.
```
Crops:
   SB:
       cb: [0.1, 0.1, 0.2, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1]
       cr: 12.
       autoharvest: true
    DF:
        leaflife: [1900-04-15, 1900-06-01, 1900-09-15, 1900-10-15]
    WR:
       croptype: WW
       name: Winter rape
       sowdate: 1900-09-01
       harvestdate: 1900-07-20
```

## Winter season and irrigation season

In Edcrop, the winter season (when there is no plant growth) is specified by
ModelParameters.wintermonth, which is a list of monthly Boolean (True or False) values defining which
months belong to the winter season. In Edcrop, the winter months are set to be November, December,
January and February. This can only be changed inside the code by modifying the
ModelParameters.wintermonth. (It has not been thoroughly checked that change of this setting to
Southern Hemisphere conditions works correctly. Try for yourself, and let me know if there are problems
with the code.)

Similarly, the irrigation season is specified by ModelParameters.irrigationmonth, which is a list of monthly
Boolean (True or False) values defining which months belong to the irrigation season. In Edcrop, the
irrigation months are set to be May, June and July. This can only be changed inside the code by modifying
the ModelParameters.irrigationmonth list.Documentation of Edcrop
