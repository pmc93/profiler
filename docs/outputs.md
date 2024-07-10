# Output files
By default, Edcrop prints a log file and, for each simulation run, two files with daily and yearly water
balance results, respectively. The user is free to choose which results are printed. The user can also choose
to have some of the input data and simulation results plotted and saved in PNG files.

## The log file – 'edcrop.log'
Edcrop always prints a log file named edcrop.log. The log file summarizes the input of all the simulations
that Edcrop loops through during execution, and it lists all the error messages and warnings that Edcrop
sends.

## Print files
For each simulation, daily and yearly water balance results are saved in two files; one with daily results, the
other with yearly results. The name of each file consists of five parts separated by underscore,”_”:

&ensp;&ensp;&ensp;&ensp; %1_%2_%3_%4_%5

The first part, %1, is the climate case short name; %2 is the soil case short name; %3 is the vegetation case
short name; %4 is the model case short name; and %5 is “_wb.out” for daily output and “_y_wb.out” for
yearly output, respectively. The short names are taken from the edcrop.yaml input file as explained above
in Chapters 5.3 to 5.6. Hereby each filename becomes unique, and a file with results from a specific
simulation is identifiable from the filename.

The variables that can be printed are listed and explained in Table 13. By default, the printed variables are


&ensp;&ensp;&ensp;&ensp; **T P Ep I Ea Dsum**

For daily output, the date is also printed; for yearly output, only the year is also printed.
It is easy to change the list of output variables by, in the right place of the “Models” block of the
edcrop.yaml file, using the key “prlistd” for the daily output, or “prlisty” for the yearly output. For example,
putting this line in the right place of the “Models” block


&ensp;&ensp;&ensp;&ensp; **prlistd: P Ea Dsum**

instructs Edcrop to only print daily precipitation, actual evapotranspiration, and drainage from the sub
zone. During reading of the value list, only names recognized from Table 13 in Chapter 7.4 as valid
variables will be used to print output. If there is no recognition of valid variables from this list, or if ‘Date’ is
the only valid variable in the list, no daily output is printed.

The same rules apply for the key “prlisty” and the yearly output. For yearly output, ‘Date’ is not valid.
For the daily output, there is an alternative to setting the output variables by use of key “prlistd”. The
alternative is to use the key named “iprnd” in the “Models” block. This key can take any of four integer
values: 1, 2, 3, or 4. For increasing value, more variables will be printed: for “iprnd : 1”, only the default
variables are printed (in which case there is no need to set “iprnd”); for “iprnd : 4”, all variables will be
printed.Documentation of Edcrop

## Plot file

In the “Models” block of the edcrop.yaml file, by setting the key and value

&ensp;&ensp;&ensp;&ensp; plotseries: true

Edcrop will make two PNG files, each with plots of some input and simulated variables. The first file plots
precipitation (or sum of precipitation and irrigation), actual evapotranspiration, and drainage from the
subzone (named P, Ea, and Db in Table 13, respectively). The other file plots temperature, root depth, leaf
area, and crop coefficient (named T, zr, L, and kc in Table 13, respectively).
The name of the two PNG files have five parts separated by underscore,”_”:

&ensp;&ensp;&ensp;&ensp; %1_%2_%3_%4_%5

The first part, %1, is the climate case short name; %2 is the soil case short name; %3 is the vegetation case
short name; %4 is the model case short name; and %5 is “_P_Ea_Db.png” and “_T_zr_L_kc.png” for the two
files, respectively.

Figure 5 shows an example of plots from Edcrop.