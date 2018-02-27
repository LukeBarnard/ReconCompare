ReconCompare
============

Introduction
------------

The scripts in this project provide the code and data that generated the results reported in our article "What can the annual 10Be solar activity reconstructions tell us about historic space weather", accepted for publication in Journal of Space Weather and Space Climate.

In this work we compare reconstructions of the heliospheric magnetic field derived from cosmogenic isotopes, sunspot records, and geomagnetic variability, along with known records of large space weather events. We interpret these results as evidence that there are small systematic differences between the cosmogenic isotope reconstructions in comparison to the geomagnetic and sunspot reconstructions. The nature of these reconstructions implies they may be due to the cosmic ray inversion procedure, and perhaps due to a bias resulting from large space weather events. For further details on the scientific background and results of this project, please see our article.

Code Dependencies
-----------------
All code is written in Python, and depends the Numpy, Pandas, SciPy, Matplotlib, os, and glob packages.

Getting started
---------------
This package is configured such that it should be simple for anyone with an up-to-date Python 2.7 distribution to re-run this analysis.

After downloading the full project (code and data files), open config.txt (in the code directory), and change the path given after "root," on the first line, to the path of this project on your system. Then run ReconCompareMain.py from either the command line or your IDE. Figures from the aritcle will be generated in the figures directory, and numerical results will printed to screen.

Data
----
All data is provided as csv or txt files, and is provided in the data subdirectory. The data used in this project were obtained from the following published articles and data repositories:

*   10Be Cosmogenic Isotope derived solar activity reconstructions: Supplementary material to McCrakcen and Beer 2015, Solar Physics https://dx.doi.org/10.1007/s11207-015-0777-x 
*   Sunspot derived solar activity reconstruction: Supplementary material to Owens et al. 2016 JGR https://dx.doi.org/10.1002/2016JA022529
*   Geomagnetically derived solar activity reconstructions: Supplementary material to Owens et al. 2016 JGR https://dx.doi.org/10.1002/2016JA022529
*   Daily Sunspot Count: Downloaded from WDC-SILSO, Royal Observatory of Belgium, Brussels, at http://www.sidc.be/silso/datafiles
*   aa geomagnetic index: Downloaded from ISGI at http://isgi.unistra.fr/indices_aa.php., and these data are provided under a CC-BY-NC lisence.

The data files are provided here for convenience, but please see the code and article for full citation data, and consult the original data providers regarding the licensing of these data.

Contact
--------

Please contact me with any questions at l.a.barnard@reading.ac.uk

Software License
-----------------
Copyright 2018 University of Reading

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
