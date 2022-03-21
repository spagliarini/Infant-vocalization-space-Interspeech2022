# Infant-vocalization-space-Interspeech2022

## Pre-processing
1. Extraction of single vocalizations using MATLAB routine. Vocalizations are extracted based on the automatic LENA system. References:
> J. Gilkerson and J. A. Richards, “The lena natural language study,” Boulder, CO: LENA Foundation. Retrieved March, vol. 3, p. 2009, 2008.
 
> D. Xu, U. Yapanel, S. Gray, J. Gilkerson, J. Richards, and J. Hansen, “Signal processing for young child speech language development,” in First Workshop on Child, Computer and Interaction, 2008.

2. Extraction of MFCCs coefficients, velocity and acceleration using OpenSMILE. Resources and guidelines can be found at https://audeering.github.io/opensmile/. References:
> Florian Eyben, Martin Wöllmer, Björn Schuller: "openSMILE - The Munich Versatile and Fast Open-Source Audio Feature Extractor", Proc. ACM Multimedia (MM), ACM, Florence, Italy, ISBN 978-1-60558-933-6, pp. 1459-1462, 25.-29.10.2010.

3. Utils to run previous and following steps. The python file vocSpace_utils.py contains: 
* list: a function to create a list containing all the babies included in the study.
* executeOS: a function to create an automatic bash file useful to run the opensmile pre-processing.
* merge: a function needed for the validation part to merge the human re-labeled vocalization with the original automatic labels, and use the new labels to extract the vocalizations as in point 1. 

4. 

## Vocalizations space and statistical analysis
1. Compute the low-dimensional representation (UMPA, t-SNE, PCA). 
2. Compute statistical properties of the clusters across families and across time (age): python and R codes.
